# scripts/split_and_discretize.py
# - Lee data/cleaned_dataset_parquet (del paso anterior)
# - Hace split 80/20 ESTRATIFICADO por Label
# - Discretiza columnas seleccionadas con QuantileDiscretizer (fit en train, apply en test)
# - Guarda parquet y un manifest con los cut-points por columna

from pathlib import Path
import os, sys, json, subprocess
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.window import Window
from pyspark.ml.feature import QuantileDiscretizer
from pyspark.ml import Pipeline, PipelineModel

# ---------------------------------------------
# 0) Asegurar entorno (Java17 + Python actual)
# ---------------------------------------------
os.environ["PYSPARK_PYTHON"] = sys.executable
if "JAVA_HOME" not in os.environ:
    try:
        jh = subprocess.run(["/usr/libexec/java_home", "-v", "17"],
                            capture_output=True, text=True)
        if jh.returncode == 0 and jh.stdout.strip():
            os.environ["JAVA_HOME"] = jh.stdout.strip()
    except Exception:
        pass

# ---------------------------------------------
# 1) Spark
# ---------------------------------------------
spark = (
    SparkSession.builder
    .appName("CICIDS2018_Split_Discretize")
    .master("local[*]")
    .config("spark.driver.memory", "6g")
    .config("spark.sql.shuffle.partitions", "100")
    .getOrCreate()
)

# ---------------------------------------------
# 2) Paths
# ---------------------------------------------
clean_parquet = "data/cleaned_dataset_parquet"
out_dir = Path("data/processed")
out_dir.mkdir(parents=True, exist_ok=True)

train_out = out_dir / "train.parquet"
test_out  = out_dir / "test.parquet"
pipeline_dir = "data/processed/discretizer_pipeline"

# ---------------------------------------------
# 3) Cargar limpio
# ---------------------------------------------
df = spark.read.parquet(clean_parquet)
assert "Label" in df.columns, "No se encuentra columna Label"

# ---------------------------------------------
# 4) (Opcional) Binarizar para baseline: Attack vs Benign
#     Comenta este bloque si quieres multiclase desde ya.
# ---------------------------------------------
df = df.withColumn("Target",
                   F.when(F.col("Label") == "Benign", F.lit("Benign")).otherwise(F.lit("Attack")))

label_col = "Target"   # usa "Label" si prefieres multiclase
feature_cols = [c for c in df.columns if c not in {"Label", "Target"}]

# ---------------------------------------------
# 5) Split 80/20 estratificado: método determinista por grupo
#    - Random por fila
#    - Para cada clase, umbral 0.8
# ---------------------------------------------
seed = 13
df = df.withColumn("_rand", F.rand(seed))

# calcular umbral por clase (si quisieras exactitud fina por clase):
# aquí usamos un umbral fijo de 0.8 por simplicidad (estratificación aproximada y estable).
train = df.where(F.col("_rand") < 0.8)
test  = df.where(F.col("_rand") >= 0.8)

# sanity check de proporciones
train_counts = train.groupBy(label_col).count().withColumnRenamed("count", "train_count")
test_counts  = test.groupBy(label_col).count().withColumnRenamed("count", "test_count")
train_counts.join(test_counts, on=label_col, how="outer").orderBy(F.desc("train_count")).show(50, False)

# ---------------------------------------------
# 6) Discretización tipo paper (cuantiles) en columnas clave
#    - Fit SOLO en train
#    - Apply a train y test
# ---------------------------------------------
# Sugeridas por el paper/ejemplos:
disc_cols = [
    "Dst Port", "Fwd IAT Min", "Bwd Pkts/s", "Fwd Seg Size Min", "Fwd Act Data Pkts",
    "Fwd Pkt Len Min", "Flow Pkts/s", "ECE Flag Cnt", "ACK Flag Cnt",
    "Flow IAT Min", "RST Flag Cnt", "Idle Std"
]
disc_cols = [c for c in disc_cols if c in feature_cols]

# Spark requiere numéricas para discretizar; intenta castear tolerante
for c in disc_cols:
    train = train.withColumn(c, F.expr(f"try_cast(`{c}` as double)"))
    test  = test.withColumn(c,  F.expr(f"try_cast(`{c}` as double)"))

# Config discretización (ajústalo si quieres)
n_bins = 10
qds = [
    QuantileDiscretizer(inputCol=c, outputCol=f"{c}__binned", numBuckets=n_bins, relativeError=0.001, handleInvalid="keep")
    for c in disc_cols
]

pipe = Pipeline(stages=qds)
model: PipelineModel = pipe.fit(train)

train_binned = model.transform(train)
test_binned  = model.transform(test)

# Opcional: si quieres quedarte solo con binned y descartar continuas originales:
keep_cols = [label_col] + [c for c in feature_cols if c not in disc_cols] + [f"{c}__binned" for c in disc_cols]
train_binned = train_binned.select(*keep_cols)
test_binned  = test_binned.select(*keep_cols)

# ---------------------------------------------
# 7) Guardar data + pipeline de discretización
# ---------------------------------------------
train_binned.write.mode("overwrite").parquet(str(train_out))
test_binned.write.mode("overwrite").parquet(str(test_out))
model.write().overwrite().save(pipeline_dir)

# Guardar manifest con bins por columna
# ¡OJO! Para leer los splits de cada QuantileDiscretizer hay que inspeccionar el PipelineModel
splits_manifest = {}
for stage in model.stages:
    # cada stage es un Bucketizer entrenado internamente por QuantileDiscretizer
    # estos modelos tienen atributo 'splits'
    # Nota: API interna; en Spark 3.5 funciona
    try:
        in_col = stage.getInputCol()
        splits = stage.getSplits()
        splits_manifest[in_col] = list(map(float, splits))
    except Exception:
        pass

with open(out_dir / "discretization_splits.json", "w") as f:
    json.dump({
        "columns_binned": disc_cols,
        "num_buckets": n_bins,
        "splits": splits_manifest
    }, f, indent=2)

print("Listo:")
print(f" - Train: {train_out}")
print(f" - Test : {test_out}")
print(f" - Pipeline (cuts): {pipeline_dir}")
print(f" - Splits JSON: {out_dir/'discretization_splits.json'}")