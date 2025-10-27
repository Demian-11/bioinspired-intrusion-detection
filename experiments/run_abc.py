# scripts/run_abc.py
# Orquesta: carga data/processed, corre ABC (búsqueda rápida) y reentrena full.

from pathlib import Path
import sys, json, time
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# --- permitir imports locales desde ./scripts ---
THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from abc_selector import ABCSelector
from fitness import evaluate_subset

# -------- Spark --------
spark = (
    SparkSession.builder
    .appName("ABC_FS")
    .master("local[*]")
    .config("spark.driver.memory", "6g")              # súbelo a "8g" si tu Mac aguanta
    .config("spark.sql.shuffle.partitions", "50")
    .config("spark.default.parallelism", "8")
    .getOrCreate()
)

# -------- Paths --------
train_p = "data/processed/train.parquet"
test_p  = "data/processed/test.parquet"
out_dir = Path("data/results")
out_dir.mkdir(parents=True, exist_ok=True)

# -------- Datos --------
train = spark.read.parquet(train_p)
test  = spark.read.parquet(test_p)

label_col = "Target" if "Target" in train.columns else "Label"
drop_cols = {"Label", "Target"}
feature_cols = [c for c in train.columns if c not in drop_cols]

# -------- ABC (parámetros ligeros para no tardar/romper memoria) --------
abc = ABCSelector(
    num_bees=8,      # antes 20
    max_iters=10,    # antes 50
    limit=5,         # antes 10
    lam=0.01,
    seed=13
)

# -------- Búsqueda acelerada: muestreo + árbol somero --------
train_small = train.sample(False, 0.20, seed=13)
test_small  = test.sample(False, 0.20, seed=13)
# materializa para evitar recálculo del sample
_ = train_small.count(); _ = test_small.count()

t0 = time.time()
best_mask, info = abc.fit_select(
    eval_fn=evaluate_subset,
    train_df=train_small,           # usar muestras en la búsqueda
    test_df=test_small,
    feature_names=feature_cols,
    max_depth=4                     # árbol somero para buscar rápido
)
elapsed_search = time.time() - t0

selected_cols = [f for f, m in zip(feature_cols, best_mask) if m == 1]

# -------- Re-entrenar final con TODO (fidelidad al paper) --------
def cast_clean(df, cols):
    for c in cols:
        df = df.withColumn(c, F.col(c).cast("double"))
        df = df.withColumn(
            c,
            F.when(~(F.isnan(F.col(c)) | F.col(c).isNull() |
                     (F.col(c) == float("inf")) | (F.col(c) == float("-inf"))),
                   F.col(c)).otherwise(F.lit(0.0))
        )
    return df

train2 = cast_clean(train, selected_cols)
test2  = cast_clean(test,  selected_cols)

va = VectorAssembler(inputCols=selected_cols, outputCol="features", handleInvalid="keep")
train3 = va.transform(train2)
test3  = va.transform(test2)

idx = StringIndexer(inputCol=label_col, outputCol="label", handleInvalid="keep").fit(train3)
train4 = idx.transform(train3)
test4  = idx.transform(test3)

dt = DecisionTreeClassifier(featuresCol="features", labelCol="label",
                            impurity="entropy", maxDepth=12, seed=13)
t1 = time.time()
model = dt.fit(train4)
pred  = model.transform(test4)
elapsed_final = time.time() - t1

metrics = {}
for metric in ["accuracy", "f1", "weightedPrecision", "weightedRecall"]:
    ev = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName=metric)
    metrics[metric] = ev.evaluate(pred)

# -------- Guardar resultados --------
(out_dir / "abc_selected_features.txt").write_text("\n".join(selected_cols))
with open(out_dir / "abc_metrics.json", "w") as f:
    json.dump({
        "metrics": metrics,
        "best_obj": info["best_obj"],
        "subset_size": len(selected_cols),
        "total_features": len(feature_cols),
        "elapsed_search_sec": elapsed_search,
        "elapsed_finaltrain_sec": elapsed_final,
        "history": info["history"]
    }, f, indent=2)

print("✅ ABC terminado")
print(f"Subset |S| = {len(selected_cols)} / {len(feature_cols)}")
print("Metrics:", metrics)
print(f"Tiempo búsqueda (ABC): {elapsed_search:.2f}s | Entrenamiento final: {elapsed_final:.2f}s")
print("Salidas:")
print(" - data/results/abc_selected_features.txt")
print(" - data/results/abc_metrics.json")