from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pathlib import Path

# ============================================
# 1. Configuración de Spark
# ============================================
spark = (
    SparkSession.builder
    .appName("BaselineTree")
    .master("local[*]")
    .config("spark.driver.memory", "6g")
    .config("spark.sql.shuffle.partitions", "100")
    .getOrCreate()
)

# ============================================
# 2. Paths
# ============================================
train_p = "data/processed/train.parquet"
test_p  = "data/processed/test.parquet"

train = spark.read.parquet(train_p)
test  = spark.read.parquet(test_p)

# ============================================
# 3. Preparación de columnas
# ============================================
label_col = "Target" if "Target" in train.columns else "Label"
drop_cols = {"Label", "Target"}
feature_cols = [c for c in train.columns if c not in drop_cols]

# ============================================
# 4. CAST a double para todas las features
# ============================================
for c in feature_cols:
    train = train.withColumn(c, F.col(c).cast("double"))
    test  = test.withColumn(c, F.col(c).cast("double"))

# ============================================
# 5. Limpieza de NaN e infinitos
# ============================================
def clean_invalid(df):
    """Reemplaza NaN e Inf por 0 (necesario para Spark ML)."""
    for c in feature_cols:
        df = df.withColumn(
            c,
            F.when(~(F.isnan(F.col(c)) | F.col(c).isNull() | (F.col(c) == float("inf")) | (F.col(c) == float("-inf"))),
                   F.col(c)).otherwise(F.lit(0.0))
        )
    return df

train = clean_invalid(train)
test = clean_invalid(test)

# ============================================
# 6. Vectorización
# ============================================
va = VectorAssembler(inputCols=feature_cols, outputCol="features", handleInvalid="keep")
train2 = va.transform(train)
test2  = va.transform(test)

# ============================================
# 7. Indexar etiqueta
# ============================================
idx = StringIndexer(inputCol=label_col, outputCol="label", handleInvalid="keep").fit(train2)
train3 = idx.transform(train2)
test3  = idx.transform(test2)

# ============================================
# 8. Entrenamiento del árbol
# ============================================
dt = DecisionTreeClassifier(
    featuresCol="features",
    labelCol="label",
    impurity="entropy",
    maxDepth=12
)

model = dt.fit(train3)
pred  = model.transform(test3)

# ============================================
# 9. Métricas
# ============================================
metrics = {}
for metric in ["accuracy", "f1", "weightedPrecision", "weightedRecall"]:
    ev = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName=metric)
    metrics[metric] = ev.evaluate(pred)

print("✅ Baseline metrics:", metrics)

# ============================================
# 10. Guardar resultados
# ============================================
out_dir = Path("data/results")
out_dir.mkdir(parents=True, exist_ok=True)

with open(out_dir / "baseline_metrics.txt", "w") as f:
    for k, v in metrics.items():
        f.write(f"{k}: {v:.6f}\n")

# Top-20 importancias
imps = list(zip(feature_cols, model.featureImportances.toArray()))
imps_sorted = sorted(imps, key=lambda x: x[1], reverse=True)[:20]
with open(out_dir / "baseline_feature_importance.txt", "w") as f:
    for name, val in imps_sorted:
        f.write(f"{name}: {val:.6f}\n")

print("📊 Guardado en:")
print(" - data/results/baseline_metrics.txt")
print(" - data/results/baseline_feature_importance.txt")