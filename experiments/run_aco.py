# scripts/run_aco.py
# ACO para selección de características (optimiz. ligera: sample + early stop)
from pathlib import Path
import sys, json, time
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# ---- imports locales ----
THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from src.aco_selector import ACOSelector
from src.fitness import evaluate_subset

# -------- Spark --------
spark = (
    SparkSession.builder
    .appName("ACO_FS")
    .master("local[*]")
    .config("spark.driver.memory", "6g")
    .config("spark.sql.shuffle.partitions", "30")
    .config("spark.default.parallelism", "6")
    .config("spark.ui.showConsoleProgress", "false")
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

# --- limpiar numéricos ---
def cast_clean(df, cols):
    for c in cols:
        if c in df.columns:
            df = df.withColumn(c, F.col(c).cast("double"))
            df = df.withColumn(
                c,
                F.when(
                    ~(F.isnan(F.col(c)) | F.col(c).isNull() |
                      (F.col(c) == float("inf")) | (F.col(c) == float("-inf"))),
                    F.col(c)
                ).otherwise(F.lit(0.0))
            )
    return df

train = cast_clean(train, feature_cols)
test  = cast_clean(test,  feature_cols)

# -------- Parámetros --------
MAX_DEPTH = 12
MAX_DEPTH_SEARCH = 6     # árbol más ligero para búsqueda
LAM = 0.01
K_MIN = 5
K_MAX = 10

# -------- Subsample para búsqueda (15%) --------
train_s = train.sample(False, 0.15, seed=13)
test_s  = test.sample(False, 0.15, seed=13)
_ = train_s.count(); _ = test_s.count()

# Warm start
warm_candidates = [
    "Dst Port", "Fwd IAT Min", "Bwd Pkts/s", "Fwd Seg Size Min", "Fwd Act Data Pkts",
    "Fwd IAT Max", "Flow IAT Min", "Fwd Pkts/s", "Fwd Header Len", "RST Flag Cnt"
]
warm_mask = [1 if f in warm_candidates else 0 for f in feature_cols]

priority = {name: i for i, name in enumerate(warm_candidates)}
def sort_by_priority(idxs):
    return sorted(idxs,
        key=lambda j: (feature_cols[j] not in priority,
                       priority.get(feature_cols[j], 10_000), j)
    )

def repair_fn(mask):
    m = mask[:]
    ones = [i for i, v in enumerate(m) if v == 1]
    zeros = [i for i, v in enumerate(m) if v == 0]

    if len(ones) > K_MAX:
        keep = sort_by_priority(ones)[:K_MAX]
        m = [1 if i in keep else 0 for i in range(len(m))]
    elif len(ones) < K_MIN:
        need = K_MIN - len(ones)
        warm_zero = [i for i in zeros if feature_cols[i] in priority]
        take = sort_by_priority(warm_zero)[:need]
        if len(take) < need:
            rest = [i for i in zeros if i not in take]
            take += rest[:(need - len(take))]
        for i in take: m[i] = 1
    if sum(m) == 0:
        m[sort_by_priority(range(len(m)))[0]] = 1
    return m

# -------- Wrapper fitness (usa sample + profundidad reducida) --------
def fitness_wrapper(mask):
    return evaluate_subset(train_s, test_s, feature_cols, mask,
                           label_col=label_col, max_depth=MAX_DEPTH_SEARCH, lam=LAM)

# -------- Instanciar ACO optimizado --------
aco = ACOSelector(
    num_ants=10,
    iters=8,
    alpha=1.0,
    beta=0.0,
    rho=0.25,
    q=1.0,
    seed=13,
    max_evals=300,   # nuevo
    patience=6       # nuevo
)

# -------- Búsqueda --------
t0 = time.time()
best_mask, info = aco.fit_select(
    eval_fn=fitness_wrapper,
    feature_names=feature_cols,
    init_mask=warm_mask,
    lam=LAM,
    max_depth=MAX_DEPTH,
    repair_fn=repair_fn,
    k_min=K_MIN,
    k_max=K_MAX
)
elapsed_search = time.time() - t0
selected_cols = [f for f, v in zip(feature_cols, best_mask) if v == 1]

# -------- Reentrenamiento final completo --------
va = VectorAssembler(inputCols=selected_cols, outputCol="features", handleInvalid="keep")
train_va = va.transform(train)
test_va  = va.transform(test)

idx = StringIndexer(inputCol=label_col, outputCol="label", handleInvalid="keep").fit(train_va)
train_idx = idx.transform(train_va)
test_idx  = idx.transform(test_va)

dt = DecisionTreeClassifier(featuresCol="features", labelCol="label",
                            impurity="entropy", maxDepth=MAX_DEPTH, seed=13)
t1 = time.time()
model = dt.fit(train_idx)
pred  = model.transform(test_idx)
elapsed_final = time.time() - t1

metrics = {}
for metric in ["accuracy", "f1", "weightedPrecision", "weightedRecall"]:
    ev = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName=metric)
    metrics[metric] = ev.evaluate(pred)

# -------- Guardado --------
(out_dir / "aco_selected_features.txt").write_text("\n".join(selected_cols))
with open(out_dir / "aco_metrics.json", "w") as f:
    json.dump({
        "subset_size": len(selected_cols),
        "selected_features": selected_cols,
        "metrics": metrics,
        "objective": info.get("best_obj", None),
        "search_info": {
            "history": info.get("history", []),
            "elapsed_search_sec": elapsed_search,
            "elapsed_finaltrain_sec": elapsed_final,
            "iters": info.get("iters", None),
            "evaluations": info.get("evaluations", None)
        }
    }, f, indent=2)

print("✅ ACO terminado (optimiz. ligera)")
print(f"Subset |S| = {len(selected_cols)} / {len(feature_cols)}")
print("Metrics:", metrics)
print(f"Tiempo búsqueda: {elapsed_search:.2f}s | Entrenamiento final: {elapsed_final:.2f}s")
print("Salidas:")
print(" - data/results/aco_selected_features.txt")
print(" - data/results/aco_metrics.json")