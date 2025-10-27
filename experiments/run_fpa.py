# scripts/run_fpa.py
# FPA para selección de características (subset 5–9), con fitness en muestra para acelerar.

from pathlib import Path
import sys, json, time
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from fpa_selector import FPASelector
from fitness import evaluate_subset

# -------- Spark --------
spark = (
    SparkSession.builder
    .appName("FPA_FS")
    .master("local[*]")
    .config("spark.driver.memory", "6g")
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

train = train.toDF(*[str(c) for c in train.columns])
test  = test.toDF(*[str(c) for c in test.columns])

label_col = "Target" if "Target" in train.columns else "Label"
drop_cols = {"Label", "Target"}
feature_cols = [c for c in train.columns if c not in drop_cols]

def cast_clean(df, cols):
    for c in cols:
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

# -------- Sample para fitness (acelera MUCHO) --------
SMALL_FRAC = 0.18
train_small = train.sample(False, SMALL_FRAC, seed=13).cache()
test_small  = test.sample(False, SMALL_FRAC, seed=13).cache()
_ = train_small.count(); _ = test_small.count()  # materializa

# -------- Parámetros “paper-like” + ligeros --------
MAX_DEPTH = 12
LAM = 0.01
K_MIN, K_MAX = 5, 9

warm_candidates = [
    "Dst Port", "Fwd IAT Min", "Bwd Pkts/s", "Fwd Seg Size Min", "Fwd Act Data Pkts",
    "Fwd Pkt Len Min", "Flow Pkts/s", "ECE Flag Cnt", "ACK Flag Cnt"
]
warm_mask = [1 if f in warm_candidates else 0 for f in feature_cols]
priority = {name: i for i, name in enumerate(warm_candidates)}

def sort_by_priority(idxs):
    return sorted(
        idxs,
        key=lambda j: (feature_cols[j] not in priority, priority.get(feature_cols[j], 10_000), j)
    )

def repair_fn(mask):
    m = mask[:]
    ones = [i for i, v in enumerate(m) if v == 1]
    zeros = [i for i, v in enumerate(m) if v == 0]

    if len(ones) > K_MAX:
        keep = sort_by_priority(ones)[:K_MAX]
        m = [1 if i in keep else 0 for i in range(len(m))]
        ones = keep
        zeros = [i for i in range(len(m)) if m[i] == 0]

    if len(ones) < max(1, K_MIN):
        need = max(1, K_MIN) - len(ones)
        warm_zero = sort_by_priority([i for i in zeros if feature_cols[i] in priority])
        take = warm_zero[:need]
        if len(take) < need:
            rest = [i for i in zeros if i not in take]
            take += rest[:(need - len(take))]
        for i in take:
            m[i] = 1
    if sum(m) == 0:
        best = sort_by_priority(list(range(len(m))))[0]
        m[best] = 1
    return m

# --- fitness en MUESTRA (usa ORDEN POSICIONAL correcto) ---
def fitness_wrapper(mask):
    return evaluate_subset(
        train_small,        # train_df reducido
        test_small,         # test_df reducido
        feature_cols,       # feature_names
        mask,               # máscara
        label_col,          # label (5º parámetro POSICIONAL)
        max_depth=MAX_DEPTH,
        lam=LAM
    )

# -------- Instanciar FPA con parámetros ligeros + early-stop --------
fpa = FPASelector(
    pop_size=10,
    iters=10,
    p_global=0.8,
    levy_beta=1.5,
    step_scale=0.25,
    seed=13,
    elitism=1,
    patience=4,          # corta si no mejora 4 iter seguidas
    max_evals=400        # tope duro de evaluaciones
)

# -------- Búsqueda --------
t0 = time.time()
best_mask, info = fpa.fit_select(
    eval_fn=fitness_wrapper,
    feature_names=feature_cols,
    init_mask=warm_mask,
    lam=LAM,
    max_depth=MAX_DEPTH,
    repair_fn=repair_fn
)
elapsed_search = time.time() - t0
selected_cols = [f for f, v in zip(feature_cols, best_mask) if v == 1]

# -------- Reentrenamiento final con TODO el dataset --------
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

# -------- Guardar --------
(out_dir / "fpa_selected_features.txt").write_text("\n".join(selected_cols))
with open(out_dir / "fpa_metrics.json", "w") as f:
    json.dump({
        "subset_size": len(selected_cols),
        "selected_features": selected_cols,
        "metrics": metrics,
        "objective": info.get("best_obj", None) if isinstance(info, dict) else None,
        "search_info": {
            "history": info.get("history", []) if isinstance(info, dict) else [],
            "elapsed_search_sec": elapsed_search,
            "elapsed_finaltrain_sec": elapsed_final,
            "iters": info.get("iters", None),
            "evaluations": info.get("evaluations", None),
            "stopped_early": info.get("stopped_early", None)
        }
    }, f, indent=2)

print("✅ FPA terminado (ligero)")
print(f"Subset |S| = {len(selected_cols)} / {len(feature_cols)}")
print("Metrics:", metrics)
print(f"Tiempo búsqueda: {elapsed_search:.2f}s | Entrenamiento final: {elapsed_final:.2f}s")
print("Salidas:\n - data/results/fpa_selected_features.txt\n - data/results/fpa_metrics.json")