# scripts/run_pso_tight.py
# PSO con tope DURO: repara la máscara para que nunca evalúe > K_CAP features.
# Además usa un ranking barato (importancias de árbol en 3%) para elegir las K mejores.

from pathlib import Path
import sys, json, time
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# --- imports locales ---
THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from pso_selector import PSOSelector
from fitness import evaluate_subset

# -------- Spark muy ligero --------
spark = (
    SparkSession.builder
    .appName("PSO_FS_TIGHT_HARDCAP")
    .master("local[1]")
    .config("spark.driver.memory", "6g")
    .config("spark.sql.shuffle.partitions", "8")
    .config("spark.default.parallelism", "1")
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
D = len(feature_cols)

# -------- Muestreo chico para búsqueda y ranking --------
train_small = train.sample(False, 0.03, seed=13)
test_small  = test.sample(False, 0.03, seed=13)

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

# -------- Ranking barato (importancia de árbol en 3%) --------
def cheap_ranking(cols):
    df = cast_clean(train_small.select(*([label_col] + cols)), cols)
    va = VectorAssembler(inputCols=cols, outputCol="features", handleInvalid="keep")
    tr2 = va.transform(df)
    idx = StringIndexer(inputCol=label_col, outputCol="label", handleInvalid="keep").fit(tr2)
    tr3 = idx.transform(tr2)
    dt = DecisionTreeClassifier(featuresCol="features", labelCol="label",
                                impurity="entropy", maxDepth=4, seed=13)
    model = dt.fit(tr3)
    imps = list(zip(cols, model.featureImportances.toArray()))
    imps_sorted = sorted(imps, key=lambda x: x[1], reverse=True)
    # Si alguna importancia viene 0, igual aparece al final; nos sirve para desempatar
    rank = {name: (i+1) for i, (name, _) in enumerate(imps_sorted)}
    return rank

rank = cheap_ranking(feature_cols)

# -------- Hard-cap helper --------
K_CAP  = 9      # objetivo del paper (~5–10)
LAMBDA = 0.03   # más presión por sparsidad
BIG_P  = 0.10   # penal extra si algo se cuela >K (debería ser raro con repair)

def repair_mask(mask, feature_names, k_cap=K_CAP):
    """Ajusta el vector binario para que tenga como máximo k_cap unos.
       Si tiene 0, activa las top-k por ranking; si tiene >k, deja las k mejores por ranking."""
    on_idx = [i for i, m in enumerate(mask) if m == 1]
    if len(on_idx) == 0:
        # activa top-k por ranking global
        topk = sorted(range(len(feature_names)), key=lambda i: rank.get(feature_names[i], 10**9))[:k_cap]
        repaired = [1 if i in topk else 0 for i in range(len(feature_names))]
        return repaired
    if len(on_idx) <= k_cap:
        return mask[:]  # ya válido
    # más de k: quedarse con las k mejores entre las encendidas
    ordered = sorted(on_idx, key=lambda i: rank.get(feature_names[i], 10**9))
    keep = set(ordered[:k_cap])
    repaired = [1 if i in keep else 0 for i in range(len(feature_names))]
    return repaired

# -------- PSO pequeño --------
pso = PSOSelector(
    swarm_size=3,   # enjambre mínimo
    iters=6,        # pocas iteraciones
    w=0.7, c1=1.4, c2=1.4, vmax=4.0, seed=13
)

# -------- Función de evaluación con hard-cap --------
def eval_with_hardcap(train_df, test_df, feature_names, mask, max_depth=2, lam=LAMBDA):
    # Reparar antes de evaluar
    mask2 = repair_mask(mask, feature_names, k_cap=K_CAP)
    # Fitness base del evaluador existente (accuracy/F1 con árbol corto)
    f_obj, acc, f1 = evaluate_subset(train_df, test_df, feature_names, mask2,
                                     max_depth=max_depth, lam=lam)
    # Reconstruir objetivo con penalización explícita
    k = int(sum(mask2))
    obj = f1 - lam * k
    if k > K_CAP:
        obj -= BIG_P * (k - K_CAP)  # casi nunca debería entrar
    return obj, acc, f1, mask2

# -------- Búsqueda --------
t0 = time.time()
history = []
best_mask = None
best_obj = -1e9

for it in range(pso.iters):
    # una iteración manual usando el pso_selector interno
    # Nota: PSOSelector.fit_select llama eval_fn(tr,te,fn,m,...)
    bm, info = pso.fit_select(
        eval_fn=lambda tr, te, fn, m, max_depth, lam: eval_with_hardcap(tr, te, fn, m, max_depth=2, lam=LAMBDA)[:3],
        train_df=train_small, test_df=test_small, feature_names=feature_cols,
        lam=LAMBDA, max_depth=2
    )
    # Reparamos el mask resultante por si acaso y reevaluamos para registrar k
    obj, acc, f1, bm_fixed = eval_with_hardcap(train_small, test_small, feature_cols, bm, max_depth=2, lam=LAMBDA)
    history.append({"iter": it, "obj": obj, "acc": acc, "f1": f1, "k": int(sum(bm_fixed))})
    if obj > best_obj:
        best_obj = obj
        best_mask = bm_fixed

elapsed_search = time.time() - t0
selected_cols = [f for f, m in zip(feature_cols, best_mask) if m == 1]

# -------- Re-entrenar final (full, fiel al paper) --------
def eval_all_metrics_full(cols):
    df_tr = cast_clean(train, cols)
    df_te = cast_clean(test,  cols)
    va = VectorAssembler(inputCols=cols, outputCol="features", handleInvalid="keep")
    tr2 = va.transform(df_tr); te2 = va.transform(df_te)
    idx = StringIndexer(inputCol=label_col, outputCol="label", handleInvalid="keep").fit(tr2)
    tr3 = idx.transform(tr2); te3 = idx.transform(te2)
    dt = DecisionTreeClassifier(featuresCol="features", labelCol="label",
                                impurity="entropy", maxDepth=12, seed=13)
    t1 = time.time()
    model = dt.fit(tr3)
    pred  = model.transform(te3)
    elapsed = time.time() - t1
    out = {}
    for metric in ["accuracy", "f1", "weightedPrecision", "weightedRecall"]:
        ev = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName=metric)
        out[metric] = ev.evaluate(pred)
    return out, elapsed

metrics_full, elapsed_final = eval_all_metrics_full(selected_cols)

# -------- Guardar --------
(out_dir / "pso_tight_selected_features.txt").write_text("\n".join(selected_cols))
with open(out_dir / "pso_tight_metrics.json", "w") as f:
    json.dump({
        "metrics": metrics_full,
        "subset_size": len(selected_cols),
        "total_features": D,
        "elapsed_search_sec": elapsed_search,
        "elapsed_finaltrain_sec": elapsed_final,
        "lambda": LAMBDA, "k_cap": K_CAP, "big_penalty": BIG_P,
        "history": history
    }, f, indent=2)

print("✅ PSO TIGHT (HARDCAP) terminado")
print(f"Subset |S| = {len(selected_cols)} / {D}")
print("Metrics:", metrics_full)
print(f"Tiempo búsqueda: {elapsed_search:.2f}s | Entrenamiento final: {elapsed_final:.2f}s")
print("Salidas:")
print(" - data/results/pso_tight_selected_features.txt")
print(" - data/results/pso_tight_metrics.json")