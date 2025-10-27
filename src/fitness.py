# scripts/fitness.py
# Función objetivo para ABC: F1_macro - λ * (|S| / N)
# Entrena un árbol con entropía y evalúa F1 sobre el test.

from typing import List, Tuple
from pyspark.sql import DataFrame
import pyspark.sql.functions as F
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

def evaluate_subset(train: DataFrame,
                    test: DataFrame,
                    all_features: List[str],
                    mask: List[int],
                    label_col: str = "Target",
                    max_depth: int = 12,
                    seed: int = 13,
                    lam: float = 0.01) -> Tuple[float, float, int]:
    """
    Devuelve (objetivo, f1_macro, k).  objetivo = F1_macro - lam * (k / N)
    - Sin .cache() para evitar OOM.
    - Limpia NaN/Inf -> 0.
    """
    N = len(all_features)
    selected = [f for f, m in zip(all_features, mask) if m == 1]
    k = len(selected)
    if k == 0:
        return -1.0, 0.0, 0

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

    train2 = cast_clean(train, selected)
    test2  = cast_clean(test,  selected)

    va = VectorAssembler(inputCols=selected, outputCol="features", handleInvalid="keep")
    train3 = va.transform(train2)
    test3  = va.transform(test2)

    idx = StringIndexer(inputCol=label_col, outputCol="label", handleInvalid="keep").fit(train3)
    train4 = idx.transform(train3)
    test4  = idx.transform(test3)

    dt = DecisionTreeClassifier(featuresCol="features", labelCol="label",
                                impurity="entropy", maxDepth=max_depth, seed=seed)
    model = dt.fit(train4)
    pred  = model.transform(test4)

    ev = MulticlassClassificationEvaluator(labelCol="label",
                                           predictionCol="prediction",
                                           metricName="f1")
    f1 = ev.evaluate(pred)
    obj = f1 - lam * (k / N)
    return obj, f1, k