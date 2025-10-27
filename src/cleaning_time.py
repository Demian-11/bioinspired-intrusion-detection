# scripts/clean_cicids.py
# Limpieza robusta de CSE-CIC-IDS2018 con PySpark (portátil para macOS/Linux)
# - Une CSVs por NOMBRE de columna (no por posición)
# - Castea con tolerancia (try_cast) para evitar crashes
# - Arregla 'inf', '+inf', '-inf', 'infinity' en tasas
# - No elimina 'Flow Duration' (se usa en filtros)
# - Escribe salida en CSV (coalesce) y en Parquet

from pathlib import Path
import os
import sys
import subprocess
from functools import reduce

from pyspark.sql import SparkSession, DataFrame
import pyspark.sql.functions as F
from pyspark.sql.functions import col

# ---------------------------------------------------------------------
# 0) Asegurar entorno (útil si lo ejecutas en distintas terminales)
# ---------------------------------------------------------------------
# Forzar el Python actual para PySpark
os.environ["PYSPARK_PYTHON"] = sys.executable
# Intentar fijar JAVA_HOME a 17 si no está configurado
if "JAVA_HOME" not in os.environ:
    try:
        jh = subprocess.run(["/usr/libexec/java_home", "-v", "17"],
                            capture_output=True, text=True)
        if jh.returncode == 0 and jh.stdout.strip():
            os.environ["JAVA_HOME"] = jh.stdout.strip()
    except Exception:
        pass

# ---------------------------------------------------------------------
# 1) SparkSession (config segura local)
# ---------------------------------------------------------------------
spark = (
    SparkSession.builder
    .appName("CICIDS2018_Cleaning")
    .master("local[*]")
    .config("spark.driver.memory", "6g")
    .config("spark.sql.shuffle.partitions", "100")
    .config("spark.sql.files.maxPartitionBytes", "134217728")  # 128MB
    .getOrCreate()
)

# ---------------------------------------------------------------------
# 2) Lectura robusta de CSVs (union by name)
#    Ejecuta este script desde la carpeta raíz del reto (donde está /Data)
# ---------------------------------------------------------------------
base = Path("Data")
files = sorted(str(p) for p in base.glob("*.csv"))
if not files:
    raise FileNotFoundError("No encontré CSVs en data/*.csv. Verifica tu ruta.")

dfs = []
for fp in files:
    df_i = (
        spark.read
        .option("header", True)
        .option("inferSchema", False)   # todo como string; castearemos luego
        # .option("recursiveFileLookup", "true")  # activa si tus CSV están en subcarpetas
        .csv(fp)
    )
    # Normaliza espacios en nombres
    df_i = df_i.toDF(*[c.strip() for c in df_i.columns])
    # Elimina filas que repiten el header
    if "Label" in df_i.columns:
        df_i = df_i.filter(F.col("Label") != "Label")
    dfs.append(df_i)

# Une por NOMBRE (no por posición)
df = reduce(lambda a, b: a.unionByName(b, allowMissingColumns=True), dfs)

print(f"Total inicial: {df.count()} filas, {len(df.columns)} columnas")

# ---------------------------------------------------------------------
# 3) Drops de columnas redundantes (no borres 'Flow Duration')
# ---------------------------------------------------------------------
drop_cols = [
    "Protocol", "Timestamp",
    "Flow ID", "Src IP", "Src Port", "Dst IP",
    # Campos listados como prescindibles en tablas del paper
    "Bwd Blk Rate Avg", "Fwd Byts/b Avg", "Bwd Pkts/b Avg", "Fwd Blk Rate Avg",
    "Bwd PSH Flags", "Fwd Pkts/b Avg", "Bwd Byts/b Avg", "Bwd URG Flags",
    "Init Fwd Win Byts", "Init Bwd Win Byts",
]
df = df.drop(*[c for c in drop_cols if c in df.columns])
print(f"Después de drop: {df.count()} filas, {len(df.columns)} columnas")

# ---------------------------------------------------------------------
# 4) Helpers de casteo tolerante y filtros seguros
# ---------------------------------------------------------------------
def try_cast_double(df: DataFrame, colname: str) -> DataFrame:
    """Castea a double tolerando strings tipo 'inf', IPs, etc. Valores inválidos -> NULL."""
    if colname not in df.columns:
        return df
    df = df.withColumn(
        colname,
        F.when(F.lower(F.col(colname)).isin("inf", "+inf", "-inf", "infinity", "+infinity", "-infinity"), None)
         .otherwise(F.col(colname))
    )
    # try_cast para evitar excepciones (Spark SQL)
    df = df.withColumn(colname, F.expr(f"try_cast(`{colname}` as double)"))
    return df

def filter_nonneg(df: DataFrame, colname: str) -> DataFrame:
    if colname in df.columns:
        df = try_cast_double(df, colname)
        df = df.filter((F.col(colname).isNull()) | (F.col(colname) >= 0))
    return df

def filter_nonzero(df: DataFrame, colname: str) -> DataFrame:
    if colname in df.columns:
        df = try_cast_double(df, colname)
        df = df.filter((F.col(colname).isNull()) | (F.col(colname) != 0))
    return df

# ---------------------------------------------------------------------
# 5) Quitar valores negativos en columnas clave
# ---------------------------------------------------------------------
for c in ["Fwd Header Len", "Flow Duration", "Flow IAT Min"]:
    df = filter_nonneg(df, c)

# ---------------------------------------------------------------------
# 6) Quitar ceros en 8 campos (según paper)
# ---------------------------------------------------------------------
for c in ["Tot Fwd Pkts", "Tot Bwd Pkts", "Pkt Len Min", "Pkt Len Max",
          "Pkt Len Mean", "Flow Byts/s", "Flow Pkts/s", "Fwd Header Len"]:
    df = filter_nonzero(df, c)

# ---------------------------------------------------------------------
# 7) Limpieza adicional de tasas (NaN/Inf -> fuera)
# ---------------------------------------------------------------------
for c in ["Flow Byts/s", "Flow Pkts/s"]:
    if c in df.columns:
        df = try_cast_double(df, c)
        df = df.filter(F.col(c).isNull() | (~F.isnan(F.col(c))))

# (Opcional) si quieres descartar cualquier fila con NULL en campos críticos, descomenta:
# critical = ["Flow Duration", "Tot Fwd Pkts", "Tot Bwd Pkts", "Flow Byts/s", "Flow Pkts/s"]
# keep = [c for c in critical if c in df.columns]
# if keep:
#     df = df.na.drop(subset=keep)

print(f"Después de limpieza: {df.count()} filas, {len(df.columns)} columnas")

# ---------------------------------------------------------------------
# 8) Guardado de resultados
# ---------------------------------------------------------------------
csv_out = "data/cleaned_dataset_csv"
parquet_out = "data/cleaned_dataset_parquet"

# CSV (coalesce para no generar cientos de archivos)
(df.coalesce(8)
   .write.mode("overwrite")
   .option("header", True)
   .csv(csv_out))

# Parquet (recomendado para pipelines siguientes)
(df.write.mode("overwrite").parquet(parquet_out))

print("Limpieza completada y datos escritos en:")
print(f" - CSV:     {csv_out}")
print(f" - Parquet: {parquet_out}")

# (Opcional) pequeño resumen por etiqueta para el reporte:
if "Label" in df.columns:
    try:
        df.groupBy("Label").count().orderBy(F.desc("count")).show(50, truncate=False)
    except Exception:
        pass

# Cerrar sesión (opcional)
# spark.stop()