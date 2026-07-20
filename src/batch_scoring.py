"""
Batch scoring pipeline for the turbofan RUL model, using PySpark.

Distinct from the FastAPI real-time path (src/api/main.py): this script is
designed to score an entire fleet of engines at once, e.g. on a schedule,
rather than answering one live prediction request at a time.

Usage:
    python src/batch_scoring.py
"""

from pathlib import Path

import joblib
import mlflow
import mlflow.pyfunc
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.window import Window

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / 'data' / 'CMAPSSData'
MODELS_DIR = PROJECT_ROOT / 'models'
OUTPUT_PATH = PROJECT_ROOT / 'data' / 'processed' / 'batch_predictions.csv'

WINDOW_SIZE = 5  # must match the rolling window used in training (01_eda.ipynb)

COL_NAMES = ['unit', 'cycle', 'op_setting_1', 'op_setting_2', 'op_setting_3'] + \
            [f'sensor_{i}' for i in range(1, 22)]

SENSOR_COLS_TO_USE = ['sensor_2', 'sensor_3', 'sensor_4', 'sensor_7',
                       'sensor_8', 'sensor_9', 'sensor_11', 'sensor_12',
                       'sensor_13', 'sensor_14', 'sensor_15', 'sensor_17',
                       'sensor_20', 'sensor_21']


def load_dataset(spark, path, dataset_tag):
    """
    Read one CMAPSS test file. These are whitespace-delimited with irregular
    spacing (not a single fixed delimiter), so we read as raw text and split
    on a whitespace regex rather than using Spark's standard CSV reader,
    which only supports a single fixed separator character.
    """
    raw = spark.read.text(str(path))
    split_col = F.split(F.trim(F.col('value')), r'\s+')

    df = raw.select(
        *[split_col.getItem(i).cast('double').alias(COL_NAMES[i]) for i in range(len(COL_NAMES))]
    )
    df = df.withColumn('dataset', F.lit(dataset_tag))
    # Composite key: unit numbers restart at 1 in every FD00X file, so a raw
    # 'unit' column alone would silently collide across datasets once merged.
    df = df.withColumn('fleet_unit', F.concat_ws('_', F.col('dataset'), F.col('unit').cast('int')))
    return df


def build_fleet(spark):
    """Load and union all four FD00X test sets into one 'fleet' DataFrame."""
    datasets = [f'FD00{i}' for i in range(1, 5)]
    dfs = [load_dataset(spark, DATA_DIR / f'test_{tag}.txt', tag) for tag in datasets]

    fleet = dfs[0]
    for df in dfs[1:]:
        fleet = fleet.unionByName(df)
    return fleet


def engineer_features(fleet):
    """
    Rolling mean/std per sensor, computed per-engine — the Spark-native
    equivalent of the pandas groupby().transform() approach used in training.
    """
    window_spec = (
        Window.partitionBy('fleet_unit')
        .orderBy('cycle')
        .rowsBetween(-(WINDOW_SIZE - 1), 0)
    )

    for sensor in SENSOR_COLS_TO_USE:
        fleet = fleet.withColumn(f'{sensor}_rollmean', F.avg(sensor).over(window_spec))
        fleet = fleet.withColumn(f'{sensor}_rollstd', F.stddev_samp(sensor).over(window_spec))

    # Sample stddev over a single row is mathematically undefined (returns
    # null) — same as pandas .rolling(min_periods=1).std() on a first row.
    # Fill with 0, matching the exact convention used in training/predict.py.
    rollstd_cols = [f'{s}_rollstd' for s in SENSOR_COLS_TO_USE]
    fleet = fleet.fillna(0, subset=rollstd_cols)

    return fleet


def select_latest_cycle_per_engine(fleet):
    """
    Batch scoring answers: 'what is each engine's predicted RUL right now?'
    That means scoring only each engine's MOST RECENT cycle, not every
    historical row.
    """
    latest_window = Window.partitionBy('fleet_unit').orderBy(F.col('cycle').desc())
    fleet = fleet.withColumn('rn', F.row_number().over(latest_window))
    return fleet.filter(F.col('rn') == 1).drop('rn')


def score_with_model(scoring_pd, feature_cols):
    """
    Apply the registered XGBoost model to the engineered features.

    NOTE ON DESIGN CHOICE: we collect to a pandas DataFrame here rather than
    using mlflow.pyfunc.spark_udf() for a fully distributed apply. The
    reason: our serving bundle keeps the fitted StandardScaler SEPARATE from
    the registered MLflow model (see 02_modeling.ipynb) — the scaler was
    never packaged INTO the MLflow model artifact itself. A true distributed
    spark_udf apply would need that scaler broadcast to every Spark worker
    and applied there too, which adds real complexity for a dataset this
    size. Since our full fleet (a few hundred engines' latest cycles) fits
    trivially in memory, collecting to pandas for this final scoring step
    is simpler and equally correct. The distributed READING and FEATURE
    ENGINEERING above still genuinely run through Spark — this is the one
    step, at the very end, where we intentionally step back to pandas.
    A larger fleet (thousands+ engines) would justify folding the scaler
    into the MLflow model itself (e.g. as an sklearn Pipeline) specifically
    to enable a fully distributed spark_udf path.
    """
    bundle = joblib.load(MODELS_DIR / 'serving_bundle.joblib')

    mlflow.set_tracking_uri("sqlite:///" + str(PROJECT_ROOT / "mlflow.db"))
    model = mlflow.pyfunc.load_model("models:/turbofan-xgb-regressor@champion")

    X = scoring_pd[bundle['feature_cols']].copy()
    X[bundle['feature_cols']] = bundle['scaler'].transform(X[bundle['feature_cols']])

    scoring_pd['predicted_rul'] = model.predict(X)
    return scoring_pd


def main():
    spark = SparkSession.builder \
        .appName("TurbofanBatchScoring") \
        .master("local[*]") \
        .getOrCreate()

    print("Loading fleet data...")
    fleet = build_fleet(spark)
    print(f"Total rows across all datasets: {fleet.count()}")

    print("Engineering rolling features (Spark Window functions)...")
    fleet = engineer_features(fleet)

    print("Selecting each engine's latest cycle...")
    fleet_latest = select_latest_cycle_per_engine(fleet)
    n_engines = fleet_latest.count()
    print(f"Scoring {n_engines} engines' current state...")

    feature_cols = SENSOR_COLS_TO_USE + \
        [f'{s}_rollmean' for s in SENSOR_COLS_TO_USE] + \
        [f'{s}_rollstd' for s in SENSOR_COLS_TO_USE]

    scoring_pd = fleet_latest.select('dataset', 'unit', 'cycle', *feature_cols).toPandas()

    spark.stop()  # release Spark resources before the pandas-based scoring step

    print("Scoring with registered XGBoost model...")
    results = score_with_model(scoring_pd, feature_cols)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output_cols = ['dataset', 'unit', 'cycle', 'predicted_rul']
    
    results['unit'] = results['unit'].astype(int)
    results['cycle'] = results['cycle'].astype(int)

    results[output_cols].to_csv(OUTPUT_PATH, index=False)

    print(f"Wrote {len(results)} predictions to {OUTPUT_PATH}")
    print(results[output_cols].head(10))


if __name__ == '__main__':
    main()
