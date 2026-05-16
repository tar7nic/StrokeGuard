import pandas as pd
import numpy as np
import sqlite3
import os
import joblib

MODELS_DIR = "models"
DB_PATH = "models/clinicalrisk.db"
COHORT_DIR = "models/cohort_results"


def load_processed_data(models_dir: str = MODELS_DIR) -> pd.DataFrame:
    X_train = pd.read_csv(f"{models_dir}/X_train_fe.csv")
    X_test = pd.read_csv(f"{models_dir}/X_test_fe.csv")
    y_train = pd.read_csv(f"{models_dir}/y_train.csv").squeeze()
    y_test = pd.read_csv(f"{models_dir}/y_test.csv").squeeze()

    train_df = X_train.copy()
    train_df["stroke"] = y_train.values
    test_df = X_test.copy()
    test_df["stroke"] = y_test.values

    df = pd.concat([train_df, test_df], ignore_index=True)
    return df


def decode_features(df: pd.DataFrame, models_dir: str = MODELS_DIR) -> pd.DataFrame:
    df = df.copy()

    df["age_group_label"] = df["age_group"].map({
        0: "<18", 1: "18-40", 2: "40-60", 3: "60+"
    })
    df["glucose_category_label"] = df["glucose_category"].map({
        0: "Normal", 1: "Prediabetic", 2: "Diabetic"
    })
    df["bmi_risk_tier_label"] = df["bmi_risk_tier"].map({
        0: "Underweight", 1: "Normal", 2: "Overweight", 3: "Obese"
    })
    return df


def load_into_sqlite(df: pd.DataFrame, db_path: str = DB_PATH):
    conn = sqlite3.connect(db_path)
    df.to_sql("patients", conn, if_exists="replace", index=False)
    conn.close()
    print(f"  Loaded {len(df)} records into {db_path}")


def run_query(conn: sqlite3.Connection, sql: str) -> pd.DataFrame:
    return pd.read_sql_query(sql, conn)


def run_cohort_queries(db_path: str = DB_PATH, output_dir: str = COHORT_DIR):
    os.makedirs(output_dir, exist_ok=True)
    conn = sqlite3.connect(db_path)

    print("\n=== Cohort Query 1: Stroke Prevalence by Age Group ===")
    q1 = """
        SELECT
            age_group_label AS age_group,
            COUNT(*) AS total_patients,
            SUM(stroke) AS stroke_cases,
            ROUND(AVG(stroke) * 100, 2) AS stroke_rate_pct
        FROM patients
        GROUP BY age_group_label
        ORDER BY
            CASE age_group_label
                WHEN '<18' THEN 0
                WHEN '18-40' THEN 1
                WHEN '40-60' THEN 2
                WHEN '60+' THEN 3
            END
    """
    df1 = run_query(conn, q1)
    df1.to_csv(f"{output_dir}/q1_stroke_by_age_group.csv", index=False)
    print(df1.to_string(index=False))

    print("\n=== Cohort Query 2: Stroke Rate by Glucose Category ===")
    q2 = """
        SELECT
            glucose_category_label AS glucose_category,
            COUNT(*) AS total_patients,
            SUM(stroke) AS stroke_cases,
            ROUND(AVG(stroke) * 100, 2) AS stroke_rate_pct
        FROM patients
        GROUP BY glucose_category_label
        ORDER BY
            CASE glucose_category_label
                WHEN 'Normal' THEN 0
                WHEN 'Prediabetic' THEN 1
                WHEN 'Diabetic' THEN 2
            END
    """
    df2 = run_query(conn, q2)
    df2.to_csv(f"{output_dir}/q2_stroke_by_glucose.csv", index=False)
    print(df2.to_string(index=False))

    print("\n=== Cohort Query 3: Stroke Rate by BMI Risk Tier ===")
    q3 = """
        SELECT
            bmi_risk_tier_label AS bmi_tier,
            COUNT(*) AS total_patients,
            SUM(stroke) AS stroke_cases,
            ROUND(AVG(stroke) * 100, 2) AS stroke_rate_pct
        FROM patients
        GROUP BY bmi_risk_tier_label
        ORDER BY
            CASE bmi_risk_tier_label
                WHEN 'Underweight' THEN 0
                WHEN 'Normal' THEN 1
                WHEN 'Overweight' THEN 2
                WHEN 'Obese' THEN 3
            END
    """
    df3 = run_query(conn, q3)
    df3.to_csv(f"{output_dir}/q3_stroke_by_bmi_tier.csv", index=False)
    print(df3.to_string(index=False))

    print("\n=== Cohort Query 4: Stroke Rate — Hypertensive vs Non-Hypertensive ===")
    q4 = """
        SELECT
            CASE hypertension WHEN 1 THEN 'Hypertensive' ELSE 'Non-Hypertensive' END AS hypertension_status,
            COUNT(*) AS total_patients,
            SUM(stroke) AS stroke_cases,
            ROUND(AVG(stroke) * 100, 2) AS stroke_rate_pct
        FROM patients
        GROUP BY hypertension
        ORDER BY hypertension DESC
    """
    df4 = run_query(conn, q4)
    df4.to_csv(f"{output_dir}/q4_stroke_by_hypertension.csv", index=False)
    print(df4.to_string(index=False))

    print("\n=== Cohort Query 5: Top 5 Highest-Risk Cohorts by Vascular Risk Score ===")
    q5 = """
        SELECT
            vascular_risk_score,
            age_group_label AS age_group,
            glucose_category_label AS glucose_category,
            COUNT(*) AS total_patients,
            SUM(stroke) AS stroke_cases,
            ROUND(AVG(stroke) * 100, 2) AS stroke_rate_pct
        FROM patients
        GROUP BY vascular_risk_score, age_group_label, glucose_category_label
        HAVING total_patients >= 10
        ORDER BY stroke_rate_pct DESC
        LIMIT 5
    """
    df5 = run_query(conn, q5)
    df5.to_csv(f"{output_dir}/q5_top_risk_cohorts.csv", index=False)
    print(df5.to_string(index=False))

    conn.close()
    print(f"\n  All cohort CSVs saved to {output_dir}/")
    return df1, df2, df3, df4, df5


def run_database_pipeline(models_dir: str = MODELS_DIR, db_path: str = DB_PATH):
    print("=== Loading processed data ===")
    df = load_processed_data(models_dir)
    df = decode_features(df, models_dir)
    print(f"  Total records: {len(df)}, Stroke cases: {df['stroke'].sum()}")

    print("\n=== Loading into SQLite ===")
    load_into_sqlite(df, db_path)

    results = run_cohort_queries(db_path)
    return results


if __name__ == "__main__":
    run_database_pipeline()