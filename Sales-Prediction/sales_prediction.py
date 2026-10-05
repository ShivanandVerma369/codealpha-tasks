"""
=============================================================================
SALES PREDICTION USING PYTHON — COMPLETE ALL-IN-ONE SCRIPT
=============================================================================
This single-file script walks through the entire Machine Learning lifecycle:
1. Data Loading (Advertising.csv)
2. Data Inspection & Cleaning (Missing values, duplicates, outliers)
3. Exploratory Data Analysis (EDA & Visualizations)
4. Feature Selection & Train/Test Split (80/20)
5. Building Multiple Regression Models:
   - Linear Regression
   - Decision Tree Regressor
   - Random Forest Regressor
   - Gradient Boosting Regressor
6. Model Evaluation (MAE, MSE, RMSE, R² Score)
7. Actual vs. Predicted Comparison
8. Advertising Impact & Feature Importance Analysis
9. What-If Scenario Analysis & Prediction Function
10. Model Saving using Joblib
=============================================================================
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def main():
    print("=" * 70)
    print("      SALES PREDICTION & ADVERTISING ANALYTICS MACHINE LEARNING      ")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # STEP 1: LOAD DATASET
    # -------------------------------------------------------------------------
    data_path = os.path.join(os.path.dirname(__file__), "data", "Advertising.csv")
    if not os.path.exists(data_path):
        data_path = "data/Advertising.csv"

    print("\n[STEP 1] Loading Advertising.csv dataset...")
    df = pd.read_csv(data_path)

    print("\n--- Dataset Head (First 5 Rows) ---")
    print(df.head())

    print("\n--- Dataset Tail (Last 5 Rows) ---")
    print(df.tail())

    print(f"\n--- Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns ---")

    print("\n--- Column Info & Data Types ---")
    print(df.info())

    print("\n--- Summary Statistics ---")
    print(df.describe())

    # -------------------------------------------------------------------------
    # STEP 2: DATA CLEANING
    # -------------------------------------------------------------------------
    print("\n[STEP 2] Cleaning Dataset...")

    # Drop index column 'Unnamed: 0'
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
        print("[INFO] Dropped index artifact column: 'Unnamed: 0'")

    # Check missing values
    missing = df.isnull().sum()
    print("\nMissing values per column:")
    print(missing)
    if missing.sum() > 0:
        df = df.fillna(df.median())
        print("[INFO] Missing values imputed with median.")
    else:
        print("[INFO] Zero missing values detected.")

    # Check duplicates
    dups = df.duplicated().sum()
    print(f"[INFO] Duplicate rows: {dups}")
    if dups > 0:
        df = df.drop_duplicates()
        print(f"[INFO] Removed {dups} duplicate rows.")

    # -------------------------------------------------------------------------
    # STEP 3: EXPLORATORY DATA ANALYSIS (EDA)
    # -------------------------------------------------------------------------
    print("\n[STEP 3] Performing Exploratory Data Analysis & Generating Plots...")
    plots_dir = os.path.join(os.path.dirname(__file__), "outputs", "plots")
    os.makedirs(plots_dir, exist_ok=True)

    corr_matrix = df.corr()
    print("\n--- Correlation with Sales ---")
    for col in ["TV", "Radio", "Newspaper"]:
        print(f"{col:12s}: {corr_matrix.loc[col, 'Sales']:.4f}")

    # Plot 1: Correlation Heatmap
    plt.figure(figsize=(6, 5))
    sns.heatmap(corr_matrix, annot=True, cmap="Blues", fmt=".3f", linewidths=1)
    plt.title("Correlation Heatmap: Advertising vs Sales", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "02_correlation_heatmap.png"), dpi=300)
    plt.close()

    # Plot 2: Scatter plots
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    channels = [("TV", "#1d3557"), ("Radio", "#e63946"), ("Newspaper", "#2a9d8f")]
    for i, (col, color) in enumerate(channels):
        sns.regplot(data=df, x=col, y="Sales", color=color, ax=axes[i], line_kws={"color": "black"})
        axes[i].set_title(f"{col} vs Sales (r = {corr_matrix.loc[col, 'Sales']:.3f})", fontweight="bold")
    fig.suptitle("Advertising Channels vs Sales", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "04_all_channels_vs_sales.png"), dpi=300)
    plt.close()

    print(f"[INFO] EDA Visualizations saved to {plots_dir}")

    # -------------------------------------------------------------------------
    # STEP 4: FEATURE SELECTION
    # -------------------------------------------------------------------------
    print("\n[STEP 4] Feature Selection...")
    X = df[["TV", "Radio", "Newspaper"]]
    y = df["Sales"]
    print("Features (X): TV, Radio, Newspaper")
    print("Target (y): Sales")

    # -------------------------------------------------------------------------
    # STEP 5: TRAIN / TEST SPLIT (80/20)
    # -------------------------------------------------------------------------
    print("\n[STEP 5] Splitting dataset into Train (80%) and Test (20%) sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Testing set : {X_test.shape[0]} samples")

    # -------------------------------------------------------------------------
    # STEP 6: BUILD MULTIPLE REGRESSION MODELS
    # -------------------------------------------------------------------------
    print("\n[STEP 6] Training Multiple Regression Models...")
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42)
    }

    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        results.append({
            "Model": name,
            "MAE": round(mae, 4),
            "MSE": round(mse, 4),
            "RMSE": round(rmse, 4),
            "R2": round(r2, 4)
        })

    # -------------------------------------------------------------------------
    # STEP 7: MODEL EVALUATION & COMPARISON
    # -------------------------------------------------------------------------
    print("\n[STEP 7] Model Evaluation on Test Set:")
    results_df = pd.DataFrame(results).sort_values(by="R2", ascending=False).reset_index(drop=True)
    print(results_df.to_string(index=False))

    best_model_name = results_df.iloc[0]["Model"]
    best_model = models[best_model_name]
    print(f"\n>>> Selected Best Model: {best_model_name} (R² = {results_df.iloc[0]['R2']:.4f})")

    # -------------------------------------------------------------------------
    # STEP 8: ACTUAL VS PREDICTED SALES
    # -------------------------------------------------------------------------
    print("\n[STEP 8] Actual vs Predicted Sales (Test Set Preview):")
    y_pred_best = best_model.predict(X_test)
    comp_df = pd.DataFrame({
        "Actual Sales": y_test.values[:10],
        "Predicted Sales": np.round(y_pred_best[:10], 2),
        "Absolute Error": np.round(np.abs(y_test.values[:10] - y_pred_best[:10]), 2)
    })
    print(comp_df.to_string(index=False))

    # -------------------------------------------------------------------------
    # STEP 9: ADVERTISING IMPACT & COEFFICIENTS
    # -------------------------------------------------------------------------
    print("\n[STEP 9] Advertising Impact Analysis:")
    lr = models["Linear Regression"]
    print(f"Linear Regression Intercept: {lr.intercept_:.4f}")
    print(f"TV Coefficient            : {lr.coef_[0]:.4f}")
    print(f"Radio Coefficient         : {lr.coef_[1]:.4f}")
    print(f"Newspaper Coefficient     : {lr.coef_[2]:.4f}")
    print("\n*Note: Coefficients reflect the model's expected change in predicted sales per unit spending, holding other variables constant, not guaranteed real-world causation.")

    # -------------------------------------------------------------------------
    # STEP 10: WHAT-IF MARKETING SCENARIO ANALYSIS
    # -------------------------------------------------------------------------
    print("\n[STEP 10] What-If Scenario Analysis:")
    scenario_a = pd.DataFrame([{"TV": 100.0, "Radio": 20.0, "Newspaper": 10.0}])
    scenario_b = pd.DataFrame([{"TV": 150.0, "Radio": 30.0, "Newspaper": 20.0}])

    pred_a = best_model.predict(scenario_a)[0]
    pred_b = best_model.predict(scenario_b)[0]
    diff = pred_b - pred_a
    pct = (diff / pred_a) * 100

    print(f"Scenario A (TV=100, Radio=20, Newspaper=10) -> Predicted Sales: {pred_a:.2f}")
    print(f"Scenario B (TV=150, Radio=30, Newspaper=20) -> Predicted Sales: {pred_b:.2f}")
    print(f"Difference: {diff:+.2f} units ({pct:+.2f}%)")

    # -------------------------------------------------------------------------
    # STEP 11: FEATURE IMPORTANCE
    # -------------------------------------------------------------------------
    if hasattr(best_model, "feature_importances_"):
        print(f"\n[STEP 11] Feature Importance ({best_model_name}):")
        fi_df = pd.DataFrame({
            "Feature": ["TV", "Radio", "Newspaper"],
            "Importance": best_model.feature_importances_
        }).sort_values(by="Importance", ascending=False)
        print(fi_df.to_string(index=False))

    # -------------------------------------------------------------------------
    # STEP 12: SAVE TRAINED MODEL
    # -------------------------------------------------------------------------
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(models_dir, exist_ok=True)
    save_file = os.path.join(models_dir, "sales_prediction_model.pkl")

    artifact = {
        "model": best_model,
        "model_name": best_model_name,
        "feature_names": ["TV", "Radio", "Newspaper"],
        "metrics": results_df.iloc[0].to_dict(),
        "all_results": results_df
    }
    joblib.dump(artifact, save_file)
    print(f"\n[STEP 12] Saved trained model to '{save_file}'")

    print("\n" + "=" * 70)
    print("EXECUTION FINISHED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()
