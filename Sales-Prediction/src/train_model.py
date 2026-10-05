"""
Model Training & Experimentation Pipeline for Sales Prediction.

This script executes the end-to-end Machine Learning pipeline:
1. Loads and cleans Advertising.csv
2. Generates exploratory data analysis plots and correlation stats
3. Splits data into 80% train / 20% test
4. Trains 4 regression models:
   - Linear Regression
   - Decision Tree Regressor
   - Random Forest Regressor
   - Gradient Boosting Regressor
5. Evaluates all models on the test set (MAE, MSE, RMSE, R²)
6. Identifies the best-performing model
7. Saves the trained best model to 'models/sales_prediction_model.pkl'
8. Saves evaluation charts to 'outputs/plots/' and report to 'outputs/reports/'
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np

# Support direct execution
sys.path.append(os.path.dirname(__file__))

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from data_preprocessing import load_data, clean_data, prepare_features_and_target, split_data
from eda import generate_eda
from evaluate_model import (
    compare_models,
    plot_model_comparisons,
    plot_actual_vs_predicted,
    plot_feature_importance
)


def train_models(X_train: pd.DataFrame, y_train: pd.Series) -> dict:
    """
    Fits multiple regression models on the training dataset.
    """
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42)
    }

    trained_models = {}
    for name, model in models.items():
        print(f"[TRAINING] Fitting {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model

    return trained_models


def df_to_md(df: pd.DataFrame) -> str:
    """Converts DataFrame to markdown table string without needing tabulate."""
    headers = list(df.columns)
    lines = ["| " + " | ".join(str(h) for h in headers) + " |"]
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    return "\n".join(lines)


def save_report(
    df_results: pd.DataFrame,
    best_model_name: str,
    lr_model: LinearRegression,
    best_model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    y_pred_best: np.ndarray,
    output_path: str = "outputs/reports/model_evaluation_report.md"
):
    """
    Saves a comprehensive markdown evaluation report with exact calculated values.
    """
    if not os.path.isabs(output_path):
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        output_path = os.path.join(project_root, output_path)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Actual vs predicted sample table
    comp_df = pd.DataFrame({
        "TV": X_test["TV"].values,
        "Radio": X_test["Radio"].values,
        "Newspaper": X_test["Newspaper"].values,
        "Actual Sales": y_test.values,
        "Predicted Sales": np.round(y_pred_best, 2),
        "Absolute Error": np.round(np.abs(y_test.values - y_pred_best), 2)
    })

    # Linear Regression coefficients
    lr_coef_df = pd.DataFrame({
        "Feature": ["Intercept", "TV", "Radio", "Newspaper"],
        "Coefficient": [
            round(lr_model.intercept_, 4),
            round(lr_model.coef_[0], 4),
            round(lr_model.coef_[1], 4),
            round(lr_model.coef_[2], 4)
        ]
    })

    # Formatted results
    formatted_results = df_results.copy()
    for col in ["MAE", "MSE", "RMSE", "R2"]:
        if col in formatted_results.columns:
            formatted_results[col] = formatted_results[col].apply(lambda x: f"{x:.4f}")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# Sales Prediction & Advertising Analytics Report\n\n")
        f.write("## 1. Model Performance Summary (Test Set, N=40)\n\n")
        f.write(df_to_md(formatted_results))
        f.write("\n\n")

        f.write(f"**Best Performing Model**: `{best_model_name}` (Selected based on highest $R^2$ and lowest RMSE).\n\n")

        f.write("## 2. Linear Regression Coefficients\n\n")
        f.write(df_to_md(lr_coef_df))
        f.write("\n\n")
        f.write("### Coefficient Interpretation (Holding other variables constant):\n")
        f.write(f"- **Intercept ({lr_model.intercept_:.4f})**: Base predicted sales when spend on all three channels is 0.\n")
        f.write(f"- **TV ({lr_model.coef_[0]:.4f})**: Within this fitted linear model, each 1-unit increase in TV spend is associated with a {lr_model.coef_[0]:.4f} unit increase in predicted sales.\n")
        f.write(f"- **Radio ({lr_model.coef_[1]:.4f})**: Within this fitted linear model, each 1-unit increase in Radio spend is associated with a {lr_model.coef_[1]:.4f} unit increase in predicted sales.\n")
        f.write(f"- **Newspaper ({lr_model.coef_[2]:.4f})**: Within this fitted linear model, each 1-unit increase in Newspaper spend is associated with a {lr_model.coef_[2]:.4f} unit change in predicted sales.\n\n")

        if hasattr(best_model, "feature_importances_"):
            f.write(f"## 3. Feature Importance ({best_model_name})\n\n")
            fi_df = pd.DataFrame({
                "Channel": ["TV", "Radio", "Newspaper"],
                "Importance Score": [f"{v:.4f}" for v in best_model.feature_importances_],
                "Percentage (%)": [f"{v*100:.2f}%" for v in best_model.feature_importances_]
            }).sort_values(by="Importance Score", ascending=False)
            f.write(df_to_md(fi_df))
            f.write("\n\n")

        f.write("## 4. Actual vs Predicted Sample Table (First 15 Test Observations)\n\n")
        f.write(df_to_md(comp_df.head(15)))
        f.write("\n\n")

        f.write("## 5. Key Marketing Insights\n")
        f.write("1. **TV and Radio** show the strongest positive relationship and contribution to predicted sales.\n")
        f.write("2. **Newspaper** shows the weakest relationship with sales and near-zero coefficient / minimal feature importance score.\n")
        f.write("3. **Non-linear Models** (e.g. Random Forest / Gradient Boosting) capture interaction effects between TV and Radio advertising.\n")
        f.write("4. Correlation and model coefficients describe observed relationships in data, not guaranteed real-world causal impacts.\n")

    print(f"[SAVED] Evaluation report saved to {output_path}")


def run_pipeline():
    """
    Main training pipeline execution.
    """
    print("=" * 60)
    print("STARTING SALES PREDICTION MACHINE LEARNING PIPELINE")
    print("=" * 60)

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_path = os.path.join(project_root, "data", "Advertising.csv")
    plots_dir = os.path.join(project_root, "outputs", "plots")
    models_dir = os.path.join(project_root, "models")
    reports_path = os.path.join(project_root, "outputs", "reports", "model_evaluation_report.md")

    # 1. Load and clean
    df_raw = load_data(data_path)
    df_clean = clean_data(df_raw)

    # 2. EDA & Plots
    print("\n--- Generating Exploratory Data Analysis & Visualizations ---")
    generate_eda(df_clean, output_dir=plots_dir)

    # 3. Prepare features & split
    X, y = prepare_features_and_target(df_clean)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)

    # 4. Train models
    print("\n--- Training Regression Models ---")
    models = train_models(X_train, y_train)

    # 5. Evaluate models
    print("\n--- Evaluating Models on Test Set ---")
    df_results = compare_models(models, X_test, y_test)
    print("\nModel Comparison Table:")
    print(df_results.to_string(index=False))

    # Identify best model
    best_model_name = df_results.iloc[0]["Model"]
    best_model = models[best_model_name]
    print(f"\n[SELECTION] Best Performing Model: {best_model_name} (R² = {df_results.iloc[0]['R2']:.4f})")

    # 6. Generate evaluation visualizations
    print("\n--- Generating Evaluation Charts ---")
    plot_model_comparisons(df_results, output_dir=plots_dir)

    y_pred_best = best_model.predict(X_test)
    plot_actual_vs_predicted(y_test, y_pred_best, best_model_name, output_dir=plots_dir)

    # Plot feature importance if best or ensemble model has it
    if hasattr(best_model, "feature_importances_"):
        plot_feature_importance(["TV", "Radio", "Newspaper"], best_model.feature_importances_, best_model_name, output_dir=plots_dir)
    elif "Random Forest" in models:
        plot_feature_importance(["TV", "Radio", "Newspaper"], models["Random Forest"].feature_importances_, "Random Forest", output_dir=plots_dir)

    # 7. Save model artifact
    os.makedirs(models_dir, exist_ok=True)
    model_save_path = os.path.join(models_dir, "sales_prediction_model.pkl")
    artifact = {
        "model": best_model,
        "model_name": best_model_name,
        "feature_names": ["TV", "Radio", "Newspaper"],
        "metrics": df_results.iloc[0].to_dict(),
        "all_results": df_results,
        "all_models": models,
        "lr_coefficients": {
            "intercept": models["Linear Regression"].intercept_,
            "TV": models["Linear Regression"].coef_[0],
            "Radio": models["Linear Regression"].coef_[1],
            "Newspaper": models["Linear Regression"].coef_[2]
        }
    }
    joblib.dump(artifact, model_save_path)
    print(f"[SAVED] Trained model artifact saved to {model_save_path}")

    # 8. Save markdown report
    save_report(
        df_results=df_results,
        best_model_name=best_model_name,
        lr_model=models["Linear Regression"],
        best_model=best_model,
        X_test=X_test,
        y_test=y_test,
        y_pred_best=y_pred_best,
        output_path=reports_path
    )

    print("\n" + "=" * 60)
    print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    run_pipeline()
