"""
Model Evaluation Module for Sales Prediction Project.

This module provides functions to:
1. Compute regression metrics: MAE, MSE, RMSE, R² Score.
2. Build comparative evaluation tables across multiple models.
3. Generate Actual vs. Predicted scatter plots and residual plots.
4. Generate Model Comparison bar charts and Feature Importance plots.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_metrics(y_true, y_pred, model_name: str = "Model") -> dict:
    """
    Calculates MAE, MSE, RMSE, and R2 score.
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    return {
        "Model": model_name,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2
    }


def compare_models(models_dict: dict, X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    """
    Evaluates all trained models on the test set and compiles a comparative DataFrame.
    """
    results = []
    for name, model in models_dict.items():
        y_pred = model.predict(X_test)
        metrics = compute_metrics(y_test, y_pred, model_name=name)
        results.append(metrics)

    df_results = pd.DataFrame(results).sort_values(by="R2", ascending=False).reset_index(drop=True)
    return df_results


def plot_model_comparisons(df_results: pd.DataFrame, output_dir: str = "outputs/plots"):
    """
    Plots a multi-bar chart comparing R² and RMSE across models.
    """
    if not os.path.isabs(output_dir):
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        output_dir = os.path.join(project_root, output_dir)

    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # R2 Score comparison
    sns.barplot(data=df_results, x="Model", y="R2", palette="Blues_r", ax=axes[0], hue="Model", legend=False)
    axes[0].set_title("Model Comparison: R² Score (Higher is Better)", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("R² Score", fontsize=11)
    axes[0].set_ylim(0, 1.05)
    for p in axes[0].patches:
        val = p.get_height()
        if not np.isnan(val) and val > 0:
            axes[0].annotate(f"{val:.4f}", (p.get_x() + p.get_width() / 2., val),
                             ha="center", va="bottom", fontsize=10, xytext=(0, 3),
                             textcoords="offset points")

    # RMSE comparison
    sns.barplot(data=df_results, x="Model", y="RMSE", palette="Reds", ax=axes[1], hue="Model", legend=False)
    axes[1].set_title("Model Comparison: RMSE (Lower is Better)", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Root Mean Squared Error", fontsize=11)
    for p in axes[1].patches:
        val = p.get_height()
        if not np.isnan(val) and val > 0:
            axes[1].annotate(f"{val:.4f}", (p.get_x() + p.get_width() / 2., val),
                             ha="center", va="bottom", fontsize=10, xytext=(0, 3),
                             textcoords="offset points")

    for ax in axes:
        ax.set_xlabel("Regression Algorithm", fontsize=11)
        ax.tick_params(axis="x", rotation=15)

    plt.savefig(os.path.join(output_dir, "06_model_comparison_metrics.png"), dpi=300)
    plt.close()
    print("[SAVED] 06_model_comparison_metrics.png")


def plot_actual_vs_predicted(y_test, y_pred, model_name: str, output_dir: str = "outputs/plots"):
    """
    Generates Actual vs Predicted Sales scatter plot with perfect prediction diagonal line.
    """
    if not os.path.isabs(output_dir):
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        output_dir = os.path.join(project_root, output_dir)

    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(7, 6))

    plt.scatter(y_test, y_pred, color="#1d3557", alpha=0.75, edgecolors="k", s=50, label="Test Predictions")

    min_val = min(y_test.min(), y_pred.min()) - 1
    max_val = max(y_test.max(), y_pred.max()) + 1
    plt.plot([min_val, max_val], [min_val, max_val], color="#e63946", linestyle="--", linewidth=2, label="Ideal Fit (y = x)")

    plt.title(f"Actual vs. Predicted Sales ({model_name})", fontsize=13, fontweight="bold")
    plt.xlabel("Actual Sales", fontsize=11)
    plt.ylabel("Predicted Sales", fontsize=11)
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    plt.legend(frameon=True)

    plt.savefig(os.path.join(output_dir, "07_actual_vs_predicted.png"), dpi=300)
    plt.close()
    print("[SAVED] 07_actual_vs_predicted.png")


def plot_feature_importance(feature_names: list, importances: np.ndarray, model_name: str, output_dir: str = "outputs/plots"):
    """
    Plots feature importance for tree-based ensemble models.
    """
    if not os.path.isabs(output_dir):
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        output_dir = os.path.join(project_root, output_dir)

    os.makedirs(output_dir, exist_ok=True)
    fi_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    plt.figure(figsize=(7, 5))
    sns.barplot(data=fi_df, x="Importance", y="Feature", palette="viridis", hue="Feature", legend=False)
    plt.title(f"Feature Importance ({model_name})", fontsize=13, fontweight="bold")
    plt.xlabel("Relative Importance Score", fontsize=11)
    plt.ylabel("Advertising Channel", fontsize=11)

    for p in plt.gca().patches:
        val = p.get_width()
        if not np.isnan(val) and val > 0:
            plt.gca().annotate(f"{val:.4f} ({val*100:.1f}%)", (val, p.get_y() + p.get_height() / 2.),
                               ha="left", va="center", fontsize=10, xytext=(4, 0),
                               textcoords="offset points")

    plt.xlim(0, max(importances) * 1.25)
    plt.savefig(os.path.join(output_dir, "08_feature_importance.png"), dpi=300)
    plt.close()
    print("[SAVED] 08_feature_importance.png")
