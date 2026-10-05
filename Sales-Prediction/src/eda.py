"""
Exploratory Data Analysis (EDA) Module for Sales Prediction Project.

This module generates visualizations and statistical correlations to understand:
- Distributions of advertising expenditures (TV, Radio, Newspaper) and Sales.
- Scatter plots showing relationships of each advertising channel with Sales.
- Correlation matrix and heatmap.
- Identifies strong vs weak relationships without inferring unjustified causation.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Support running directly or from root
sys.path.append(os.path.dirname(__file__))
from data_preprocessing import load_data, clean_data

# Set styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["figure.autolayout"] = True


def generate_eda(df: pd.DataFrame, output_dir: str = "outputs/plots") -> dict:
    """
    Generates and saves all EDA charts and computes correlation statistics.
    """
    if not os.path.isabs(output_dir):
        # Resolve path relative to project root
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        output_dir = os.path.join(project_root, output_dir)

    os.makedirs(output_dir, exist_ok=True)

    # 1. Calculate Correlation Matrix
    corr_matrix = df.corr()
    sales_corr = corr_matrix["Sales"].drop("Sales").sort_values(ascending=False)

    print("\n" + "=" * 50)
    print("--- CORRELATION WITH SALES ---")
    for feature, val in sales_corr.items():
        print(f"{feature:12s}: {val:.4f}")
    print("=" * 50 + "\n")

    # --- Plot 1: Sales Distribution & Boxplot ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.histplot(df["Sales"], kde=True, color="#2b5c8f", ax=axes[0])
    axes[0].set_title("Sales Distribution (Histogram & KDE)", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Sales (in thousands of units)", fontsize=11)
    axes[0].set_ylabel("Frequency", fontsize=11)

    sns.boxplot(y=df["Sales"], color="#4ea8de", ax=axes[1])
    axes[1].set_title("Sales Boxplot (Checking Outliers)", fontsize=13, fontweight="bold")
    axes[1].set_ylabel("Sales (in thousands of units)", fontsize=11)

    plt.savefig(os.path.join(output_dir, "01_sales_distribution.png"), dpi=300)
    plt.close()
    print("[SAVED] 01_sales_distribution.png")

    # --- Plot 2: Correlation Heatmap ---
    plt.figure(figsize=(7, 6))
    sns.heatmap(corr_matrix, annot=True, cmap="Blues", fmt=".3f", linewidths=1, square=True, cbar_kws={"shrink": 0.8})
    plt.title("Advertising Channels & Sales Correlation Heatmap", fontsize=13, fontweight="bold", pad=12)
    plt.savefig(os.path.join(output_dir, "02_correlation_heatmap.png"), dpi=300)
    plt.close()
    print("[SAVED] 02_correlation_heatmap.png")

    # --- Plot 3: Individual Channel Scatter Plots vs Sales ---
    channels = [
        ("TV", "#1d3557", "TV Advertising Spend (in thousands $)"),
        ("Radio", "#e63946", "Radio Advertising Spend (in thousands $)"),
        ("Newspaper", "#2a9d8f", "Newspaper Advertising Spend (in thousands $)")
    ]

    for channel, color, xlabel in channels:
        plt.figure(figsize=(7, 5))
        sns.regplot(
            data=df,
            x=channel,
            y="Sales",
            color=color,
            scatter_kws={"alpha": 0.7, "edgecolor": "k", "s": 45},
            line_kws={"color": "#111111", "linewidth": 2}
        )
        r_val = corr_matrix.loc[channel, "Sales"]
        plt.title(f"{channel} Spending vs. Sales (r = {r_val:.3f})", fontsize=13, fontweight="bold")
        plt.xlabel(xlabel, fontsize=11)
        plt.ylabel("Sales (in thousands of units)", fontsize=11)
        plt.savefig(os.path.join(output_dir, f"03_{channel.lower()}_vs_sales.png"), dpi=300)
        plt.close()
        print(f"[SAVED] 03_{channel.lower()}_vs_sales.png")

    # --- Plot 4: Combined 3-Panel Scatter Plot ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    for i, (channel, color, xlabel) in enumerate(channels):
        sns.regplot(
            data=df,
            x=channel,
            y="Sales",
            color=color,
            ax=axes[i],
            scatter_kws={"alpha": 0.7, "s": 35},
            line_kws={"color": "#222222", "linewidth": 1.8}
        )
        r_val = corr_matrix.loc[channel, "Sales"]
        axes[i].set_title(f"{channel} vs Sales (r = {r_val:.3f})", fontsize=12, fontweight="bold")
        axes[i].set_xlabel(channel + " Spend", fontsize=11)
        if i == 0:
            axes[i].set_ylabel("Sales", fontsize=11)

    fig.suptitle("Advertising Spend Across All 3 Channels vs Sales", fontsize=15, fontweight="bold", y=1.02)
    plt.savefig(os.path.join(output_dir, "04_all_channels_vs_sales.png"), dpi=300)
    plt.close()
    print("[SAVED] 04_all_channels_vs_sales.png")

    # --- Plot 5: Channel Spending Distributions ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for i, (channel, color, _) in enumerate(channels):
        sns.histplot(df[channel], kde=True, color=color, ax=axes[i], bins=15)
        axes[i].set_title(f"{channel} Spend Distribution", fontsize=12, fontweight="bold")
        axes[i].set_xlabel(f"{channel} Spending", fontsize=10)
        axes[i].set_ylabel("Count", fontsize=10)

    plt.savefig(os.path.join(output_dir, "05_channel_distributions.png"), dpi=300)
    plt.close()
    print("[SAVED] 05_channel_distributions.png")

    return {
        "correlation_matrix": corr_matrix,
        "sales_correlation": sales_corr
    }


if __name__ == "__main__":
    df = clean_data(load_data())
    generate_eda(df)
