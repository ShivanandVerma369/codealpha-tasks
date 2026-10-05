"""
Prediction Module for Sales Prediction Project.

Provides utility functions to:
1. Load the trained machine learning model.
2. Predict sales from TV, Radio, and Newspaper advertising budgets.
3. Conduct what-if scenario analyses.
4. Run interactive command-line predictions.
"""

import os
import joblib
import pandas as pd
import numpy as np


def load_trained_model(model_path: str = "models/sales_prediction_model.pkl"):
    """
    Loads the serialized model artifact dictionary from disk.
    """
    if not os.path.isabs(model_path):
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        model_path = os.path.join(project_root, model_path)

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Trained model not found at {model_path}. Please run train_model.py first!"
        )

    artifact = joblib.load(model_path)
    return artifact


def predict_sales(tv: float, radio: float, newspaper: float, model=None) -> float:
    """
    Predicts sales (in thousands of units) based on advertising expenditures.

    Parameters:
    - tv (float): TV advertising spend (in thousands $)
    - radio (float): Radio advertising spend (in thousands $)
    - newspaper (float): Newspaper advertising spend (in thousands $)
    - model (optional): Pre-loaded model or artifact dict

    Returns:
    - float: Predicted Sales
    """
    if model is None:
        artifact = load_trained_model()
        model = artifact["model"]
    elif isinstance(model, dict) and "model" in model:
        model = model["model"]

    # Input validation
    tv = max(0.0, float(tv))
    radio = max(0.0, float(radio))
    newspaper = max(0.0, float(newspaper))

    input_df = pd.DataFrame([{
        "TV": tv,
        "Radio": radio,
        "Newspaper": newspaper
    }])

    prediction = model.predict(input_df)[0]
    return float(np.round(prediction, 2))


def compare_scenarios(
    scenario_a: dict,
    scenario_b: dict,
    model=None
) -> dict:
    """
    Compares two advertising budget scenarios and calculates absolute and percentage difference.

    Scenarios format: {'TV': x, 'Radio': y, 'Newspaper': z}
    """
    pred_a = predict_sales(scenario_a.get("TV", 0), scenario_a.get("Radio", 0), scenario_a.get("Newspaper", 0), model)
    pred_b = predict_sales(scenario_b.get("TV", 0), scenario_b.get("Radio", 0), scenario_b.get("Newspaper", 0), model)

    diff = pred_b - pred_a
    pct_diff = (diff / pred_a * 100) if pred_a != 0 else 0.0

    return {
        "Scenario A": {"budget": scenario_a, "predicted_sales": pred_a},
        "Scenario B": {"budget": scenario_b, "predicted_sales": pred_b},
        "Difference": round(diff, 2),
        "Percentage Change (%)": round(pct_diff, 2)
    }


def interactive_cli():
    """
    Runs an interactive CLI for users to test sales predictions.
    """
    print("\n" + "=" * 55)
    print("      SALES PREDICTION SYSTEM — CLI INTERFACE      ")
    print("=" * 55)

    try:
        artifact = load_trained_model()
        print(f"[INFO] Active Model: {artifact.get('model_name', 'Trained Regressor')}")
        print(f"[INFO] Test R² Score: {artifact.get('metrics', {}).get('R2', 'N/A'):.4f}")
    except Exception as e:
        print(f"[ERROR] {e}")
        return

    while True:
        print("\n--- Enter Advertising Budget (in thousands $) ---")
        try:
            tv = float(input("Enter TV advertising spend (e.g. 150): ").strip() or 0)
            radio = float(input("Enter Radio advertising spend (e.g. 30): ").strip() or 0)
            newspaper = float(input("Enter Newspaper advertising spend (e.g. 20): ").strip() or 0)

            sales = predict_sales(tv, radio, newspaper, model=artifact["model"])

            print("\n" + "-" * 40)
            print(f"Predicted Sales: {sales:.2f} (thousand units)")
            print("-" * 40)

        except ValueError:
            print("[ERROR] Please enter valid numerical values.")
        except Exception as e:
            print(f"[ERROR] Prediction failed: {e}")

        cont = input("\nDo you want to predict again? (y/n): ").strip().lower()
        if cont != 'y':
            print("\nExiting Sales Prediction CLI. Goodbye!")
            break


if __name__ == "__main__":
    interactive_cli()
