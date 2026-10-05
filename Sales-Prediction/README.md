# 📈 Sales Prediction & Advertising Analytics Using Python

An end-to-end Machine Learning and Business Analytics project that predicts product sales volume based on multi-channel advertising expenditures across **TV**, **Radio**, and **Newspaper**.

---

## 📌 Project Overview

In advertising and marketing campaigns, businesses invest capital across various media channels to drive sales. This project builds and compares multiple regression machine learning models to accurately estimate sales volume from advertising spending, quantify channel importance, and provide actionable marketing optimization insights.

---

## 📊 Dataset Information

The project utilizes the classic `Advertising.csv` dataset containing 200 empirical market records.

| Column | Data Type | Description |
| :--- | :--- | :--- |
| `TV` | Numeric (float) | Advertising expenditure on Television (in thousands of dollars) |
| `Radio` | Numeric (float) | Advertising expenditure on Radio (in thousands of dollars) |
| `Newspaper` | Numeric (float) | Advertising expenditure on Newspaper (in thousands of dollars) |
| `Sales` | Numeric (float) | **Target Variable**: Total sales volume (in thousands of units) |

> **Note on Data Cleaning**: The dataset contains an initial row index column `Unnamed: 0` which is systematically removed during preprocessing. No synthetic or extraneous columns are introduced.

---

## 🛠️ Technology Stack

- **Language**: Python 3.x
- **Data Manipulation & Analysis**: `pandas`, `numpy`
- **Visualization**: `matplotlib`, `seaborn`
- **Machine Learning**: `scikit-learn`
- **Model Serialization**: `joblib`
- **Web Application / Dashboard**: `streamlit`

---

## 🏗️ Project Architecture & Structure

```text
Sales-Prediction/
│
├── data/
│   └── Advertising.csv                    # Cleaned market dataset (200 records)
│
├── notebooks/
│   └── sales_prediction_analysis.ipynb    # Interactive step-by-step Jupyter Notebook
│
├── src/
│   ├── data_preprocessing.py              # Data cleaning and train/test split
│   ├── eda.py                             # Exploratory data analysis and visualizations
│   ├── train_model.py                     # Multi-model training and evaluation pipeline
│   ├── evaluate_model.py                  # Regression metrics and evaluation charts
│   └── predict.py                         # Single prediction & scenario comparison module
│
├── models/
│   └── sales_prediction_model.pkl         # Serialized best-performing ML model
│
├── outputs/
│   ├── plots/                             # Saved high-resolution figures
│   │   ├── 01_sales_distribution.png
│   │   ├── 02_correlation_heatmap.png
│   │   ├── 03_tv_vs_sales.png
│   │   ├── 03_radio_vs_sales.png
│   │   ├── 03_newspaper_vs_sales.png
│   │   ├── 04_all_channels_vs_sales.png
│   │   ├── 05_channel_distributions.png
│   │   ├── 06_model_comparison_metrics.png
│   │   ├── 07_actual_vs_predicted.png
│   │   └── 08_feature_importance.png
│   └── reports/
│       └── model_evaluation_report.md     # Auto-generated analytical markdown report
│
├── sales_prediction.py                    # Standalone single-file beginner script
├── app.py                                 # Interactive Streamlit Web UI & Scenario Simulator
├── requirements.txt                       # Project dependencies
└── README.md                              # Comprehensive project documentation
```

---

## 🔄 Machine Learning Workflow

```text
Data Collection (Advertising.csv)
       ↓
Data Cleaning & Preprocessing (Remove Unnamed: 0, Validate nulls/types)
       ↓
Exploratory Data Analysis (Correlation analysis, Distribution & Regplots)
       ↓
Feature & Target Definition (X: TV, Radio, Newspaper | y: Sales)
       ↓
Train/Test Split (80% Train: 160 samples | 20% Test: 40 samples, random_state=42)
       ↓
Model Training (Linear Regression, Decision Tree, Random Forest, Gradient Boosting)
       ↓
Model Evaluation & Benchmarking (MAE, MSE, RMSE, R² Score)
       ↓
Model Selection (Gradient Boosting: R² = 0.9831, RMSE = 0.7298)
       ↓
What-If Scenario Simulation & Web App Deployment (Streamlit)
```

---

## 📈 Exploratory Data Analysis & Statistical Findings

### 1. Pearson Correlation with Sales ($r$)
- **TV Advertising**: `0.7822` (Strongest positive relationship)
- **Radio Advertising**: `0.5762` (Moderate-to-strong positive relationship)
- **Newspaper Advertising**: `0.2283` (Weakest relationship)

### 2. Linear Regression Coefficients
$$\text{Predicted Sales} = 2.9791 + (0.0447 \times \text{TV}) + (0.1892 \times \text{Radio}) + (0.0028 \times \text{Newspaper})$$

- **TV Coefficient ($0.0447$)**: Holding other variables constant, every additional $1,000 spend on TV ads is associated with an estimated increase of ~44.7 units in sales.
- **Radio Coefficient ($0.1892$)**: Holding other variables constant, each additional $1,000 spend on Radio ads is associated with an estimated increase of ~189.2 units in sales.
- **Newspaper Coefficient ($0.0028$)**: Marginal observed association with predicted sales.

---

## 🏆 Model Evaluation & Benchmark Results

All models were evaluated on the held-out **20% unseen test partition (40 records)**:

| Rank | Model Algorithm | MAE | MSE | RMSE | R² Score |
| :---: | :--- | :---: | :---: | :---: | :---: |
| 🥇 | **Gradient Boosting Regressor** | **0.6187** | **0.5326** | **0.7298** | **0.9831** |
| 🥈 | **Random Forest Regressor** | 0.6203 | 0.5909 | 0.7687 | 0.9813 |
| 🥉 | **Decision Tree Regressor** | 0.9850 | 2.1750 | 1.4748 | 0.9311 |
| 4 | **Linear Regression** | 1.4608 | 3.1741 | 1.7816 | 0.8994 |

### Metric Definitions:
- **MAE (Mean Absolute Error)**: Average absolute magnitude of errors between actual and predicted sales.
- **RMSE (Root Mean Squared Error)**: Standard deviation of prediction residuals; penalizes larger errors.
- **R² Score (Coefficient of Determination)**: Percentage of variance in sales explained by the models. The best model explains **98.31%** of variance on test data.

---

## 🌲 Feature Importance Analysis (Gradient Boosting)

- **TV Spending**: ~`60.4%` relative feature importance
- **Radio Spending**: ~`38.7%` relative feature importance
- **Newspaper Spending**: ~`0.9%` relative feature importance

---

## 💡 Key Business & Marketing Takeaways

1. **Prioritize TV & Radio Synergies**: TV provides widespread market visibility while Radio offers high conversion efficiency per dollar.
2. **Reallocate Newspaper Budget**: Because Newspaper shows negligible feature importance (<1%) and near-zero coefficient, reallocating this budget to TV or Radio is statistically projected to yield higher sales volume.
3. **Causal Interpretation Precaution**: Regression coefficients represent observed patterns in historical data rather than guaranteed causal effects. Controlled A/B testing is advised when adjusting budgets.

---

## 🚀 How to Run the Project

### 1. Clone or Navigate to the Directory
```bash
cd codealpha-tasks/Sales-Prediction
```

### 2. (Optional) Create and Activate Virtual Environment
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run Model Training Pipeline
```bash
python src/train_model.py
```
*(Or run the single-file beginner script: `python sales_prediction.py`)*

### 5. Launch Interactive Streamlit Web Application
```bash
streamlit run app.py
```

### 6. Interactive Command-Line Predictions
```bash
python src/predict.py
```

---

## 👨‍💻 Author & Academic Portfolio
- **Project**: Sales Prediction and Advertising Analytics
- **Domain**: Data Science & Machine Learning
- **Focus**: Regression modeling, EDA, feature attribution, interactive model deployment.
