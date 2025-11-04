

# California Housing Price Prediction: A Comparative ML Workflow

[](https://www.python.org/)
[](https://opensource.org/licenses/MIT)
[](https://www.google.com/search?q=link-to-huggingface-or-nbviewer)

## Project Overview

This project implements a rigorous, end-to-end machine learning pipeline to predict median house values in California census block groups using the popular scikit-learn California Housing dataset. The primary goal was to demonstrate an iterative, disciplined approach to model optimization, culminating in a high-performance **XGBoost Regressor** benchmarked against a **PyTorch Deep Learning** model.

**Data Sourece:** scikit-learn

The core analysis showcases advanced techniques, including strategic **feature selection**, **ensemble methods**, and **custom spatial feature engineering**.

## Methodology and Key Insights

The project followed a three-phase optimization strategy, progressing from baseline models to a production-ready solution:

### 1\. Model-Driven Feature Selection

  * **Initial Baseline:** Established performance with Linear Regression and a Decision Tree.
  * **Strategic Pivot:** After confirming that initial, manual feature engineering did not yield improvements, a Decision Tree's inherent **Feature Importance** mechanism was used to identify and eliminate low-impact features. This streamlined the data, reducing the feature count from 11 to **7**, and demonstrated efficiency.

### 2\. Advanced Ensemble Optimization

  * **Iterative Comparison:** The Random Forest Regressor significantly outperformed the simpler models.
  * **XGBoost Tuning:** The optimized 7-feature set was used to train a hyperparameter-tuned **XGBoost Regressor**, achieving the highest initial score.

### 3\. Spatial Feature Engineering & Deep Learning Benchmark

  * **K-Means Clustering:** Recognizing the high importance of `Latitude` and `Longitude`, **K-Means Clustering** was applied to the coordinates to group the state into **15 distinct geographic clusters**. This new categorical feature was used to capture non-linear, regional price effects, leading to the final performance gain.
  * **PyTorch Benchmark:** A **Multi-Layer Perceptron (MLP)** was implemented in **PyTorch** to benchmark the optimal classical ML results against a modern deep learning architecture, demonstrating proficiency in both frameworks.

## Final Performance Metrics

The iterative process led to a significant **$28.6\%$ reduction in prediction error (RMSE)** compared to the initial Decision Tree model.

| Model | Feature Count | RMSE | $R^2$ Score | $\Delta$ Improvement (RMSE Reduction) |
| :--- | :--- | :--- | :--- | :--- |
| Decision Tree (Baseline) | 11 | $0.6096$ | $0.7194$ | Baseline |
| XGBoost (Selected) | 7 | $0.4420$ | $0.8520$ | $+27.5\%$ |
| **XGBoost (with Clustering)** | **$7 + 14$** | **$0.4355$** | **$0.8567$** | **$+28.6\%$** |
| PyTorch MLP Regressor | $7 + 14$ | $0.6630$ | $0.6680$ | - |

> **Final Conclusion:** The optimized **XGBoost Regressor** with K-Means clustering achieved an $R^2$ score of **$0.8567$**, making it the top model for this task.

## Technical Stack

  * **Primary Language:** Python 3.9+
  * **Core Libraries:** `pandas`, `numpy`, `matplotlib`, `scikit-learn`
  * **Advanced ML:** `xgboost` (for ensemble boosting)
  * **Deep Learning:** **`PyTorch`** (for MLP benchmark and demonstrating low-level framework knowledge)
  * **Utilities:** `joblib` (for model persistence)
  * **Deployment Potential:** Designed for integration with **Gradio/Hugging Face Spaces**.

## Repository Structure

```
.
├── California_housing_price.ipynb  # Main analysis, training, and evaluation notebook
├── README.md                               # This file
├── cali_housing_best_xgb_model.joblib      # [Future] Saved final XGBoost model (joblib)
└── requirements.txt                        # List of project dependencies
```

##  How to Run Locally

### Prerequisites

Ensure you have Python 3.9 or higher installed.

```bash
# Clone the repository
git clone https://github.com/imranlabs/California_Housing_Price_Prediction.git
cd California_Housing_Price_Prediction
```

### Setup Environment

It is recommended to use a virtual environment.

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt
```

*(Note: You will need to create the `requirements.txt` file based on the libraries listed in the stack.)*

### Execution

Open the Jupyter Notebook and run all cells sequentially to reproduce the entire analysis, training, and evaluation pipeline.

```bash
jupyter notebook California_housing_data_project.ipynb
```

## Live Demo and Deployment

This project's final XGBoost model is ready for deployment. The next planned enhancement is a simple **Gradio** application demonstrating the model's predictions in real-time based on user input (Median Income, Latitude, Longitude, etc.).
