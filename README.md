# Sales Prediction Using Python

## Project Overview

This project predicts future sales using Machine Learning based on:

- Advertising Spend
- Target Audience
- Platform

The goal is to understand how marketing factors affect sales outcomes and support business decisions.

---

## Objective

Build a regression-based machine learning model to:

✔ Predict future sales  
✔ Clean and prepare data  
✔ Encode categorical features  
✔ Evaluate model performance  
✔ Visualize prediction results  

---

## Dataset Information

Rows: 10,000

Columns:

- Advertising_Spend
- Target_Audience
- Platform
- Sales

Target Variable:

Sales

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib

---

## Machine Learning Algorithm

Linear Regression

---

## Workflow

1. Load Dataset
2. Data Cleaning
3. Feature Selection
4. Encode Platform Column
5. Train-Test Split
6. Train Regression Model
7. Predict Sales
8. Evaluate Performance
9. Visualize Results

---

## Model Performance

Dataset Shape:

(10000, 4)

Evaluation Results:

MAE:
2001.14

R² Score:
0.78

Example Predicted Sales:

17352.61

---

## Graph

Scatter Plot:

Actual Sales vs Predicted Sales

Closer points indicate better prediction.

---

## How to Run

Install dependencies:

pip install -r requirements.txt


Run:

python main.py

---

## Project Structure

SalesPrediction/

│

├── sales_data_samples.csv

├── main.py

├── requirements.txt

└── README.md

---

## Business Insights

- Higher advertising spend generally increases sales.
- Different platforms perform differently.
- Useful for planning marketing budgets.
- Supports data-driven business decisions.

---

## Future Improvements

- Add time-series forecasting
- Deploy using Streamlit
- Compare multiple regression models

---

## Author

Machine Learning Internship Project