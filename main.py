# Sales Prediction Using Python

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

from sklearn.metrics import (
    mean_absolute_error,
    r2_score
)

# Load Dataset
df = pd.read_csv(
    "sales_data_samples.csv"
)

print("Dataset Shape:")
print(df.shape)

print("\nFirst 5 Rows:")
print(df.head())


# Features
X = df[
[
"Advertising_Spend",
"Target_Audience",
"Platform"
]
]

# Target
y = df["Sales"]


# Encoding
preprocess = ColumnTransformer(
[
(
"platform",
OneHotEncoder(),
["Platform"]
)
],
remainder="passthrough"
)


# Pipeline
model = Pipeline(
[
(
"preprocess",
preprocess
),
(
"model",
LinearRegression()
)
]
)


# Split
(
X_train,
X_test,
y_train,
y_test

)=train_test_split(
X,
y,
test_size=0.2,
random_state=42
)


# Train
model.fit(
X_train,
y_train
)


# Prediction
y_pred = model.predict(
X_test
)


# Metrics
print("\nResults")

print(
"MAE:",
round(
mean_absolute_error(
y_test,
y_pred
),
2
)
)

print(
"R2 Score:",
round(
r2_score(
y_test,
y_pred
),
2
)
)


# Example Prediction
sample = pd.DataFrame(
[
[
250000,
70000,
"Google"
]
],
columns=[
"Advertising_Spend",
"Target_Audience",
"Platform"
]
)

sales = model.predict(
sample
)

print(
"\nPredicted Sales:"
)

print(
round(
sales[0],
2
)
)


# Graph
plt.scatter(
y_test,
y_pred
)

plt.xlabel(
"Actual Sales"
)

plt.ylabel(
"Predicted Sales"
)

plt.title(
"Sales Prediction"
)

plt.show()