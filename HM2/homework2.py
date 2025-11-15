import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

data = pd.read_csv("tips.csv")

training_data = data[["total_bill", "sex", "smoker", "day", "time", "size"]]
training_data_tip_column = data["tip"]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(drop="first"), ["sex", "smoker", "day", "time"]),
        ("num", "passthrough", ["total_bill", "size"]),
    ]
)

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

model.fit(training_data, training_data_tip_column)

example_df = pd.DataFrame([{
    "total_bill": 2220.34,
    "sex": "Male",
    "day": "Sun",
    "time": "Dinner",
    "size": 111,
    "smoker": "Yes"
}])

print("Example to predict ---------------------")
print(example_df.to_string())
print("----------------------------------------")
predicted_tip = model.predict(example_df)[0]
print("Predicted tip:", predicted_tip)
