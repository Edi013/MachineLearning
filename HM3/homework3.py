import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

df = pd.read_csv("breast-cancer-wisconsin.csv")

df.columns = df.columns.str.strip().str.lower()

df = df.drop(columns=["id"], errors="ignore")

df = df.apply(pd.to_numeric, errors="coerce")

# Although we have no empty/null rows in our training csv, we still need to include bypass for empty/null values
df = df.dropna()

print("=== First 5 rows ===")
print(df.head())

print("\n=== Missing values after cleaning ===")
print(df.isna().sum())

print("\n=== Min/Max per column ===")
print(df.describe().loc[["min", "max"]])

print("\n=== Class distribution ===")
print(df["class"].value_counts())

training_data = df.drop(columns=["class"])
training_data_target_column = df["class"]

X_train, X_test, y_train, y_test = train_test_split(training_data, training_data_target_column, test_size=0.2, random_state=42)

# Create SVM pipeline
model = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="rbf", C=1.0, gamma="scale"))
])

# Training
model.fit(X_train, y_train)

# Training Evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\n=== SVM Model Accuracy ===")
print(accuracy)

# Prediction
example = pd.DataFrame([{
    "clump thickness": 5,
    "uniformity of cell size": 1,
    "uniformity of cell shape": 1,
    "marginal adhesion": 1,
    "single epithelial cell size": 2,
    "bare nuclei": 1,
    "bland chromatin": 3,
    "normal nucleoli": 1,
    "mitoses": 1,
}])

pred = model.predict(example)[0]
print("\nExample prediction:", "Malignant (4)" if pred == 4 else "Benign (2)")
