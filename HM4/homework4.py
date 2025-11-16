# homework4_full.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# --- 1. Load dataset ---
df = pd.read_csv('CarPrice.csv')

# Clean column names
df.columns = df.columns.str.strip()

# Drop unnecessary columns if they exist
df = df.drop(columns=[col for col in ['car_ID', 'CarName'] if col in df.columns])

# Encode categorical columns
categorical_cols = df.select_dtypes(include=['object']).columns
le = LabelEncoder()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# Separate features and target
X = df.drop('price', axis=1)
y = df['price']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 2. Decision Tree (baseline) ---
dt_model = DecisionTreeRegressor(max_depth=3, random_state=42)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)
print("Decision Tree:")
print(f"  MSE: {mean_squared_error(y_test, y_pred_dt):.2f}")
print(f"  R²: {r2_score(y_test, y_pred_dt):.2f}")

# Plot the tree
plt.figure(figsize=(15,8))
plot_tree(dt_model, feature_names=X.columns, filled=True, rounded=True, fontsize=10)
plt.title("Decision Tree for Car Price Prediction (Max Depth = 3)")
plt.show()

# --- 3. Bagging ---
bag_model = BaggingRegressor(
    estimator=DecisionTreeRegressor(max_depth=3),
    n_estimators=50,
    random_state=42
)
bag_model.fit(X_train, y_train)
y_pred_bag = bag_model.predict(X_test)
print("Bagging Regressor:")
print(f"  MSE: {mean_squared_error(y_test, y_pred_bag):.2f}")
print(f"  R²: {r2_score(y_test, y_pred_bag):.2f}")

# --- 4. Random Forest ---
rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=3,
    random_state=42
)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
print("Random Forest Regressor:")
print(f"  MSE: {mean_squared_error(y_test, y_pred_rf):.2f}")
print(f"  R²: {r2_score(y_test, y_pred_rf):.2f}")

# --- 5. Gradient Boosting ---
gb_model = GradientBoostingRegressor(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    random_state=42
)
gb_model.fit(X_train, y_train)
y_pred_gb = gb_model.predict(X_test)
print("Gradient Boosting Regressor:")
print(f"  MSE: {mean_squared_error(y_test, y_pred_gb):.2f}")
print(f"  R²: {r2_score(y_test, y_pred_gb):.2f}")

# --- 6. Neural Network (MLP) ---
# Scale features for neural network
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

nn_model = MLPRegressor(
    hidden_layer_sizes=(128, 64),
    activation='relu',
    solver='adam',
    max_iter=5000,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=50,
    random_state=42
)

nn_model.fit(X_train_scaled, y_train)
y_pred_nn = nn_model.predict(X_test_scaled)
print("Neural Network (MLP) Regressor:")
print(f"  MSE: {mean_squared_error(y_test, y_pred_nn):.2f}")
print(f"  R²: {r2_score(y_test, y_pred_nn):.2f}")
