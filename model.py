import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv('insurance_prediction_app\\insurance.csv')

# Data Preprocessing
# Convert categorical variables to numerical: sex, smoker, region

def preprocess_data(data):
    
    # Define features and target
    X = data.drop('expenses', axis=1)
    y = data['expenses']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    categorical_features = ['sex', 'smoker', 'region']
    numerical_cols = [col for col in X.columns if col not in categorical_features]

    encoder = OneHotEncoder(drop='first')
    encoder.fit(X_train[categorical_features])
    X_train_encoded = encoder.transform(X_train[categorical_features]).toarray()
    X_test_encoded = encoder.transform(X_test[categorical_features]).toarray()
    

    # Sayısal sütunları al
    X_train_num = X_train[numerical_cols].values
    X_test_num = X_test[numerical_cols].values
    if X_train_num.ndim == 1:
        X_train_num = X_train_num.reshape(-1, 1)
    if X_test_num.ndim == 1:
        X_test_num = X_test_num.reshape(-1, 1)
    # Kategorik ve sayısal veriyi birleştir
    X_train_final = np.hstack([X_train_num, X_train_encoded])
    X_test_final = np.hstack([X_test_num, X_test_encoded])

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_final)
    X_test_scaled = scaler.transform(X_test_final)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, encoder, X.columns

    
def train_decision_tree(X_train, y_train):
    # Train Decision Tree Regressor model
    rf = RandomForestRegressor(
        max_depth=10, max_leaf_nodes=10, min_samples_split= 4, n_estimators= 500)
    rf.fit(X_train, y_train)

    return rf

def evaluate_model(model, X_test, y_test, feature_names):
    # Predict on the test set
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Model Performance:")
    print(f"Mean Squared Error: ${mse:.2f}")
    print(f"Root Mean Squared Error: ${rmse:.2f}")
    print(f"R² Score: {r2:.4f}")
    
    

    
    return rmse, r2
def predicted_prices(model, scaler, features):
    
    # Convert features to DataFrame
    features_df = pd.DataFrame([features])
    
    # Scale features
    features_scaled = scaler.transform(features_df)
    
    # Make prediction
    predicted_price = model.predict(features_scaled)[0]
    
    return predicted_price

def main():
    X_train, X_test, y_train, y_test, scaler, encoder, feature_names = preprocess_data(data)
    
    # Train models
    rf = train_decision_tree(X_train, y_train)
    
    print("\nEvaluating Random Forest Regressor Model:")
    evaluate_model(rf, X_test, y_test, feature_names)
    
if __name__ == "__main__":
    main()