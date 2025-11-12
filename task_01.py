# --- Task 1: House Price Prediction ---
# This is my main script for the first task.
# The goal is to build a simple Linear Regression model.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import warnings

# Just adding this to ignore any warnings, makes the output cleaner.
warnings.filterwarnings('ignore')

print("--- Running Task 1: House Price Prediction Model ---")

try:
    # --- 1. Load Data ---
    # Load the training data from the CSV file.
    # It's in the same folder, so just the filename is needed.
    df = pd.read_csv('train.csv')
    print("Step 1: Data loaded successfully.")

    # --- 2. Feature Engineering & Selection ---
    # The task asked for "bathrooms", but the dataset has 4 different columns for them!
    # So, I'm creating my own "TotalBathrooms" feature.
    # I'll count half-baths as 0.5 and add them all together.
    df['TotalBathrooms'] = df['FullBath'] + (0.5 * df['HalfBath']) + df['BsmtFullBath'] + (0.5 * df['BsmtHalfBath'])

    # These are the 3 features (inputs) I'm using, as per the task:
    # 'GrLivArea' = Square Footage
    # 'BedroomAbvGr' = Bedrooms
    # 'TotalBathrooms' = My new bathrooms feature
    features = ['GrLivArea', 'BedroomAbvGr', 'TotalBathrooms']
    
    # This is the target (output) I'm trying to predict:
    target = 'SalePrice'
    print("Step 2: 'TotalBathrooms' feature created.")

    # --- 3. Handle Missing Data ---
    # I checked, and my new 'TotalBathrooms' column had a couple of missing (NaN) values.
    # To fix this, I'm just filling those empty spots with the "median" (middle value)
    # of the whole column. Seems like a safe and simple way to handle it.
    df['TotalBathrooms'] = df['TotalBathrooms'].fillna(df['TotalBathrooms'].median())
    print("Step 3: Missing bathroom data handled.")

    # Now, I'll create my final X (inputs) and y (output)
    X = df[features]
    y = df[target]

    # --- 4. Split Data ---
    # I need to test my model on data it hasn't seen before.
    # So, I'm splitting my dataset:
    # 80% for training the model
    # 20% for testing the model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Step 4: Data split: {len(X_train)} training samples, {len(X_test)} testing samples.")

    # --- 5. Build & Train Model ---
    # Here's the core of the task:
    # 1. Create a LinearRegression model object
    # 2. Train it ("fit" it) using my 80% training data
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("Step 5: Model trained successfully.")

    # --- 6. Evaluate Model ---
    # Now, I'll use the model to make predictions on my 20% "test" data.
    y_pred = model.predict(X_test)

    # Let's see how good the model is...
    # R-squared: How much of the price change my model can explain (e.g., 0.65 = 65%)
    # RMSE: The average error of my model's price prediction (in dollars)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print("\n--- Model Evaluation Results ---")
    print(f"  R-squared (R2): {r2:.3f}")
    print(f"  Root Mean Squared Error (RMSE): ${rmse:,.2f}")

    # --- 7. View Coefficients ---
    # This shows what the model "learned".
    # e.g., for every 1 sq. ft. increase, the price goes up by $X.
    print("\n--- Model Coefficients ---")
    print(f"  Intercept (base price): ${model.intercept_:.2f}")
    print(f"  Living Area (per sq. ft.): ${model.coef_[0]:.2f}")
    print(f"  Bedrooms (per bedroom): ${model.coef_[1]:.2f}")
    print(f"  Total Bathrooms (per bath): ${model.coef_[2]:.2f}")


except FileNotFoundError:
    print("\n--- ERROR ---")
    print("Error: 'train.csv' not found. Bummer.")
    print("Please make sure 'train.csv' is in the same folder as this script.")
except Exception as e:
    # Catch any other weird errors
    print(f"\nAn error occurred: {e}")