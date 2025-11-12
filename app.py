import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# --- Caching the Model ---
# This @st.cache_data thing is awesome.
# It basically tells Streamlit to run this function ONCE, load the model,
# and then save it in memory.
# So, it doesn't have to retrain the model every time I move a slider. Super fast.
@st.cache_data
def train_model():
    """Loads data, processes it, and trains a Linear Regression model."""
    try:
        # 1. Load Data
        # Reading the CSV file that's in the same folder.
        df = pd.read_csv('train.csv')

        # 2. Feature Engineering & Selection
        # The task needs "bathrooms", but the data splits it into 4 columns.
        # So, I'm just adding them all up. Half baths count as 0.5. Simple.
        df['TotalBathrooms'] = df['FullBath'] + (0.5 * df['HalfBath']) + df['BsmtFullBath'] + (0.5 * df['BsmtHalfBath'])
        
        # These are the 3 features the task asked for, plus the one we want to predict.
        features = ['GrLivArea', 'BedroomAbvGr', 'TotalBathrooms']
        target = 'SalePrice'

        # 3. Handle Missing Data
        # My 'TotalBathrooms' column had some NaN values (missing data).
        # I'm just filling those missing spots with the median (the "middle" value)
        # of all the other bathrooms. It's a simple fix.
        df['TotalBathrooms'] = df['TotalBathrooms'].fillna(df['TotalBathrooms'].median())
        
        # Just in case 'GrLivArea' or 'BedroomAbvGr' also have NaNs,
        # I'm creating a new, clean dataframe with just the columns I need
        # and dropping any rows that are still missing data.
        df_model = df[features + [target]].dropna()

        # Create our X (the inputs) and y (the price)
        X = df_model[features]
        y = df_model[target]

        # 4. Build & Train Model
        # This is the core of the task. Create the Linear Regression model
        # and "fit" it to my X and y data.
        model = LinearRegression()
        model.fit(X, y)
        
        # Return the trained model so the app can use it.
        return model

    except FileNotFoundError:
        st.error("Error: 'train.csv' not found. Make sure it's in the same folder.")
        return None

# --- Streamlit App ---
# This is where the app's frontend is built.

# Set up the title
st.title("🏡 House Price Predictor")
st.write("My app for Prodigy InfoTech Task 1!")
st.write("Enter the details of a house to predict its sale price.")
st.write("---")

# Load the trained model (it comes from the cached function)
model = train_model()

# Only run the app if the model loaded successfully
if model:
    # --- 1. User Inputs in Sidebar ---
    # st.sidebar puts all these widgets on the left side.
    st.sidebar.header("Enter House Features:")
    
    # number_input for square feet. Set some reasonable min/max/default values.
    sq_ft = st.sidebar.number_input(
        "Square Footage (GrLivArea)", 
        min_value=500, 
        max_value=6000, 
        value=1500, 
        step=50
    )
    
    # A slider just feels better for bedrooms and bathrooms.
    bedrooms = st.sidebar.slider(
        "Bedrooms (BedroomAbvGr)", 
        min_value=0, 
        max_value=8, 
        value=3, 
        step=1
    )
    
    bathrooms = st.sidebar.slider(
        "Total Bathrooms", 
        min_value=1.0, 
        max_value=6.0, 
        value=2.0, 
        step=0.5 # Step by 0.5 for half-baths
    )

    # --- 2. Make Prediction ---
    
    # I need to put the user's inputs into a pandas DataFrame,
    # because that's what the model was trained on.
    # The column names MUST match the ones I used for training.
    input_data = pd.DataFrame({
        'GrLivArea': [sq_ft],
        'BedroomAbvGr': [bedrooms],
        'TotalBathrooms': [bathrooms]
    })

    # Use the model to predict the price from the input_data.
    prediction = model.predict(input_data)
    
    # --- 3. Display Prediction ---
    st.header(f"Predicted Sale Price:")
    
    # Show the result in a nice green box!
    # prediction[0] gets the first (and only) number from the prediction array.
    # The :_ - format adds commas (like 150,000) and .2f rounds to 2 decimal places.
    st.success(f"**${prediction[0]:,.2f}**")