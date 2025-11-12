### Prodigy InfoTech ML Internship - Task 1

Repository: ```PRODIGY_ML_01```

This repository contains my completed work for the first task of the Prodigy InfoTech Machine Learning Internship.

Task: Implement a linear regression model to predict the prices of houses based on their square footage and the number of bedrooms and bathrooms.

### 🚀 Live Demo

I have deployed this model as an interactive web app using Streamlit Community Cloud.

You can try the live app here:
https://vishwash-ml-task-01.streamlit.app/

### 📁 Files in this Repository

This project is divided into two main Python files:

1. ```task_01.py```

- This is the core script that fulfills the internship task.

- It loads the ```train.csv data.```

- It performs feature engineering by combining four bathroom columns into a single TotalBathrooms feature.

- It builds and trains a ```LinearRegression``` model on the ```GrLivArea``` (Square Footage), ```BedroomAbvGr``` (Bedrooms), and ```TotalBathrooms``` features.

- It prints the model's performance (R-squared and RMSE) to the terminal.

2. ```app.py```

- This is the interactive Streamlit web application.

- It loads the pre-trained model (using caching for speed).

- It provides a user-friendly interface in the sidebar for users to input square footage, bedrooms, and bathrooms.

- It takes the user's input, feeds it to the model, and displays the predicted house price in real-time.

### ⚙️ How to Run this Project Locally

To run this project on your own machine, please follow these steps:

1. Clone the repository:
```
git clone https://github.com/Redoxftw/PRODIGY_ML_01.git
```


2. Create and activate a virtual environment:
```
# Create the venv
python -m venv venv

# Activate on Windows (cmd)
.\venv\Scripts\activate
```

3. Install the required libraries:
```
pip install -r requirements.txt
```

To Run the Terminal Script:

This will run the original task and print the model's R-squared and RMSE scores.
```
python task_01.py
```

To Run the Interactive Web App:

This will launch the app in your browser.
```
streamlit run app.py
```