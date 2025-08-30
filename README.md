Melbourne House Price Prediction Project

<img width="588" height="647" alt="image" src="https://github.com/user-attachments/assets/f6f0c9bc-aaa4-484e-b845-907121d19178" />


📋 1. Project Overview
This project demonstrates a complete, end-to-end machine learning workflow to accurately predict house prices in Melbourne, Australia. The process includes thorough data cleaning, in-depth exploratory data analysis (EDA), advanced feature engineering, and hyperparameter tuning of an XGBoost model. The final, high-performance model is deployed as a user-friendly, interactive web application using Streamlit.

Live Application: https://melbournepriceprediction.streamlit.app

✨ 2. Key Features
Advanced Data Cleaning: Handled significant missing values using sophisticated imputation techniques, such as filling missing Landsize and YearBuilt with the median values of their respective suburbs.

In-Depth Exploratory Data Analysis: Conducted a comprehensive EDA to uncover market trends. Visualizations included:

A geospatial map using Folium to plot property prices across Melbourne.

A correlation heatmap to understand relationships between numerical features.

Time-series analysis to track median prices over time.

Strategic Feature Engineering: Created impactful new features, such as Property Age (from YearBuilt) and applied log transformations to skewed variables (Price, Landsize) to improve model stability and performance.

High-Performance Modeling: Utilized an XGBoost Regressor, with its hyperparameters optimized using RandomizedSearchCV to maximize predictive accuracy on unseen data.

Interactive Deployment: Built and deployed a user-friendly Streamlit web application that allows for real-time price predictions based on user-selected property features.

🚀 3. Model Performance
The final tuned model achieved the following performance on the unseen test set:

R-squared (R²): 0.8842 (The model successfully explains 88.4% of the variance in house prices).

Mean Absolute Error (MAE): ~$138,000 (The model's predictions are, on average, off by this amount).

💻 4. Technology Stack
Language: Python

Libraries:

Data Manipulation & Analysis: Pandas, NumPy

Machine Learning: Scikit-learn, XGBoost

Data Visualization: Matplotlib, Seaborn, Folium

Web Application: Streamlit

Model Persistence: Joblib

🛠️ 5. How to Run This Project
To run the application locally, please follow these steps:

Clone the repository:

git clone https://github.com/chethansheelam/melbourne-price-prediction.git
cd melbourne-price-prediction

Install the required dependencies:
It is recommended to use a virtual environment.

pip install -r requirements.txt

Run the Streamlit application:

streamlit run app.py

The application will open in your default web browser.
