# Singapore-Flat-Resale
![Singapore Flast Resale](https://github.com/srisuryaprakash55/Singapore-Flat-Resale/assets/139371882/8b0e679a-856f-4a37-bd03-203ca381bc1a)
Singapore Resale Flat Price Prediction
Overview
This project aims to develop a machine learning model that predicts the resale prices of flats in Singapore based on historical transaction data. Additionally, a user-friendly web application will be deployed to assist potential buyers and sellers in estimating the resale value of flats.

Motivation
The resale flat market in Singapore is highly competitive, making it challenging to accurately estimate resale values. This project addresses this challenge by leveraging machine learning to provide estimated resale prices based on various factors such as location, flat type, floor area, and lease duration.

Project Tasks
1. Data Collection and Preprocessing
Gather resale flat transaction data from the Singapore Housing and Development Board (HDB) spanning from 1990 to the present.
Clean and structure the data for machine learning by performing necessary preprocessing steps.
2. Feature Engineering
Extract relevant features from the dataset, including town, flat type, storey range, floor area, flat model, and lease commence date.
Create additional features to enhance prediction accuracy if needed.
3. Model Selection and Training
Choose an appropriate regression model (e.g., linear regression, decision trees, random forests).
Train the model on historical data, utilizing a portion for training purposes.
4. Model Evaluation
Assess the model's predictive performance using metrics like Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R2 Score.
5. Streamlit Web Application
Develop a user-friendly web application using Streamlit to allow users to input flat details and obtain resale price predictions using the trained model.
6. Deployment
Deploy the Streamlit application on the Render platform or any Cloud Platform for accessibility over the internet.
7. Testing and Validation
Thoroughly test the deployed application to ensure accurate predictions and proper functionality.
Deliverables
Well-trained machine learning model for resale price prediction.
User-friendly web application deployed on a platform for accessibility.
Documentation and instructions for using the application.
Project report summarizing data analysis, model development, and deployment.
Technologies Used
Python
Pandas, NumPy, Scikit-learn
Streamlit / Flask / Django
Render / Any Cloud Platform
Usage
Clone the repository.
Follow instructions in the documentation to set up and run the application locally or access the deployed web application link.
Project Structure
data/: Contains the dataset and processed data.
models/: Includes trained machine learning models.
web_app/: Code for the Streamlit web application.
documentation/: Instructions and documentation files.
