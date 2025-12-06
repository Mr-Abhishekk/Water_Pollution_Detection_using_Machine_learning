Water Pollution Detection Using Machine Learning

This project uses machine learning to predict the quality of water and classify it as Safe, Moderate, or Contaminated. By entering common water parameters like pH, BOD, DO, and Coliform levels, the system can instantly estimate the pollution level and help users understand the condition of the water.

It also comes with a user-friendly Streamlit web app where users can manually enter values or search predictions based on state, location, and year from the dataset.

*What This Project Can Do:

Predict water quality using trained ML models

Compare river sample values with safe environmental limits

View charts for physicochemical and biological parameters

Search water data by State → Location → Year

Clean and interactive Streamlit interface

Includes multiple ML models: Random Forest, Decision Tree, Logistic Regression, and KNN

*Tech Used:

1)Python

2)Scikit-Learn

3)Pandas, NumPy

4)Streamlit

5)Altair Charts

6)Joblib

7)Matplotlib & Seaborn

*Project Structure:
water-pollution-detection-ml/
1)streamlit_app.py                # Main Streamlit App
2)requirements.txt                # Python Dependencies
3)README.md                       # Documentation
4)model/
   4.1)dt_model_pipeline.joblib    # Trained ML Model
    4.2)label_encoder.joblib        # Encoder for labels
5)data/
   5.1)water_data.csv              # Dataset
6)notebooks/
     WaterPollution_Project.ipynb   # Complete ML Pipeline Notebook

*Project Description:

Water quality plays a huge role in public health, agriculture, and the environment. However, manually checking water safety is time-consuming and requires lab testing.
This project aims to make that process faster by using machine learning to estimate the pollution level instantly.

You can:

Enter the water parameters manually

Select a real location from the dataset

Get a prediction with clear color-coded results

View charts to understand why the water is classified as Safe, Moderate, or Contaminated

It’s a simple, practical tool for learning, research, or environmental analysis.

*How to Run the App
1️)Install the required packages:
2)pip install -r requirements.txt

Start the Streamlit app:
streamlit run streamlit_app.py

Future Improvements:

Real-time sensor integration

Cloud-hosted dashboard

Add water quality index (WQI) score

Mobile-friendly version


Developed by-
Abhishek Munavalli
Project developed as part of the Green AI Training Program (Skills4Future – Edunet Foundation).

