# Healthcare Recommendation System

## Project Overview

This project is a Personalized Healthcare Recommendation System that leverages machine learning to predict common diseases based on key health parameters and provides tailored medicine recommendations. The system features an interactive Streamlit dashboard displaying predictions, health risk scores, and analytics.

## Features

- Disease prediction based on age, blood pressure, glucose level, and heart rate.
- Medicine recommendations customized for predicted disease conditions.
- Data visualization including health parameter gauges and dataset analytics.
- Light/Dark mode toggle with a clean, medical-themed interface.
- Dataset augmentation with synthetic data to improve model accuracy.

## Project Structure

- `dataset_expanded.csv`: Dataset used for model training, containing real and synthetic samples.
- `dataset_expander.py`: Script for augmenting the dataset with synthetic data.
- `disease_model.py`: Model training script utilizing logistic regression with cross-validation and feature scaling.
- `streamlit_app.py`: Streamlit web application for user interaction, prediction, and recommendations.
- `styles.css`: CSS file defining the application’s look and feel (used by the Streamlit app).
- `requirements.txt`: Python dependencies required for setup.
- `Personalized-Reco-System.txt`: Project overview and notes (optional).

## Setup and Usage

1. Clone this repository.
2. Create and activate a Python virtual environment.
3. Install dependencies:
```
pip install -r requirements.txt
```
4. Train the model (optional, if you want the latest model):
```
python disease_model.py
```
5. Run the Streamlit app:
```
streamlit run streamlit_app.py
```
6. Open the provided localhost link in your browser to use the app.

## Future Improvements

- Incorporate additional health features such as BMI, cholesterol levels, and lifestyle factors.
- Enhance model accuracy with advanced algorithms like XGBoost or neural networks.
- Develop user authentication and personalized profile management.
- Deploy the model behind an API for scalable and secure access.
- Add sentiment analysis on patient feedback for more refined recommendations.

## Disclaimer

This project is for educational and demonstration purposes only. It is not intended for medical diagnosis or treatment. Users should consult professional healthcare providers regarding medical decisions.

## Author

Project developed by Mohan Narayanapuram.  
For questions or feedback, please contact mohanvenkat017@gmail.com .
