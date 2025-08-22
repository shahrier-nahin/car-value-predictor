import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load model and data
model = pickle.load(open('LinearRegressionModel.pkl', 'rb'))
car = pd.read_csv('Cleaned_Car_data.csv')

# Prepare choices for form
companies = ["Select Company"] + sorted(car['company'].unique())
car_models = ["Select Model"] + sorted(car['name'].unique())
years = ["Select Year"] + sorted(car['year'].unique(), reverse=True)
fuel_types = ["Select Fuel Type"] + list(car['fuel_type'].unique())

st.title('Car Value Predictor')
st.write('Get an instant estimate of your car’s selling price. Fill in the details below to begin.')

company = st.selectbox('Select the company:', companies, index=0)
car_model = st.selectbox('Select the model:', car_models, index=0)
year = st.selectbox('Select Year of Purchase:', years, index=0)
fuel_type = st.selectbox('Select the Fuel Type:', fuel_types, index=0)
kms_driven = st.text_input('Enter the Number of Kilometres that the car has travelled:')

if st.button('Predict Price'):
    if company and car_model and year and fuel_type and kms_driven.isdigit():
        input_df = pd.DataFrame(
            columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'],
            data=np.array([car_model, company, year, int(kms_driven), fuel_type]).reshape(1,
                                                                                          5)
        )
        prediction = model.predict(input_df)
        st.success(f"Estimated Car Price: BDT {np.round(prediction[0], 2):,}")
    else:
        st.error("Please fill all fields correctly.")

