# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 11:34:26 2024

@author: TEMPALLI HEMANTH
"""

import numpy as np
import pickle
import streamlit as st

loaded_model = pickle.load(open('C:/Users/TEMPALLI HEMANTH/Downloads/trained_model.sav', 'rb')) 


#creating function for prediction


def diabetes_prediction(input_data):
    """
    Predict if a person is diabetic based on input data.

    Parameters:
    input_data (tuple): A tuple containing feature values.

    Returns:
    str: The prediction result.
    """
    # Convert input data to a numpy array
    input_data_to_numpy_array = np.asarray(input_data)

    # Reshape the array for the model (1 instance with n features)
    reshaped_array = input_data_to_numpy_array.reshape(1, -1)

    # Make a prediction
    prediction = loaded_model.predict(reshaped_array)
    print("Prediction:", prediction)

    # Return the result based on the prediction
    if prediction[0] == 0:
        return "The person is not diabetic"
    else:
        return "The person is diabetic"



# Main function
def main():
    # Title of the app
    st.title("Diabetes Prediction Web App")
    
    # Input fields for user data
    Pregnancies = st.number_input("Number of Pregnancies")
    Glucose = st.number_input("Glucose Level")
    BloodPressure = st.number_input("Blood Pressure Level")
    SkinThickness = st.number_input("Skin Thickness")
    Insulin = st.number_input("Insulin Level")
    BMI = st.number_input("BMI (Body Mass Index)")
    DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function")
    Age = st.number_input("Age")
    
    # Button for prediction
    if st.button("Predict"):
        # Collect input data into a tuple
        input_data = (Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age)
        
        # Call the prediction function
        result = diabetes_prediction(input_data)
        
        # Display the result
        st.success(result)

# Run the main function
if __name__ == "__main__":
    main()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    