import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_model():
    with open ("Model File",'rb') as file:
        model,scaler,le = pickle.load(file)
    return model, scaler, le 

def preprocessing_input_data(data,scaler, le ):
    data["Extracurricular Activities"]= le.transform([data["Extracurricular Activities"]])[0]
    df= pd.DataFrame([data])
    df_transformed = scaler.transform(df)
    return df_transformed

def predict(data):
    model,scaler,le = load_model()
    processed_data = preprocessing_input_data(data,scaler,le)
    prediction = model.predict(processed_data)
    return prediction
 
def main():
    st.title("Student Performance Prediction")
    st.write("Enter your data to get prediction for your performance")
    hours = st.number_input("Houes studies",min_value= 1, max_value=10, value = 5)
    previous = st.number_input("Previous Scores", min_value=1,max_value=100, value = 50)
    extra = st.selectbox("Extracurricular Activities",["Yes","No"])
    sleep = st.number_input("Sleeping hours", min_value=4,max_value=20, value = 5)    
    sample_paper = st.number_input("Sample Question Papers Practiced", min_value=0,max_value=20, value = 5)    

    if st.button("Predict Your Score"):
        user_data = {
            "Hours Studied" : hours,
            "Previous Scores" : previous,
            "Extracurricular Activities": extra,
            "Sleep Hours" : sleep,
            "Sample Question Papers Practiced" :sample_paper

        }

        prediction= predict(user_data)
        st.success(f"Your Prediction Results {prediction[0]:.2f}")

if __name__ == "__main__" :
    main()