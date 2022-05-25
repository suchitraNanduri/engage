
import streamlit as st
import pickle
import numpy as np
import pandas as pd
#load the model and dataframe
df = pd.read_csv("df2.csv")
pipe = pickle.load(open("pipe1.pkl", "rb"))
st.title("Car Price Predictor")
#Now we will take user input one by one as per our dataframe
company = df['Make'].unique() #list of all brands
company = list(company)
n = df.iloc[:,0:2]
list1=[]
list2=[]
for i in company:
    list1.append(i)
    list2.append( n.value_counts()[i].index.tolist())
#Brand
#company = st.selectbox('Brand', df['Company'].unique())
Brand = st.selectbox('Brand', df['Make'].unique())
#Type of Model
ind = list1.index(Brand)
    
Model = st.selectbox("Model", list2[ind])#list of only models of the brand selected
#Displacement
Displacement = st.number_input("Displacement(in cc)")
#Cylinders
Cylinders = st.selectbox("No.of Cylinders",df['Cylinders'].unique())
#Where engine located
Engine_Location = st.selectbox("Location of Engine", df['Engine_Location'].unique() )
#Fuel_Tank_Capacity
Tank_Capacity = st.number_input("Fuel Tank Capacity")
#Fuel_Type
Fuel_Type = st.selectbox('Type of Fuel',df['Fuel_Type'].unique())
#Body_Type
Body_Type = st.selectbox('Body of Car',df['Body_Type'].unique())
#Type
Type_ofcar = st.selectbox('Type of car',df['Type'].unique())
#Prediction
if st.button('Predict Price'):
    
    
    query = np.array([Brand,Model,Displacement,Cylinders,Engine_Location,Tank_Capacity,Fuel_Type,Body_Type,Type_ofcar])
    query = query.reshape(1, 9)
    prediction = str(int(np.exp(pipe.predict(query)[0])))
    st.title("The predicted price of this configuration is " + prediction)
