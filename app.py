import pandas as pd
import numpy as np 
import pickle as pk
import streamlit as st

model=pk.load(open("model.pk1",'rb'))
st.title("Car Price Prediction Using ML")
car_data=pd.read_csv('Cardetails.csv')

def get_brand(car_name):
    car_name=car_name.split(" ")[0]
    return car_name.strip()

car_data['name']=car_data['name'].apply(get_brand)
name=st.selectbox("Select Car Brand",car_data['name'].unique())
year=st.slider("Year of Manufacture",1994,2024)
km_driven=st.slider("Number of kms driven",11,120000)
fuel=st.selectbox("Fuel type",car_data['fuel'].unique())
seller_type=st.selectbox("Seller Type",car_data['seller_type'].unique())
transmission=st.selectbox("Transmission type",car_data['transmission'].unique())
owner=st.selectbox("Owner",car_data['owner'].unique())
mileage=st.slider("Car Mileage",10,40)
engine=st.slider("Car Engine CC",700,5000)
max_power=st.slider("Car Max power",0,200)
seats=st.slider("Number of Seats",5,10)

if st.button("Predict"):
    input_data_model=pd.DataFrame(
        [[name,year,km_driven,fuel,seller_type,transmission,owner,mileage,engine,max_power,seats]],
        columns=['name','year','km_driven','fuel','seller_type','transmission','owner','mileage','engine','max_power','seats'])
    #st.write(input_data_model)
    unique_names = car_data['name'].unique()
    input_data_model['name'].replace(unique_names, range(len(unique_names)), inplace=True)
    unique_trans=car_data['transmission'].unique()
    input_data_model['transmission'].replace(unique_trans, range(len(unique_trans)), inplace=True)
    unique_seller=car_data['seller_type'].unique()
    input_data_model['seller_type'].replace(unique_seller,range(len(unique_seller)),inplace=True)
    unique_fuel=car_data["fuel"].unique()
    input_data_model['fuel'].replace(unique_fuel,range(len(unique_fuel)),inplace=True)
    unique_owners = car_data['owner'].unique()
    input_data_model['owner'].replace(unique_owners, range(len(unique_owners)), inplace=True)

    #st.write(input_data_model)
    st.write(model.predict(input_data_model))



