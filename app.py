import pandas as pd
import numpy as np
import pickle as pk
import sklearn
import streamlit as st
from sklearn.preprocessing import StandardScaler

model = pk.load(open('model.pkl','rb'))

st.header('Credit Card User Clusters Prediction')

df = pd.read_csv('Clustered_data')

card = st.selectbox('Select Card type',df['Card_type'].unique())
city = st.selectbox('Select City',df['City'].unique())
gender = st.selectbox('Select Gender',df['Gender'].unique())
credit = st.slider('Select Credit Limit',0,1400)
balance = st.slider('Select Balance',0,4900)
payments = st.slider('Select payments',0,4200)
purchases = st.slider('Select Purchases',0,2800)
oneoffpurc = st.slider('Select Oneoff Purchases',0,1500)
installment = st.slider('Select Installment_purchases',0,1200)
cash = st.slider('Select Cash Avance',0,2800)
purchase_freq = st.selectbox('Select Purchase Frequency',df['Purchase_frequency'].unique())
oneoff_purc_freq = st.selectbox('Select Oneoff Purchase frequency',df['Oneoff_purchase_frequency'].unique())
installment_freq = st.selectbox('Select Purchase Installment frequency',df['Purchase_installment_frequency'].unique())


if st.button('Predict Clusters'):

    input = pd.DataFrame([[card,city,gender,credit,balance,payments,purchases,oneoffpurc,
                      installment,cash,purchase_freq,oneoff_purc_freq,installment_freq]],columns=['Card_type','City','Gender','Credit_limit','Balance','Payments','Purchases','Oneoff_purchases','Installment_purchases','Cash_advance','Purchase_frequency','Oneoff_purchase_frequency','Purchase_installment_frequency'])
    
    input['Card_type'] = input['Card_type'].replace(['Silver','Gold','Platinum','Titanium'],[1,2,3,4])
    input['City'] = input['City'].replace(['Pune','Kolkata','Delhi','Bengaluru','Chennai','Mumbai'],[1,3,5,4,2,6])
    input['Gender'] = input['Gender'].replace(['Male','Female'],[2,1])

    

    cluster = model.predict(input)
    st.markdown('The user belongs to Cluster : ' + str(cluster[0]))


