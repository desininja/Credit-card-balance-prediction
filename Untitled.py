import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px




import pickle
from PIL import Image

# file location:   /Users/HB/Desktop/python/Job-internship assessments/spark_grip


st.title("CREDIT CARD BALANCE PREDICTION MODEL")

def image_load():
    image = Image.open('study_image.jpg')
    st.image(image, caption='Know Your Balance!',use_column_width=True)

#image_load()

#@st.cache(persist=True)




def predictions(Income , Rating , Cards , Student):
    filename = 'finalized_model.sav'
    lm = pickle.load(open(filename, 'rb'))
    
    

    data = {'Income': [Income],'Rating': [Rating],'Cards': [Cards],'Student': [Student]}
    data = pd.DataFrame.from_dict(data)
    scalar = pickle.load(open('scaler_feature.pkl', 'rb'))
    scalar.fit(data) 
    scaled_data = scalar.transform(data)
    
    X1 = pd.DataFrame(data=scaled_data, columns=data.columns)
    
    pred = lm.predict(X1)
    if(pred<0):
        pred = -pred
    
    return pred
        

        
Income = st.number_input('Income')       

Rating = st.number_input('Rating')
Cards = st.number_input('Cards')


Student = st.selectbox('Please Enter Yes if you are student, otherwise No',('Yes','No'))
#st.write('You selected:', Student)
if Student == 'Yes':
    Student = 1
else:
    Student == 0
    

pred = predictions(Income , Rating , Cards , Student)
st.write("The Average Predicted Balance is",pred)
