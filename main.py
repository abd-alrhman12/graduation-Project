#!/usr/bin/env python
# coding: utf-8

# In[7]:


import webbrowser
import streamlit as st 
import streamlit.components.v1 as components
# from streamlit.components.v1 import html
import pandas as pd
from sklearn.preprocessing import  LabelEncoder
import xgboost as xgb
import numpy as np
import random
import keras
from PIL import Image

model = keras.models.load_model('model.h5')

st.header("illnesses Prediction App")
st.text_input("Enter your Name: ", key="name")
from sklearn.preprocessing import OrdinalEncoder

data = pd.read_csv("data.csv",index_col=0)
ordinal_encoder = OrdinalEncoder()
data[['hasAlzheimerDisease ']] = ordinal_encoder.fit_transform(data[['hasAlzheimerDisease ']])
data[['hasThyroidDisorders ']] = ordinal_encoder.fit_transform(data[['hasThyroidDisorders ']])
data[['hasSpinaBifida ']] = ordinal_encoder.fit_transform(data[['hasSpinaBifida ']])
data[['hasMultipleSclerosis ']] = ordinal_encoder.fit_transform(data[['hasMultipleSclerosis ']])
data[['hasHighBloodPressure ']] = ordinal_encoder.fit_transform(data[['hasHighBloodPressure ']])
data[['hasArthritis ']] = ordinal_encoder.fit_transform(data[['hasArthritis ']])
data[['hasCancer ']] = ordinal_encoder.fit_transform(data[['hasCancer ']])
data[['hasDementia ']] = ordinal_encoder.fit_transform(data[['hasDementia ']])
data[['hasDiabetes ']] = ordinal_encoder.fit_transform(data[['hasDiabetes ']])
data[['hasParkinsonDisease ']] = ordinal_encoder.fit_transform(data[['hasParkinsonDisease ']])
data[['hasHeartDisease ']] = ordinal_encoder.fit_transform(data[['hasHeartDisease ']])


selected_options_option1 = set()
selected_options_option2=set()
selected_options_option3=set()
selected_options_option4=set()
selected_options_option5=set()
selected_options_option6=set()
selected_options_option7=set()
selected_options_option8=set()
selected_options_option9=set()
selected_options_option10=set()
selected_options_option11=set()
encoder = LabelEncoder()
encoder.classes_ = np.load('classes.npy',allow_pickle=True)
best_xgboost_model = xgb.XGBRegressor()
best_xgboost_model.load_model("best_model.json")
if st.checkbox('Show dataframe'):
    data
css = """
<link href="styles.css" rel="stylesheet">
<style>
  body {
        background-image: url("imageEdit.jpg");
        background-size: cover;


    }
</style>
"""
st.markdown("___family infromation:___")
st.text("father")
with st.container():

 diseases = [
"AlzheimerDisease","ArthritisDisease","Cancerdisease"
,"Dementiadisease","Diabetesdisease","HeartDisease",
"HighBloodPressure ","MultipleSclerosis disease","ParkinsonDisease","SpinaBifidadisease","ThyroidDisordersdisease "
]
selected_options_option_mother=set()

option_Mother=["AlzheimerDisease","ArthritisDisease","Cancerdisease"
,"Dementiadisease","Diabetesdisease","HeartDisease",
"HighBloodPressure ","MultipleSclerosis disease","ParkinsonDisease","SpinaBifidadisease"
,"ThyroidDisordersdisease "]
selected_diseases = []
with st.expander("Show Checkboxes"):
    
    checkbox1s = st.checkbox(diseases[0].split()[0],key="unique")
    checkbox2s = st.checkbox(diseases[1].split()[0],key=1)
    checkbox3s = st.checkbox(diseases[2].split()[0],key=11)
    checkbox4s = st.checkbox(diseases[3].split()[0],key=111)
    checkbox5s = st.checkbox(diseases[4].split()[0],key=125)
    checkbox6s = st.checkbox(diseases[5].split()[0],key=165)
    checkbox7s = st.checkbox(diseases[6].split()[0],key=157)
    checkbox8s = st.checkbox(diseases[7].split()[0],key=174)
    checkbox9s = st.checkbox(diseases[8].split()[0],key=1589)
    checkbox10s = st.checkbox(diseases[9].split()[0],key=1724)
    checkbox11s = st.checkbox(diseases[10].split()[0],key=1474)
percentage_checkbox1_fahter = (1 if checkbox1s else 0) * 25
percentage_checkbox2 = (1 if checkbox2s else 0) * 25
percentage_checkbox3 = (1 if checkbox3s else 0) * 25
percentage_checkbox4 = (1 if checkbox4s else 0) * 25
percentage_checkbox5 = (1 if checkbox5s else 0) * 25
percentage_checkbox6 = (1 if checkbox6s else 0) * 25
percentage_checkbox7 = (1 if checkbox7s else 0) * 25
percentage_checkbox8 = (1 if checkbox8s else 0) * 25
percentage_checkbox9 = (1 if checkbox9s else 0) * 25
percentage_checkbox10 = (1 if checkbox10s else 0) * 25
percentage_checkbox11 = (1 if checkbox11s else 0) * 25
    
if checkbox1s:
    selected_diseases.append(diseases[0])
if checkbox2s:
    selected_diseases.append(diseases[1])
if checkbox3s:
    selected_diseases.append(diseases[2])
if checkbox4s:
    selected_diseases.append(diseases[3])
if checkbox5s:
    selected_diseases.append(diseases[4])
if checkbox6s:
    selected_diseases.append(diseases[5])
if checkbox7s:
    selected_diseases.append(diseases[6])
if checkbox8s:
    selected_diseases.append(diseases[7])
if checkbox9s:
    selected_diseases.append(diseases[8])
if checkbox10s:
    selected_diseases.append(diseases[9])
if checkbox11s:
    selected_diseases.append(diseases[10])
selected_diseases_mother = []
st.text("mother")
with st.expander("Show Checkboxes"):

    checkbox1_mother = st.checkbox(option_Mother[0].split()[0])
    checkbox2_mother = st.checkbox(option_Mother[1].split()[0])
    checkbox3_mother = st.checkbox(option_Mother[2].split()[0])
    checkbox4_mother = st.checkbox(option_Mother[3].split()[0])
    checkbox5_mother = st.checkbox(option_Mother[4].split()[0])
    checkbox6_mother = st.checkbox(option_Mother[5].split()[0])
    checkbox7_mother = st.checkbox(option_Mother[6].split()[0])
    checkbox8_mother = st.checkbox(option_Mother[7].split()[0])
    checkbox9_mother = st.checkbox(option_Mother[8].split()[0])
    checkbox10_mother = st.checkbox(option_Mother[9].split()[0])
    checkbox11_mother = st.checkbox(option_Mother[10].split()[0])

percentage_checkbox1_mother = (1 if checkbox1_mother else 0) * 25
percentage_checkbox2_mother = (1 if checkbox2_mother else 0) * 25
percentage_checkbox3_mother = (1 if checkbox3_mother else 0) * 25
percentage_checkbox4_mother = (1 if checkbox4_mother else 0) * 25
percentage_checkbox5_mother = (1 if checkbox5_mother else 0) * 25
percentage_checkbox6_mother = (1 if checkbox6_mother else 0) * 25
percentage_checkbox7_mother = (1 if checkbox7_mother else 0) * 25
percentage_checkbox8_mother = (1 if checkbox8_mother else 0) * 25
percentage_checkbox9_mother = (1 if checkbox9_mother else 0) * 25
percentage_checkbox10_mother = (1 if checkbox10_mother else 0) * 25
percentage_checkbox11_mother = (1 if checkbox11_mother else 0) * 25
if checkbox1_mother:
    selected_diseases_mother.append(option_Mother[0])
if checkbox2_mother:
    selected_diseases_mother.append(option_Mother[1])
if checkbox3_mother:
    selected_diseases_mother.append(option_Mother[2])
if checkbox4_mother:
    selected_diseases_mother.append(option_Mother[3])
if checkbox5_mother:
    selected_diseases_mother.append(option_Mother[4])
if checkbox6_mother:
    selected_diseases_mother.append(option_Mother[5])
if checkbox7_mother:
    selected_diseases_mother.append(option_Mother[6])
if checkbox8_mother:
    selected_diseases_mother.append(option_Mother[7])
if checkbox9_mother:
    selected_diseases_mother.append(option_Mother[8])
if checkbox10_mother:
    selected_diseases_mother.append(option_Mother[9])
if checkbox11_mother:
    selected_diseases_mother.append(option_Mother[10])
st.text("My mother's father")
motherfather=["AlzheimerDisease","ArthritisDisease","Cancerdisease"
,"Dementiadisease","Diabetesdisease","HeartDisease",
"HighBloodPressure ","MultipleSclerosis disease","ParkinsonDisease","SpinaBifidadisease","ThyroidDisordersdisease "]
selected_motherfather = []
with st.expander("Show Checkboxes"):
    checkbox1_motherfather = st.checkbox(motherfather[0].split()[0] ,key=55)
    checkbox2_motherfather = st.checkbox(motherfather[1].split()[0] ,key=555)
    checkbox3_motherfather = st.checkbox(motherfather[2].split()[0] ,key=547)
    checkbox4_motherfather = st.checkbox(motherfather[3].split()[0], key=45)
    checkbox5_motherfather = st.checkbox(motherfather[4].split()[0] ,key=758)
    checkbox6_motherfather = st.checkbox(motherfather[5].split()[0] ,key=748)
    checkbox7_motherfather = st.checkbox(motherfather[6].split()[0] ,key=41)
    checkbox8_motherfather = st.checkbox(motherfather[7].split()[0],key=78)
    checkbox9_motherfather = st.checkbox(motherfather[8].split()[0] ,key=89)
    checkbox10_motherfather = st.checkbox(motherfather[9].split()[0] ,key=5498)
    checkbox11_motherfather = st.checkbox(motherfather[10].split()[0] ,key=3269)
    percentage_motherfather1 = (1 if checkbox1_motherfather else 0) * 25
    percentage_motherfather2 = (1 if checkbox2_motherfather else 0) * 25
    percentage_motherfather3 = (1 if checkbox3_motherfather else 0) * 25
    percentage_motherfather4 = (1 if checkbox4_motherfather else 0) * 25
    percentage_motherfather5 = (1 if checkbox5_motherfather else 0) * 25
    percentage_motherfather6 = (1 if checkbox6_motherfather else 0) * 25
    percentage_motherfather7 = (1 if checkbox7_motherfather else 0) * 25
    percentage_motherfather8 = (1 if checkbox8_motherfather else 0) * 25
    percentage_motherfather9 = (1 if checkbox9_motherfather else 0) * 25
    percentage_motherfather10 = (1 if checkbox10_motherfather else 0) * 25
    percentage_motherfather11 = (1 if checkbox11_motherfather else 0) * 25
if checkbox1_motherfather:
    selected_motherfather.append(motherfather[0])
if checkbox2_motherfather:
    selected_motherfather.append(motherfather[1])
if checkbox3_motherfather:
    selected_motherfather.append(motherfather[2])
if checkbox4_motherfather:
    selected_motherfather.append(motherfather[3])
if checkbox5_motherfather:
    selected_motherfather.append(motherfather[4])
if checkbox6_motherfather:
    selected_motherfather.append(motherfather[5])
if checkbox7_motherfather:
    selected_motherfather.append(motherfather[6])
if checkbox8_motherfather:
    selected_motherfather.append(motherfather[7])
if checkbox9_motherfather:
    selected_motherfather.append(motherfather[8])
if checkbox10_motherfather:
    selected_motherfather.append(motherfather[9])
if checkbox11_motherfather:
    selected_motherfather.append(motherfather[10])

st.text("my mother's mother")
mothermother=["AlzheimerDisease","ArthritisDisease","Cancerdisease"
,"Dementiadisease","Diabetesdisease","HeartDisease",
"HighBloodPressure ","MultipleSclerosis disease","ParkinsonDisease","SpinaBifidadisease","ThyroidDisordersdisease "]
selected_mothermother = []
with st.expander("Show Checkboxes"):

 checkbox1_mothermother = st.checkbox(mothermother[0].split()[0] ,key=132)
 checkbox2_mothermother = st.checkbox(mothermother[1].split()[0] ,key=133)
 checkbox3_mothermother = st.checkbox(mothermother[2].split()[0] ,key=000)
 checkbox4_mothermother = st.checkbox(mothermother[3].split()[0], key=124)
 checkbox5_mothermother = st.checkbox(mothermother[4].split()[0] ,key=777)
 checkbox6_mothermother = st.checkbox(mothermother[5].split()[0] ,key=888)
 checkbox7_mothermother = st.checkbox(mothermother[6].split()[0] ,key=999)
 checkbox8_mothermother = st.checkbox(mothermother[7].split()[0],key=3658)
 checkbox9_mothermother = st.checkbox(mothermother[8].split()[0] ,key=1475)
 checkbox10_mothermother = st.checkbox(mothermother[9].split()[0] ,key=3697)
 checkbox11_mothermother = st.checkbox(mothermother[10].split()[0] ,key=2568)
percentage_mothermother1 = (1 if checkbox1_mothermother else 0) * 25
percentage_mothermother2 = (1 if checkbox2_mothermother else 0) * 25
percentage_mothermother3 = (1 if checkbox3_mothermother else 0) * 25
percentage_mothermother4 = (1 if checkbox4_mothermother else 0) * 25
percentage_mothermother5 = (1 if checkbox5_mothermother else 0) * 25
percentage_mothermother6 = (1 if checkbox6_mothermother else 0) * 25
percentage_mothermother7 = (1 if checkbox7_mothermother else 0) * 25
percentage_mothermother8 = (1 if checkbox8_mothermother else 0) * 25
percentage_mothermother9 = (1 if checkbox9_mothermother else 0) * 25
percentage_mothermother10 = (1 if checkbox10_mothermother else 0) * 25
percentage_mothermother11 = (1 if checkbox11_mothermother else 0) * 25
if checkbox1_mothermother:
    selected_mothermother.append(mothermother[0])
if checkbox2_mothermother:
    selected_mothermother.append(mothermother[1])
if checkbox3_mothermother:
    selected_mothermother.append(mothermother[2])
if checkbox4_mothermother:
    selected_mothermother.append(mothermother[3])
if checkbox5_mothermother:
    selected_mothermother.append(mothermother[4])
if checkbox6_mothermother:
    selected_mothermother.append(mothermother[5])
if checkbox7_mothermother:
    selected_mothermother.append(mothermother[6])
if checkbox8_mothermother:
    selected_mothermother.append(mothermother[7])
if checkbox9_mothermother:
    selected_mothermother.append(mothermother[8])
if checkbox10_mothermother:
    selected_mothermother.append(mothermother[9])
if checkbox11_mothermother:
    selected_mothermother.append(mothermother[10])

st.text("my father's father")
fatherfather=["AlzheimerDisease","ArthritisDisease","Cancerdisease"
,"Dementiadisease","Diabetesdisease","HeartDisease",
"HighBloodPressure ","MultipleSclerosis disease","ParkinsonDisease","SpinaBifidadisease","ThyroidDisordersdisease "]
selected_fatherfather = []
with st.expander("Show Checkboxes"):
 checkbox1_fatherfather = st.checkbox(fatherfather[0].split()[0] ,key=1132)
 checkbox2_fatherfather = st.checkbox(fatherfather[1].split()[0] ,key=1133)
 checkbox3_fatherfather = st.checkbox(fatherfather[2].split()[0] ,key=1000)
 checkbox4_fatherfather = st.checkbox(fatherfather[3].split()[0], key=1124)
 checkbox5_fatherfather = st.checkbox(fatherfather[4].split()[0] ,key=1777)
 checkbox6_fatherfather = st.checkbox(fatherfather[5].split()[0] ,key=1888)
 checkbox7_fatherfather = st.checkbox(fatherfather[6].split()[0] ,key=1999)
 checkbox8_fatherfather = st.checkbox(fatherfather[7].split()[0],key=13658)
 checkbox9_fatherfather = st.checkbox(fatherfather[8].split()[0] ,key=11475)
 checkbox10_fatherfather = st.checkbox(fatherfather[9].split()[0] ,key=13697)
 checkbox11_fatherfather = st.checkbox(fatherfather[10].split()[0] ,key=12568)
 percentage_fatherfather1 = (1 if checkbox1_fatherfather else 0) * 25
 percentage_fatherfather2 = (1 if checkbox2_fatherfather else 0) * 25
 percentage_fatherfather3 = (1 if checkbox3_fatherfather else 0) * 25
 percentage_fatherfather4 = (1 if checkbox4_fatherfather else 0) * 25
 percentage_fatherfather5 = (1 if checkbox5_fatherfather else 0) * 25
 percentage_fatherfather6 = (1 if checkbox6_fatherfather else 0) * 25
 percentage_fatherfather7 = (1 if checkbox7_fatherfather else 0) * 25
 percentage_fatherfather8 = (1 if checkbox8_fatherfather else 0) * 25
 percentage_fatherfather9 = (1 if checkbox9_fatherfather else 0) * 25
 percentage_fatherfather10 = (1 if checkbox10_fatherfather else 0) * 25
 percentage_fatherfather11 = (1 if checkbox11_fatherfather else 0) * 25
if checkbox1_fatherfather:
    selected_fatherfather.append(fatherfather[0])
if checkbox2_fatherfather:
    selected_fatherfather.append(fatherfather[1])
if checkbox3_fatherfather:
    selected_fatherfather.append(fatherfather[2])
if checkbox4_fatherfather:
    selected_fatherfather.append(fatherfather[3])
if checkbox5_fatherfather:
    selected_fatherfather.append(fatherfather[4])
if checkbox6_fatherfather:
    selected_fatherfather.append(fatherfather[5])
if checkbox7_fatherfather:
    selected_fatherfather.append(fatherfather[6])
if checkbox8_fatherfather:
    selected_fatherfather.append(fatherfather[7])
if checkbox9_fatherfather:
    selected_fatherfather.append(fatherfather[8])
if checkbox10_fatherfather:
    selected_fatherfather.append(fatherfather[9])
if checkbox11_fatherfather:
    selected_fatherfather.append(fatherfather[10])
st.text(" my father mother")
fathermother=["AlzheimerDisease","ArthritisDisease","Cancerdisease"
,"Dementiadisease","Diabetesdisease","HeartDisease",
"HighBloodPressure ","MultipleSclerosis disease","ParkinsonDisease","SpinaBifidadisease","ThyroidDisordersdisease "]
selected_fathermother = []
with st.expander("Show Checkboxes"):
 checkbox1_fathermother = st.checkbox(fathermother[0].split()[0] ,key=11132)
 checkbox2_fathermother = st.checkbox(fathermother[1].split()[0] ,key=11133)
 checkbox3_fathermother = st.checkbox(fathermother[2].split()[0] ,key=10100)
 checkbox4_fathermother = st.checkbox(fathermother[3].split()[0], key=11124)
 checkbox5_fathermother = st.checkbox(fathermother[4].split()[0] ,key=11777)
 checkbox6_fathermother = st.checkbox(fathermother[5].split()[0] ,key=11888)
 checkbox7_fathermother = st.checkbox(fathermother[6].split()[0] ,key=11999)
 checkbox8_fathermother = st.checkbox(fathermother[7].split()[0],key=113658)
 checkbox9_fathermother = st.checkbox(fathermother[8].split()[0] ,key=111475)
 checkbox10_fathermother = st.checkbox(fathermother[9].split()[0] ,key=113697)
 checkbox11_fathermother = st.checkbox(fathermother[10].split()[0] ,key=112568)
percentage_fathermother1 = (1 if checkbox1_fathermother else 0) * 25
percentage_fathermother2 = (1 if checkbox2_fathermother else 0) * 25
percentage_fathermother3 = (1 if checkbox3_fathermother else 0) * 25
percentage_fathermother4 = (1 if checkbox4_fathermother else 0) * 25
percentage_fathermother5 = (1 if checkbox5_fathermother else 0) * 25
percentage_fathermother6 = (1 if checkbox6_fathermother else 0) * 25
percentage_fathermother7 = (1 if checkbox7_fathermother else 0) * 25
percentage_fathermother8 = (1 if checkbox8_fathermother else 0) * 25
percentage_fathermother9 = (1 if checkbox9_fathermother else 0) * 25
percentage_fathermother10 = (1 if checkbox10_fathermother else 0) * 25
percentage_fathermother11 = (1 if checkbox11_fathermother else 0) * 25
if checkbox1_fathermother:
    selected_fathermother.append(fathermother[0])
if checkbox2_fathermother:
    selected_fathermother.append(fathermother[1])
if checkbox3_fathermother:
    selected_fathermother.append(fathermother[2])
if checkbox4_fathermother:
    selected_fathermother.append(fathermother[3])
if checkbox5_fathermother:
    selected_fathermother.append(fathermother[4])
if checkbox6_fathermother:
    selected_fathermother.append(fathermother[5])
if checkbox7_fathermother:
    selected_fathermother.append(fathermother[6])
if checkbox8_fathermother:
    selected_fathermother.append(fathermother[7])
if checkbox9_fathermother:
    selected_fathermother.append(fathermother[8])
if checkbox10_fathermother:
    selected_fathermother.append(fathermother[9])
if checkbox11_fathermother:
    selected_fathermother.append(fathermother[10])

st.text("uncle")
uncle=["AlzheimerDisease","ArthritisDisease","Cancerdisease"
,"Dementiadisease","Diabetesdisease","HeartDisease",
"HighBloodPressure ","MultipleSclerosis disease","ParkinsonDisease","SpinaBifidadisease","ThyroidDisordersdisease "]
selected_uncle = []
with st.expander("Show Checkboxes"):

 checkbox1_uncle = st.checkbox(uncle[0].split()[0] ,key=111132)
 checkbox2_uncle = st.checkbox(uncle[1].split()[0] ,key=111133)
 checkbox3_uncle = st.checkbox(uncle[2].split()[0] ,key=101100)
 checkbox4_uncle = st.checkbox(uncle[3].split()[0], key=111124)
 checkbox5_uncle = st.checkbox(uncle[4].split()[0] ,key=111777)
 checkbox6_uncle = st.checkbox(uncle[5].split()[0] ,key=111888)
 checkbox7_uncle = st.checkbox(uncle[6].split()[0] ,key=111999)
 checkbox8_uncle = st.checkbox(uncle[7].split()[0],key=1113658)
 checkbox9_uncle = st.checkbox(uncle[8].split()[0] ,key=1111475)
 checkbox10_uncle = st.checkbox(uncle[9].split()[0] ,key=1113697)
 checkbox11_uncle = st.checkbox(uncle[10].split()[0] ,key=1112568)
percentage_uncle1 = (1 if checkbox1_uncle else 0) * 25
percentage_uncle2 = (1 if checkbox2_uncle else 0) * 25
percentage_uncle3 = (1 if checkbox3_uncle else 0) * 25
percentage_uncle4 = (1 if checkbox4_uncle else 0) * 25
percentage_uncle5 = (1 if checkbox5_uncle else 0) * 25
percentage_uncle6 = (1 if checkbox6_uncle else 0) * 25
percentage_uncle7 = (1 if checkbox7_uncle else 0) * 25
percentage_uncle8 = (1 if checkbox8_uncle else 0) * 25
percentage_uncle9 = (1 if checkbox9_uncle else 0) * 25
percentage_uncle10 = (1 if checkbox10_uncle else 0) * 25
percentage_uncle11 = (1 if checkbox11_uncle else 0) * 25
if checkbox1_uncle:
    selected_uncle.append(uncle[0])
if checkbox2_uncle:
    selected_uncle.append(uncle[1])
if checkbox3_uncle:
    selected_uncle.append(uncle[2])
if checkbox4_uncle:
    selected_uncle.append(uncle[3])
if checkbox5_uncle:
    selected_uncle.append(uncle[4])
if checkbox6_uncle:
    selected_uncle.append(uncle[5])
if checkbox7_uncle:
    selected_uncle.append(uncle[6])
if checkbox8_uncle:
    selected_uncle.append(uncle[7])
if checkbox9_uncle:
    selected_uncle.append(uncle[8])
if checkbox10_uncle:
    selected_uncle.append(uncle[9])
if checkbox11_uncle:
    selected_uncle.append(uncle[10])

st.text("aunt")
aunt=["AlzheimerDisease","ArthritisDisease","Cancerdisease"
,"Dementiadisease","Diabetesdisease","HeartDisease",
"HighBloodPressure ","MultipleSclerosis disease","ParkinsonDisease","SpinaBifidadisease","ThyroidDisordersdisease "]
selected_aunt = []
with st.expander("Show Checkboxes"):

 checkbox1_aunt = st.checkbox(aunt[0].split()[0] ,key=1111132)
 checkbox2_aunt = st.checkbox(aunt[1].split()[0] ,key=1111133)
 checkbox3_aunt = st.checkbox(aunt[2].split()[0] ,key=1011100)
 checkbox4_aunt = st.checkbox(aunt[3].split()[0], key=1111124)
 checkbox5_aunt = st.checkbox(aunt[4].split()[0] ,key=1111777)
 checkbox6_aunt = st.checkbox(aunt[5].split()[0] ,key=1111888)
 checkbox7_aunt = st.checkbox(aunt[6].split()[0] ,key=1111999)
 checkbox8_aunt = st.checkbox(aunt[7].split()[0],key=11131658)
 checkbox9_aunt = st.checkbox(aunt[8].split()[0] ,key=11111475)
 checkbox10_aunt = st.checkbox(aunt[9].split()[0] ,key=11113697)
 checkbox11_aunt = st.checkbox(aunt[10].split()[0] ,key=11112568)
percentage_aunt1 = (1 if checkbox1_aunt else 0) * 25
percentage_aunt2 = (1 if checkbox2_aunt else 0) * 25
percentage_aunt3 = (1 if checkbox3_aunt else 0) * 25
percentage_aunt4 = (1 if checkbox4_aunt else 0) * 25
percentage_aunt5 = (1 if checkbox5_aunt else 0) * 25
percentage_aunt6 = (1 if checkbox6_aunt else 0) * 25
percentage_aunt7 = (1 if checkbox7_aunt else 0) * 25
percentage_aunt8 = (1 if checkbox8_aunt else 0) * 25
percentage_aunt9 = (1 if checkbox9_aunt else 0) * 25
percentage_aunt10 = (1 if checkbox10_aunt else 0) * 25
percentage_aunt11 = (1 if checkbox11_aunt else 0) * 25
if checkbox1_aunt:
    selected_aunt.append(aunt[0])
if checkbox2_aunt:
    selected_aunt.append(aunt[1])
if checkbox3_aunt:
    selected_aunt.append(aunt[2])
if checkbox4_aunt:
    selected_aunt.append(aunt[3])
if checkbox5_aunt:
    selected_aunt.append(aunt[4])
if checkbox6_aunt:
    selected_aunt.append(aunt[5])
if checkbox7_aunt:
    selected_aunt.append(aunt[6])
if checkbox8_aunt:
    selected_aunt.append(aunt[7])
if checkbox9_aunt:
    selected_aunt.append(aunt[8])
if checkbox10_aunt:
    selected_aunt.append(aunt[9])
if checkbox11_aunt:
    selected_aunt.append(aunt[10])


st.text("mymaternaluncle")
mymaternaluncle=["AlzheimerDisease","ArthritisDisease","Cancerdisease"
,"Dementiadisease","Diabetesdisease","HeartDisease",
"HighBloodPressure ","MultipleSclerosis disease","ParkinsonDisease","SpinaBifidadisease","ThyroidDisordersdisease "]
selected_mymaternaluncle = []
with st.expander("Show Checkboxes"):

 checkbox1_mymaternaluncle = st.checkbox(mymaternaluncle[0].split()[0] ,key=11111132)
 checkbox2_mymaternaluncle = st.checkbox(mymaternaluncle[1].split()[0] ,key=11111133)
 checkbox3_mymaternaluncle = st.checkbox(mymaternaluncle[2].split()[0] ,key=10111100)
 checkbox4_mymaternaluncle = st.checkbox(mymaternaluncle[3].split()[0], key=11111124)
 checkbox5_mymaternaluncle = st.checkbox(mymaternaluncle[4].split()[0] ,key=11111777)
 checkbox6_mymaternaluncle = st.checkbox(mymaternaluncle[5].split()[0] ,key=11111888)
 checkbox7_mymaternaluncle = st.checkbox(mymaternaluncle[6].split()[0] ,key=11111999)
 checkbox8_mymaternaluncle = st.checkbox(mymaternaluncle[7].split()[0],key=111311658)
 checkbox9_mymaternaluncle = st.checkbox(mymaternaluncle[8].split()[0] ,key=111111475)
 checkbox10_mymaternaluncle = st.checkbox(mymaternaluncle[9].split()[0] ,key=111113697)
 checkbox11_mymaternaluncle = st.checkbox(mymaternaluncle[10].split()[0] ,key=111112568)
percentage_mymaternaluncle1 = (1 if checkbox1_mymaternaluncle else 0) * 25
percentage_mymaternaluncle2 = (1 if checkbox2_mymaternaluncle else 0) * 25
percentage_mymaternaluncle3 = (1 if checkbox3_mymaternaluncle else 0) * 25
percentage_mymaternaluncle4 = (1 if checkbox4_mymaternaluncle else 0) * 25
percentage_mymaternaluncle5 = (1 if checkbox5_mymaternaluncle else 0) * 25
percentage_mymaternaluncle6 = (1 if checkbox6_mymaternaluncle else 0) * 25
percentage_mymaternaluncle7 = (1 if checkbox7_mymaternaluncle else 0) * 25
percentage_mymaternaluncle8 = (1 if checkbox8_mymaternaluncle else 0) * 25
percentage_mymaternaluncle9 = (1 if checkbox9_mymaternaluncle else 0) * 25
percentage_mymaternaluncle10 = (1 if checkbox10_mymaternaluncle else 0) * 25
percentage_mymaternaluncle11 = (1 if checkbox11_mymaternaluncle else 0) * 25
if checkbox1_mymaternaluncle:
    selected_mymaternaluncle.append(mymaternaluncle[0])
if checkbox2_mymaternaluncle:
    selected_mymaternaluncle.append(mymaternaluncle[1])
if checkbox3_mymaternaluncle:
    selected_mymaternaluncle.append(mymaternaluncle[2])
if checkbox4_mymaternaluncle:
    selected_mymaternaluncle.append(mymaternaluncle[3])
if checkbox5_mymaternaluncle:
    selected_mymaternaluncle.append(mymaternaluncle[4])
if checkbox6_mymaternaluncle:
    selected_mymaternaluncle.append(mymaternaluncle[5])
if checkbox7_mymaternaluncle:
    selected_mymaternaluncle.append(mymaternaluncle[6])
if checkbox8_mymaternaluncle:
    selected_mymaternaluncle.append(mymaternaluncle[7])
if checkbox9_mymaternaluncle:
    selected_mymaternaluncle.append(mymaternaluncle[8])
if checkbox10_mymaternaluncle:
    selected_mymaternaluncle.append(mymaternaluncle[9])
if checkbox11_mymaternaluncle:
    selected_mymaternaluncle.append(mymaternaluncle[10])

st.text("Mymaternalaunt")
Mymaternalaunt=["AlzheimerDisease","ArthritisDisease","Cancerdisease"
,"Dementiadisease","Diabetesdisease","HeartDisease",
"HighBloodPressure ","MultipleSclerosis disease","ParkinsonDisease","SpinaBifidadisease","ThyroidDisordersdisease "]
selected_Mymaternalaunt = []
with st.expander("Show Checkboxes"): 
 checkbox1_Mymaternalaunt = st.checkbox(Mymaternalaunt[0].split()[0] ,key=111111132)
 checkbox2_Mymaternalaunt = st.checkbox(Mymaternalaunt[1].split()[0] ,key=111111133)
 checkbox3_Mymaternalaunt = st.checkbox(Mymaternalaunt[2].split()[0] ,key=101111100)
 checkbox4_Mymaternalaunt = st.checkbox(Mymaternalaunt[3].split()[0], key=111111124)
 checkbox5_Mymaternalaunt = st.checkbox(Mymaternalaunt[4].split()[0] ,key=111111777)
 checkbox6_Mymaternalaunt = st.checkbox(Mymaternalaunt[5].split()[0] ,key=111111888)
 checkbox7_Mymaternalaunt = st.checkbox(Mymaternalaunt[6].split()[0] ,key=111111999)
 checkbox8_Mymaternalaunt = st.checkbox(Mymaternalaunt[7].split()[0],key=1113111658)
 checkbox9_Mymaternalaunt = st.checkbox(Mymaternalaunt[8].split()[0] ,key=1111111475)
 checkbox10_Mymaternalaunt = st.checkbox(Mymaternalaunt[9].split()[0] ,key=1111113697)
 checkbox11_Mymaternalaunt = st.checkbox(Mymaternalaunt[10].split()[0] ,key=1111112568)
percentage_Mymaternalaunt1 = (1 if checkbox1_Mymaternalaunt else 0) * 25
percentage_Mymaternalaunt2 = (1 if checkbox2_Mymaternalaunt else 0) * 25
percentage_Mymaternalaunt3 = (1 if checkbox3_Mymaternalaunt else 0) * 25
percentage_Mymaternalaunt4 = (1 if checkbox4_Mymaternalaunt else 0) * 25
percentage_Mymaternalaunt5 = (1 if checkbox5_Mymaternalaunt else 0) * 25
percentage_Mymaternalaunt6 = (1 if checkbox6_Mymaternalaunt else 0) * 25
percentage_Mymaternalaunt7 = (1 if checkbox7_Mymaternalaunt else 0) * 25
percentage_Mymaternalaunt8 = (1 if checkbox8_Mymaternalaunt else 0) * 25
percentage_Mymaternalaunt9 = (1 if checkbox9_Mymaternalaunt else 0) * 25
percentage_Mymaternalaunt10 = (1 if checkbox10_Mymaternalaunt else 0) * 25
percentage_Mymaternalaunt11 = (1 if checkbox11_Mymaternalaunt else 0) * 25
if checkbox1_Mymaternalaunt:
    selected_Mymaternalaunt.append(Mymaternalaunt[0])
if checkbox2_Mymaternalaunt:
    selected_Mymaternalaunt.append(Mymaternalaunt[1])
if checkbox3_Mymaternalaunt:
    selected_Mymaternalaunt.append(Mymaternalaunt[2])
if checkbox4_Mymaternalaunt:
    selected_Mymaternalaunt.append(Mymaternalaunt[3])
if checkbox5_Mymaternalaunt:
    selected_Mymaternalaunt.append(Mymaternalaunt[4])
if checkbox6_Mymaternalaunt:
    selected_Mymaternalaunt.append(Mymaternalaunt[5])
if checkbox7_Mymaternalaunt:
    selected_Mymaternalaunt.append(Mymaternalaunt[6])
if checkbox8_Mymaternalaunt:
    selected_Mymaternalaunt.append(Mymaternalaunt[7])
if checkbox9_Mymaternalaunt:
    selected_Mymaternalaunt.append(Mymaternalaunt[8])
if checkbox10_Mymaternalaunt:
    selected_Mymaternalaunt.append(Mymaternalaunt[9])
if checkbox11_Mymaternalaunt:
    selected_Mymaternalaunt.append(Mymaternalaunt[10])
option1=["memory loss","problem which language","misplacing things","change in personality"
         ,"Problems with abstract thinking"]
option2=["joint pain, tenderness and stiffness","inflammation in and around the joints","restricted movement of the joints","warm red skin over the affected joint"
         , "weakness and muscle wasting"]
option3=["Bladder changes","Bowel changes","Eating problems","Cough or hoarseness that does not go away"
         , "Bleeding or bruising, for no known reason"]  
option4=["Repeating questions","Wandering and getting lost in a familiar neighborhood","Using unusual words to refer to familiar objects"
         ,"Taking longer to complete normal daily tasks"
         , "Acting impulsively"]
option5=["Urinate (pee) a lot, often at night","Are very thirsty"
         ,"Lose weight"
         ,"Are very hungry"
         , "Have blurry vision"]
option6=["Chest pain","Shortness of breath"
         ,"Swelling in your legs"
         ,"Fatigue"]
option7=["Blurred vision","Nosebleeds","Shortness of breath"
         ,"Chest pain"
         ,"Dizziness"]
option8=["vision problems","numbness and tingling","muscle spasms, stiffness and weakness"
         ,"mobility problems"
         ,"pain"]
option9=["Tremor in hands, arms, legs, jaw, or head","Muscle stiffness","Slowed movement (bradykinesia)"
         ,"Speech changes"
         ,"Writing changes"]
option10=["weakness or total paralysis of the legs","bowel incontinence and urinary incontinence"
          ,"loss of skin sensation in the legs and around the bottom – the child is unable to feel hot or cold, which can lead to accidental injury"
         ]
option11=["twitching or trembling","warm skin and excessive sweating"
          ,"red palms of your hands","loose nails","patchy hair loss or thinning"
         ]
st.markdown("___Personal infromation:___")
st.text("Alzheimer's disease symptoms")
with st.expander("Show Options"):
 for i, option in enumerate(option1):
    # Create a unique key for each checkbox
    checkbox_key = f"checkbox_{i}"
    if st.checkbox(option, key=checkbox_key):
        selected_options_option1.add(option)
st.text("Arthritis's disease symptoms")
with st.expander("Show Options"):

 for option in option2:
    # If the checkbox is checked, add the option to the session state dictionary
    checkbox_key = f"checkbox_{option}"
    if st.checkbox(option, key=checkbox_key):
        selected_options_option2.add(option)
st.text("Cancer disease symptoms")      
# for option in option3:
with st.expander("Show Options"):

 for option in option3:
    # If the checkbox is checked, add the option to the session state dictionary
    checkbox_key = f"checkbox_{option}"
    if st.checkbox(option, key=checkbox_key):
        selected_options_option3.add(option)
st.text("Dementia disease symptoms") 
with st.expander("Show Options"):

 for option in option4:
    # If the checkbox is checked, add the option to the session state dictionary
    checkbox_key = f"checkbox_{option}"
    if st.checkbox(option, key=checkbox_key):
        selected_options_option4.add(option)
st.text("Diabetes disease symptoms") 
with st.expander("Show Options"):

 for option in option5:
    # If the checkbox is checked, add the option to the session state dictionary
    checkbox_key = f"checkbox_{option}"
    if st.checkbox(option, key=checkbox_key):
        selected_options_option5.add(option)
st.text("Heart disease symptoms") 
with st.expander("Show Options"):

 for option in option6:
    # If the checkbox is checked, add the option to the session state dictionary
    checkbox_key = f"checkbox_{option}"
    if st.checkbox(option, key=checkbox_key):
        selected_options_option6.add(option)
st.text("High blood pressure symptoms")
with st.expander("Show Options"):

 for option in option7:
    # If the checkbox is checked, add the option to the session state dictionary
    checkbox_key = f"checkboxs_{option}"
    if st.checkbox(option, key=checkbox_key):
        selected_options_option7.add(option)
st.text("Multiple sclerosis symptoms")
with st.expander("Show Options"):

 for option in option8:
    # If the checkbox is checked, add the option to the session state dictionary
    checkbox_key = f"checkboxss_{option}"
    if st.checkbox(option, key=checkbox_key):
        selected_options_option8.add(option)
st.text("Parkinson's disease symptoms")
with st.expander("Show Options"):

 for option in option9:
    # If the checkbox is checked, add the option to the session state dictionary
    checkbox_key = f"checkboxsss_{option}"
    if st.checkbox(option, key=checkbox_key):
        selected_options_option9.add(option)
st.text("Spina bifida symptoms")
with st.expander("Show Options"):

 for option in option10:
    # If the checkbox is checked, add the option to the session state dictionary
    checkbox_key = f"checkboxssss_{option}"
    if st.checkbox(option, key=checkbox_key):
        selected_options_option10.add(option)    
st.text("Thyroid disorders symptoms")
with st.expander("Show Options"):

 for option in option11:
    # If the checkbox is checked, add the option to the session state dictionary
    checkbox_key = f"checkboxsssss_{option}"
    if st.checkbox(option, key=checkbox_key):
        selected_options_option11.add(option)


percentage1 = (len(selected_options_option1) / len(option1)) * 50+percentage_checkbox1_fahter+percentage_checkbox1_mother+percentage_motherfather1+percentage_mothermother1+percentage_fatherfather1+percentage_fathermother1+percentage_uncle1+percentage_aunt1+percentage_mymaternaluncle1+percentage_Mymaternalaunt1
percentage2 = (len(selected_options_option2) / len(option2)) * 50+percentage_checkbox2+percentage_checkbox2_mother+percentage_motherfather2+percentage_mothermother2+percentage_fatherfather2+percentage_fathermother2+percentage_uncle2+percentage_aunt2+percentage_mymaternaluncle2+percentage_Mymaternalaunt2
percentage3 = (len(selected_options_option3) / len(option3)) * 50+percentage_checkbox3+percentage_checkbox3_mother+percentage_motherfather3+percentage_mothermother3+percentage_fatherfather3+percentage_fathermother3+percentage_uncle3+percentage_aunt3+percentage_mymaternaluncle3+percentage_Mymaternalaunt3
percentage4 = (len(selected_options_option4) / len(option4)) * 50+percentage_checkbox4+percentage_checkbox4_mother+percentage_motherfather4+percentage_mothermother4+percentage_fatherfather4+percentage_fathermother4+percentage_uncle4+percentage_aunt4+percentage_mymaternaluncle4+percentage_Mymaternalaunt4
percentage5 = (len(selected_options_option5) / len(option5)) * 50+percentage_checkbox5+percentage_checkbox5_mother+percentage_motherfather5+percentage_mothermother5+percentage_fatherfather5+percentage_fathermother5+percentage_uncle5+percentage_aunt5+percentage_mymaternaluncle5+percentage_Mymaternalaunt5
percentage6 = (len(selected_options_option6) / len(option6)) * 50+percentage_checkbox6+percentage_checkbox6_mother+percentage_motherfather6+percentage_mothermother6+percentage_fatherfather6+percentage_fathermother6+percentage_uncle6+percentage_aunt6+percentage_mymaternaluncle6+percentage_Mymaternalaunt6
percentage7 = (len(selected_options_option7) / len(option7)) * 50+percentage_checkbox7+percentage_checkbox7_mother+percentage_motherfather7+percentage_mothermother7+percentage_fatherfather7+percentage_fathermother7+percentage_uncle7+percentage_aunt7+percentage_mymaternaluncle7+percentage_Mymaternalaunt7
percentage8 = (len(selected_options_option8) / len(option8)) * 50+percentage_checkbox8+percentage_checkbox8_mother+percentage_motherfather8+percentage_mothermother8+percentage_fatherfather8+percentage_fathermother8+percentage_uncle8+percentage_aunt8+percentage_mymaternaluncle8+percentage_Mymaternalaunt8
percentage9 = (len(selected_options_option9) / len(option9)) * 50+percentage_checkbox9+percentage_checkbox9_mother+percentage_motherfather9+percentage_mothermother9+percentage_fatherfather9+percentage_fathermother9+percentage_uncle9+percentage_aunt9+percentage_mymaternaluncle9+percentage_Mymaternalaunt9
percentage10 = (len(selected_options_option10) / len(option10)) * 50+percentage_checkbox10+percentage_checkbox10_mother+percentage_motherfather10+percentage_mothermother10+percentage_fatherfather10+percentage_fathermother10+percentage_uncle10+percentage_aunt10+percentage_mymaternaluncle10+percentage_Mymaternalaunt10
percentage11 = (len(selected_options_option11) / len(option11)) * 50+percentage_checkbox11+percentage_checkbox11_mother+percentage_motherfather11+percentage_mothermother11+percentage_fatherfather11+percentage_fathermother11+percentage_uncle11+percentage_aunt11+percentage_mymaternaluncle11+percentage_Mymaternalaunt11


sum = 7


insureence = pd.read_csv("assests\insurence co.csv")
insureence['Discount']=insureence['Discount'].str.rstrip("%").astype(float)
docs = pd.read_csv("assests\doc.csv")
labs = pd.read_excel("assests\labs.xlsx",header=None)
phy = pd.read_excel("assests\phy.xlsx",header=None)

if st.button('Make Prediction'):
    inputs = [[percentage1/100,percentage2/100,percentage3/100,percentage4/100,percentage5/100,percentage6/100,percentage7/100,percentage8/100,percentage9/100,percentage10/100,percentage11/100]]
    # st.write(inputs)
    prediction = model.predict(inputs)
    # st.write(prediction)
    st.write(f"The percentage of  Alzheimer's disease is: {100*prediction[0][0]}%")
    st.write(f"The percentage of  Arthritis's disease is: {100*prediction[0][1]}%")
    st.write(f"The percentage of Cancer disease is: {100*prediction[0][2]}%")
    st.write(f"The percentage of Dementia disease: {100*prediction[0][3]}%")
    st.write(f"The percentage of Diabetes disease: {100*prediction[0][4]}%")
    st.write(f"The percentage of Heart disease: {100*prediction[0][5]}%")
    st.write(f"The percentage of High blood pressure disease: {100*prediction[0][6]}%")
    st.write(f"The percentage of Multiple sclerosis disease: {100*prediction[0][7]}%")
    st.write(f"The percentage of Parkinson's disease: {100*prediction[0][8]}%")
    st.write(f"The percentage of Spina bifida disease: {100*prediction[0][9]}%")
    st.write(f"The percentage of Thyroid disorders disease: {100*prediction[0][10]}%")

    for i in prediction[0]:
        sum = sum + i

    st.markdown("___insurance companies that fit:___")
    insureence.loc[(insureence['Discount'] > sum-1.5) & (insureence['Discount'] < sum+1.5)]

    higheschance = max(prediction[0])
    st.markdown("___list of Doctors that will help:___")
    st.markdown("**based on you highes Percentage:**")

    if higheschance == prediction[0][0]:
        docs.loc[docs["Specialization"]=="Specializes in Alzheimer's disease"]
        st.markdown("**recommended lab:**")
        lab = labs.loc[labs[0]=="Alzheimer's disease "]
        lab[1].values[0]

    if higheschance == prediction[0][1]:
        docs.loc[docs["Specialization"]==" Specializes in Arthritis"]
        st.markdown("**recommended lab:**")
        lab = labs.loc[labs[0]=="Arthritis "]
        lab[1].values[0]

    if higheschance == prediction[0][2]:
        docs.loc[docs["Specialization"]=="Specializes in Cancer"]
        st.markdown("**recommended lab:**")
        lab = labs.loc[labs[0]=="Cancer "]
        lab[1].values[0]

    if higheschance == prediction[0][3]:
        docs.loc[docs["Specialization"]=="Specializes in Dementia"]
        st.markdown("**recommended lab:**")
        lab = labs.loc[labs[0]=="Dementia "]
        lab[1].values[0]
        
    if higheschance == prediction[0][4]:
        docs.loc[docs["Specialization"]=="Specializes in Diabetes"]
        st.markdown("**recommended lab:**")
        lab = labs.loc[labs[0]=="Diabetes "]
        lab[1].values[0]

    if higheschance == prediction[0][5]:
        docs.loc[docs["Specialization"]=="Specializes in Heart disease"]
        st.markdown("**recommended lab:**")
        lab = labs.loc[labs[0]=="Heart disease "]
        lab[1].values[0]

    if higheschance == prediction[0][6]:
        docs.loc[docs["Specialization"]=="Specializes in High blood pressure"]
        st.markdown("**recommended lab:**")
        lab = labs.loc[labs[0]=="High blood pressure "]
        lab[1].values[0]

    if higheschance == prediction[0][7]:
        docs.loc[docs["Specialization"]=="Specializes in Multiple sclerosis"]
        st.markdown("**recommended lab:**")
        lab = labs.loc[labs[0]=="Multiple sclerosis "]
        lab[1].values[0]

    if higheschance == prediction[0][8]:
        docs.loc[docs["Specialization"]=="Specializes in Parkinson's disease"]
        st.markdown("**recommended lab:**")
        lab = labs.loc[labs[0]=="Parkinson's disease "]
        lab[1].values[0]

    if higheschance == prediction[0][9]:
        docs.loc[docs["Specialization"]=="Specializes in Spina bifida"]
        st.markdown("**recommended lab:**")
        lab = labs.loc[labs[0]=="Spina bifida "]
        lab[1].values[0]

    if higheschance == prediction[0][10]:
        docs.loc[docs["Specialization"]=="Specializes in Thyroid disorders"]
        st.markdown("**recommended lab:**")
        lab = labs.loc[labs[0]=="Thyroid disorders "]
        lab[1].values[0]

    st.markdown("**Pharmacis that we recommend**")
    Pharmacis = phy.sample(n=5)
    Pharmacis


img=Image.open("imageEdit.jpg")