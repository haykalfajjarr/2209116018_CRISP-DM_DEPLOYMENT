import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import altair as alt
import matplotlib.pyplot as plt
from function import *

df = pd.read_csv('insurance.csv')

with st.sidebar :
    selected = option_menu('Insurance Predict',['Data Distribution','Relation','Composition & Comparison','Predict','Clustering'],default_index=0)

if (selected == 'Data Distribution'):
    st.title("Data Distribution")
    histplot(df)
    
if (selected == 'Relation'):
    st.title('Relations')
    heatmap(df)

if (selected == 'Composition & Comparison'):
    st.title('Composition')
    compositionAndComparison(df)

if (selected == 'Predict'):
    st.title('Predicting Data')
    Predict()

if (selected == 'Clustering'):
    st.title('Clust The Data')
    clustering(df)
