#import std libraries
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
# Write a title
st.title('Data Explorer app')
# Write data taken from https://allisonhorst.github.io/palmerpenguins/
st.write('A **simple** *app* to explore `penguin` [data](https://allisonhorst.github.io/palmerpenguins/) :penguin:')
# Put image https://raw.githubusercontent.com/allisonhorst/palmerpenguins/master/man/figures/lter_penguins.png
st.image('https://raw.githubusercontent.com/allisonhorst/palmerpenguins/master/man/figures/lter_penguins.png')
# Write heading for Data
st.header('Data')
# Read csv file and output a sample of 20 data points
df = pd.read_csv('penguins_extra.csv')
st.write(df)
# Add a selectbox for species
species = st.selectbox('Select your species',df.species.unique())
# Display a sample of 20 data points according to the species selected with corresponding title
df_species = df.loc[df.species==species]
st.write(df_species.sample(20))
# Plotting seaborn
st.subheader('Simple seaborn plot')
fig, ax = plt.subplots()
ax = sns.scatterplot(data=df,x='bill_length_mm',y='flipper_length_mm',hue='species',size='sex')
st.pyplot(fig)
# Plotting plotly
st.subheader('Interactive plotly plot')
fig = px.scatter(df,x='bill_length_mm',y='flipper_length_mm',animation_frame='species',range_x=[25,80],range_y=[150,250],color='sex')
st.plotly_chart(fig)
# Bar chart count of species per island
st.subheader('Barchart counting the number of species per island')
st.bar_chart(df.groupby('species')['island'].count())
# Maps
st.subheader('Potting the data points in a map')
st.map(df)

file = st.file_uploader('Upload csv files',type=['csv'])
if file is not None:
    data = pd.read_csv(file)
    st.write(data)


file_img = st.sidebar.file_uploader('Input image',type=['png','jpg','jpeg'])
from PIL import Image
if file_img is not None:
    img = Image.open(file_img)
    st.image(img)


# Reference https://deckgl.readthedocs.io/en/latest/
st.write('If you are intersted to further explore mapping check out [pydeck](https://deckgl.readthedocs.io/en/latest/)')

# sidebar comment
choices = st.sidebar.radio('Hope you found this interseting ', ['yes', 'no'])
if choices =='yes':
    st.write('Yes')
else:
    st.write('no')
# Add background image in
st.markdown(
        f"""
        <style>
        .stApp {{
            background: url("https://images.unsplash.com/photo-1542281286-9e0a16bb7366");
            background-size: cover
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


