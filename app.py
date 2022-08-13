from fastai.vision.all import *
import streamlit as st
import pathlib
import plotly.express as px
import platform

plt = platform.system()
if plt=='Linux' : pathlib.WindowsPath = pathlib.PosixPath

# temp = pathlib.PosixPath
# pathlib.PosixPath = pathlib.WindowsPath

st.title('Image Predict')
st.write('This model is familiar with Musical_instrument, Fruit, Ball, Weapon  and Telephone')
file = st.file_uploader('Image upload', type=['png', 'jpg', 'gif', 'svg'])
if file:
    st.image(file)
    img = PILImage.create(file)
    model = load_learner('deep_learning.pkl')
    pred, id, prob = model.predict(img)
    st.success(f"Prediction:{pred}")
    st.info(f"Probilaty: {prob[id]*100:.1f}%")

    fig = px.bar(x=prob*100,y=model.dls.vocab)
    st.plotly_chart(fig)