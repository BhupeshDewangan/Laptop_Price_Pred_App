import streamlit as st
import pickle 
import numpy as np
import math
import requests
import sklearn

from streamlit_lottie import st_lottie
from streamlit_lottie import st_lottie_spinner

pipe = pickle.load(open('pipe.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))

PAGE_TITLE = "💻 Laptop Price Predictor !!"

st.set_page_config(page_title=PAGE_TITLE)

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_url_hello = "https://assets8.lottiefiles.com/packages/lf20_xnbikipz.json"
lottie_hello = load_lottieurl(lottie_url_hello)

st_lottie(lottie_hello, key="hello")

st.title("Laptop Price Predictor")


company = st.selectbox('Brand', df['Company'].unique())

if company == "Apple":
    typeName = st.selectbox('Type', ['Macbook'])
else:
    typeName = st.selectbox('Type', ['Notebook', 'Ultrabook', 'Netbook', 'Gaming', '2 in 1 Convertible', 'Workstation'])

ram = st.selectbox('RAM(in GB)', [8, 12, 16, 32, 64])

weight = st.number_input('Weight of the Laptop')

touchscreen = st.selectbox('TouchScreen', ['No', 'Yes'])

ips = st.selectbox('IPS', ['No', 'Yes'])

screen_size = st.number_input('Screen Size')

resolution = st.selectbox('Screen Resolution',['1920x1080','3840x2160','3200x1800','2880x1800','2560x1600','2560x1440','2304x1440'])

cpu = st.selectbox('CPU', df['Cpu Brand'].unique())

hdd = st.selectbox('HDD(in GB)', [512, 1024, 2048] )

ssd = st.selectbox('SSD(in GB)',[128, 256, 512, 1024])

gpu = st.selectbox('GPU',df['Gpu Brand'].unique())


if company == "Apple":
    os = st.selectbox('OS', ['MacOS'])

else:
    os = st.selectbox('OS', ['Other/No OS/ Linux', 'Windows'])

if st.button('Predict Price'):
    # query
    ppi = None
    if touchscreen == 'Yes':
        touchscreen = 1
    else:
        touchscreen = 0

    if ips == 'Yes':
        ips = 1
    else:
        ips = 0

    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])

    ppi = ((X_res**2) + (Y_res**2))**0.5/screen_size

    query = np.array([company,typeName,ram,weight,touchscreen,ips,ppi,cpu,hdd,ssd,gpu,os])

    query = query.reshape(1,12)
    st.title("The predicted price of this configuration is " + str(int(np.exp(pipe.predict(query)[0]))))

