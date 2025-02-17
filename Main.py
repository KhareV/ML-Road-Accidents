import streamlit as st
import time
import base64
from datetime import datetime

st.set_page_config(page_title="Accident Analysis", page_icon="🤕", layout="wide")

def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

img = get_img_as_base64("Images/p1.jpg")
page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("https://cdn.pixabay.com/photo/2013/11/28/10/36/road-220058__480.jpg");
background-size: 100%;
background-position: top left;
background-repeat: no-repeat;
background-attachment: local;
}}
[data-testid="stSidebar"] > div:first-child {{
background-image: url("data:image/png;base64,{img}");
background-position: center; 
background-repeat: no-repeat;
background-attachment: fixed;
}}
[data-testid="stHeader"] {{
background: rgba(0,0,0,0);
}}
[data-testid="stToolbar"] {{
right: 2rem;
}}
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)

def run_text(text, interval=0.1):
    st_placeholder = st.empty()
    for i in range(100):
        for i in range(len(text)):
            st_placeholder.write(f"<div style='text-align: center'><span style='color: #000000; font-weight: bold;font-size: 30px;font-family: Comic Sans MS;'>{text[:i+1]}</span>", unsafe_allow_html=True)            
            time.sleep(interval)
        time.sleep(1)

st.markdown(f"\n\n\n\n\n\n\n\n\n\n\n\n")
st.markdown(f"\n\n\n\n\n\n\n\n\n\n\n\n")
st.markdown(f"\n\n\n\n\n\n\n\n\n\n\n\n") 

st.markdown("""
<div style='text-align:center'>
    <a href="http://localhost:8501/Home_page" target ="_self" style="
        display: inline-block;
        padding: 0.5em 1em;
        background-color: #000000;
        border: none;
        border-radius: 3px;
        box-shadow: 0px 2px 3px rgba(0, 0, 0, 0.2);
        text-decoration: none;
        color: #FFFFFF;
        transition: all 0.2s ease-in-out;
    ">
        Home🏡
    </a>
</div>
""", unsafe_allow_html=True)
st.markdown(f"\n\n\n\n\n\n\n\n\n\n\n\n")

text = "Stop accidents before they stop you.💀"
run_text(text)



