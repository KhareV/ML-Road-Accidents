import streamlit as st
from PIL import Image

st.set_page_config(page_title="Helpline", page_icon="📞", layout="wide")
logo = Image.open("Images/logo.jpg")
st.image(logo, width=100)
gif_url = "https://ewscripps.brightspotcdn.com/dims4/default/7c4e68f/2147483647/strip/true/crop/800x450+0+0/resize/320x180!/quality/90/?url=http%3A%2F%2Fewscripps-brightspot.s3.amazonaws.com%2F3d%2F82%2Fcd5de20740f5b50f843fab609997%2Fcaraccidentgif-storyblocks.gif"
st.sidebar.image(gif_url, use_column_width=True)

st.write("<h2>General Helpline Numbers:</h2>", unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)
with col1:
    st.image("police.png", width=80)
    st.write("Police:[100](tel:100)")
with col2:
    st.image("fire.png", width=80)
    st.write("Fire: [101](tel:101)")
with col3:
    st.image("ambulance.png", width=80)
    st.write("Ambulance: [102](tel:102)")

col4, col5, col6 = st.columns(3)
with col4:
    st.image("women.png", width=80)
    st.write("Women Helpline:[1091](tel:1091)")
with col5:
    st.image("child.png", width=80)
    st.write("Child Helpline: [1098](tel:1098)")
with col6:
    st.image("emergency.png", width=80)
    st.write("National Emergency Number:[112](tel:112)")


st.header("Contact Us")
st.write("Please fill out the form below to contact us.")
name = st.text_input("Name")
email = st.text_input("Email")
message = st.text_area("Message")
if st.button("Submit"):
    st.success("Thank you for contacting us. We will get back to you soon.")

st.sidebar.write("More info : https://morth.nic.in/")
st.markdown("<br><hr>", unsafe_allow_html=True)
st.markdown("<h20 style='text-align: centre; color:white; font-size: 10px;font-family: Comic Sans MS;'>This app was created as a part of the course project for Data Analysis and Visualization (DAVL).</h20>", unsafe_allow_html=True)
st.markdown("<h25 style='text-align: centre; color:white;font-size: 8px; font-family: Comic Sans MS;'>Made with ❤️</h25>", unsafe_allow_html=True)


