import streamlit as st
from PIL import Image

st.set_page_config(page_title="About", page_icon="⁉️", layout="wide")
logo = Image.open('Images/logo.jpg')
st.image(logo, width=100)
gif_url = "https://ewscripps.brightspotcdn.com/dims4/default/7c4e68f/2147483647/strip/true/crop/800x450+0+0/resize/320x180!/quality/90/?url=http%3A%2F%2Fewscripps-brightspot.s3.amazonaws.com%2F3d%2F82%2Fcd5de20740f5b50f843fab609997%2Fcaraccidentgif-storyblocks.gif"
st.sidebar.image(gif_url, use_column_width=True)

st.markdown("<h1 style='text-align: center; color:white;font-size: 30px; font-family: Comic Sans MS;'>About us</h1>", unsafe_allow_html=True)
st.write('We are a group of three students from PSG College of Technology, who worked together for this project. Our work gave us an opportunity to practice learning in classrooms today by gaining valuable insights and developing innovative solutions to real life problems. This experience showed us that working together as small groups is essential to the success of any project, and that teamwork is one of the strongest core philosophies in the working world.')
st.markdown("<h1 style='text-align: center; color:white;font-size: 30px; font-family: Comic Sans MS;'>Our project</h1>", unsafe_allow_html=True)
st.write('Road safety is an essential aspect of modern society and affects everyone, whether as drivers, passengers, or pedestrians.In India it is a crucial issue that needs urgent attention. India has one of the highest rates of road accidents in the world, with an average of around 1,50,000 deaths and 4,50,000 injuries every year. The primary causes of road accidents in India are reckless driving, overspeeding, drunk driving, poor road infrastructure, and inadequate traffic management.')
st.write('We have conducted an analysis of road safety to better understand the various factors that contribute to accidents and injuries on the road. We have explored the statistics on road accidents, the causes and effects of accidents, and the measures that can be taken to improve road safety.In this analysis, we have also focused on the situation of road safety in different regions of our country where road accidents are a significant concern. ')
st.markdown("<h1 style='text-align: center; color:white;font-size: 30px; font-family: Comic Sans MS;'>Our Mission</h1>", unsafe_allow_html=True)
st.write('Our mission is to provide insights and recommendations that can help address the issue of road safety and promote safer roads for everyone. We hope that our analysis will contribute to the ongoing efforts to improve road safety around the world.')


image_paths = ['https://www.mapsofindia.com/ci-moi-images/my-india/2016/04/road-accidents-in-india.jpg','https://i.ytimg.com/vi/3pKwjbHj1Ms/maxresdefault.jpg','https://assets.telegraphindia.com/telegraph/2021/Nov/1638128124_police.jpg']
image_width=300
image='accident2.jpg'
#st.image(image, width=200,use_column_width=True)
st.markdown(
    """
    <style>
    .stImage > img {
        display: block;
        margin: 30px auto 0 auto;
    }
    </style>
    """,
    unsafe_allow_html=True
)

image_html = ''
for url in image_paths:
    image_html += f'<img src="{url}" style="width:2000px; height:500px; object-fit: cover; display: block;margin-right: 20px;">'

st.markdown(f"""<div style="display: flex; overflow-x: scroll; padding: 10px 0;">{image_html}</div>""", unsafe_allow_html=True )


st.markdown("<br><br><br><hr>", unsafe_allow_html=True)
 
st.markdown("<h25 style='text-align: right; color:white;font-size: 25px; font-family: Comic Sans MS;'>Modules used for Prediction</h25>", unsafe_allow_html=True)
st.write('-Pandas')
st.write('-Numpy')
st.write('-Statsmodels')
st.write('-Sklearn')
st.write('-Matplotlib')
st.write('-Random Forest Regressor')

st.markdown("<br><hr>", unsafe_allow_html=True)
st.markdown("<h25 style='text-align: right; color:white;font-size: 25px; font-family: Comic Sans MS;'>References</h25>", unsafe_allow_html=True)
st.write("https://morth.nic.in/")
st.write("https://www.ndtv.com/topic/road-accidents")
st.write("https://bro.gov.in/WriteReadData/linkimages/5768690382-14.pdf")

st.sidebar.write("https://morth.nic.in/")
st.markdown("<br><hr>", unsafe_allow_html=True)
st.markdown("<h20 style='text-align: right; color:white; font-size: 10px;font-family: Comic Sans MS;'>This app was created as a part of the course project for Data Analysis and Visualization (DAVL).</h20>", unsafe_allow_html=True)
st.markdown("<h25 style='text-align: right; color:white;font-size: 8px; font-family: Comic Sans MS;'>Made with ❤️</h25>", unsafe_allow_html=True)



