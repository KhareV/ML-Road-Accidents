import streamlit as st
from PIL import Image
import time


st.set_page_config(page_title="Accident Analysis", page_icon="🤕", layout="wide")
gif_url = "https://ewscripps.brightspotcdn.com/dims4/default/7c4e68f/2147483647/strip/true/crop/800x450+0+0/resize/320x180!/quality/90/?url=http%3A%2F%2Fewscripps-brightspot.s3.amazonaws.com%2F3d%2F82%2Fcd5de20740f5b50f843fab609997%2Fcaraccidentgif-storyblocks.gif"
st.sidebar.image(gif_url, use_column_width=True)
logo = Image.open("Images/logo.jpg")
st.image(logo, width=100)
st.sidebar.write("More info : https://morth.nic.in/")
st.markdown("<h1 style='text-align: center; color:white; font-family: Comic Sans MS;'>THE COST OF CARELESSNESS</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center;color:white;font-size: 24px;'>Analyzing Road Accidents</h2><hr>", unsafe_allow_html=True)
tab1, tab2= st.tabs(["Overview","Latest News"])

st.markdown(
    """
    <div class="icon-container">
      <a href="#" onclick="showAccount('instagram')"><img src="https://cdn4.iconfinder.com/data/icons/social-media-icons-the-circle-set/48/instagram_circle-512.png" width="30" height="30"></a>
      <a href="#" onclick="showAccount('whatsapp')"><img src="https://cdn4.iconfinder.com/data/icons/social-messaging-ui-color-shapes-2-free/128/social-whatsapp-circle-512.png" width="30" height="30"></a>
      <a href="#" onclick="showAccount('twitter')"><img src="https://cdn3.iconfinder.com/data/icons/social-icons-5/607/Twitterbird.png" width="30" height="30"></a>
      <a href="#" onclick="showAccount('mail')"><img src="https://cdn2.iconfinder.com/data/icons/social-icons-circular-color/512/gmail-512.png" width="30" height="30"></a>
    </div>
    """,
    unsafe_allow_html=True
)

with tab2:
       st.markdown("""
        <html>
        <head>
            <link rel="stylesheet" type="text/css" href="style.css">
        </head>
        <body>
        <div class="container">
        <h2>Latest News</h2>
        <ul class="cards">
            <li class="card">
            <img src="https://www.shutterstock.com/image-illustration/dead-cartoon-guy-under-word-260nw-161130011.jpg">
            <div><br>
                <h4 class="card-title">LIVE Breaking News !!</h4>
                <div class="card-content">
                <p></p>
                </div>
            </div>
            <div class="card-link-wrapper">
                <a href="https://www.ndtv.com/topic/accident" class="card-link">Learn More</a>
            </div>
            </li>
            <li class="card">
            <img src="https://static.toiimg.com/thumb/msid-98748773,imgsize-33070,width-400,resizemode-4/98748773.jpg" alt="news2" style="width:100%">
            <div><br>
                <h4 class="card-title">4 killed, 28 injured as bus overturns in Jammu and Kashmir's.</h4>
                <div class="card-content">
                <p>Srinagar: Four people were killed and 28 other suffered injuries when the bus in which they were travelling overturned as the driver lost control of the vehicle in Pulwama district of Jammu and Kashmir on Saturday.</p>
                </div>
            </div>
            <div class="card-link-wrapper">
                <a href="https://timesofindia.indiatimes.com/city/srinagar/4-killed-28-injured-as-bus-overturns-in-jammu-and-kashmirs-pulwama/articleshow/98748365.cms" class="card-link">Learn More</a>
            </div>
            </li>
            <li class="card">
            <img src="https://static.toiimg.com/thumb/msid-98643986,imgsize-76883,width-400,resizemode-4/98643986.jpg">
            <div><br>
                <h4 class="card-title">Child Killed As Truck Hits Scooter</h4>
                <div class="card-content">
                <p> Shahjahanpur: Three people, including a child,were killed here when a truck hit the scooter on which they were travelling and dragged the two-wheeler for almost 500 metres.</p>
                </div>
            </div>
            <div class="card-link-wrapper">
                <a href="https://www.ndtv.com/others-news/child-among-3-killed-as-truck-hits-scooter-drags-it-for-500-metres-in-up-3871799" class="card-link">Learn More</a>
            </div>
            </li>
            <li class="card">
            <img src="https://static.toiimg.com/thumb/msid-98745786,imgsize-784319,width-400,resizemode-4/98745786.jpg" alt="news4" style="width:100%">
            <div><br>
                <h4 class="card-title">Nagoa man dead as tourist's vehicle hits him</h4>
                <div class="card-content">
                <p>PANAJI: A speeding Gujarati tourist on Friday rammed into the bike of a Nagoa man dropping his children to school, killing him and injuring the children.</p>
                </div>
            </div>
            <div class="card-link-wrapper">
                <a href="https://timesofindia.indiatimes.com/city/goa/nagoa-man-dead-as-gujarati-tourists-vehicle-hits-him/articleshow/98745777.cms"class="card-link">Learn More</a>
            </div>
            </li>
            <li class="card">
            <img src="https://static.toiimg.com/thumb/msid-98670063,imgsize-173800,width-400,resizemode-4/98670063.jpg" alt="news5" style="width:100%">
            <div><br>
                <h4 class="card-title">4 dead, 4 injured in head-on collision between two cars</h4>
                <div class="card-content">
                <p>PATIALA: Four persons died and four others were injured following a head-on collision between two cars in Punjab's Sangrur district on Wednesday morning..</p>
                </div>
            </div>
            <div class="card-link-wrapper">
                <a href="https://timesofindia.indiatimes.com/city/amritsar/4-dead-4-injured-in-head-on-collision-between-two-cars-in-punjabs-sangrur/articleshow/98669094.cms" class="card-link">Learn More</a>
            </div>
            </li>
            <li class="card">
            <img src="https://static.toiimg.com/thumb/msid-98719197,imgsize-600105,width-400,resizemode-4/98719197.jpg" alt="news6" style="width:100%">
            <div><br>
                <h4 class="card-title">Man in car carrying liquor dies in crash</h4>
                <div class="card-content">
                <p>NEW DELHI: A man died after his car carrying illicit liquor bottles met with an accident in Chhawla on Monday. Police said another person was injured in the accident..</p>
                </div>
            </div>
            <div class="card-link-wrapper">
                <a href="https://timesofindia.indiatimes.com/city/delhi/man-in-car-carrying-liquor-dies-in-crash-in-delhi/articleshow/98719173.cms" class="card-link">Learn More</a>
            </div>
            </li>
            <li class="card">
            <img src="https://c.ndtvimg.com/2023-03/si6p27k4_sadvhi-jyoti-innova-crash-650_625x300_16_March_23.jpg" alt="news7" style="width:100%">
            <div><br>
                <h3 class="card-title">Union Minister Injured as car Crashes Into Truck</h3>
                <div class="card-content">
                <p>Vijayapara: Union Minister Sadhvi Niranjan Jyoti suffered injuries after the car she was travelling in met an accident today in Karnataka's Vijayapura..</p>
                </div>
            </div>
            <div class="card-link-wrapper">
                <a href="https://www.ndtv.com/india-news/union-minister-injured-as-innova-crashes-into-truck-on-karnataka-highway-3867789" class="card-link">Learn More</a>
            </div>
            </li>
        </ul>
        </div>
        </body>
        </html>
        <style>
        :root {
        --red: #ef233c;
        --darkred: #c00424;
        --platinum: #e5e5e5;
        --black: #2b2d42;
        --white: #fff;
        --thumb: #edf2f4;
        }
        * {
        box-sizing: border-box;
        padding: 0;
        margin: 0;
        }
        body {
        font: 16px / 24px "Rubik", sans-serif;
        color: var(--white);
        background: var(--black);
        margin: 50px 0;
        }
        .container {
        width: 1000px;
        height:1200px;
        padding: 0 15px;
        margin: 0 auto;
        }
        h2 {
        font-size: 32px;
        margin-bottom: 1em;
        }
        .cards {
            width: 1000px;
        height:900px;
        display: flex;
        padding: 25px 0px;
        list-style: none;
        overflow-x: scroll;
        scroll-snap-type: x mandatory;
        }
        .card {
            width: 900px;
        height:500px;
        display: flex;
        flex-direction: column;
        flex: 0 0 100%;
        padding: 20px;
        background: var(--black);
        border-radius: 12px;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 15%);
        scroll-snap-align: start;
        transition: all 0.2s;
        }
        .card:not(:last-child) {
        margin-right: 10px;
        }
        .card:hover {
        color: var(--white);
        background:#2fbac4;
        }
        .card .card-title {
        font-size: 20px;
        }
        .card .card-content {
        margin: 20px 0;
        max-width: 85%;
        }
        .card .card-link-wrapper {
        margin-top: auto;
        }
        .card .card-link {
        display: inline-block;
        text-decoration: none;
        color: white;
        background: var(--red);
        padding: 6px 12px;
        border-radius: 8px;
        transition: background 0.2s;
        }
        .card:hover .card-link {
        background: var(--darkred);
        }
        .cards::-webkit-scrollbar {
        height: 12px;
        }
        .cards::-webkit-scrollbar-thumb,
        .cards::-webkit-scrollbar-track {
        border-radius: 92px;
        }
        .cards::-webkit-scrollbar-thumb {
        background: var(--darkred);
        }
        .cards::-webkit-scrollbar-track {
        background: var(--thumb);
        }
        @media (width: 200%) {
        .card {
            flex-basis: calc(50% - 10px);
        }
        .card:not(:last-child) {
            margin-right: 20px;
        }
        }
        @media (min-width: 700px) {
        .card {
            flex-basis: calc(calc(100% / 3) - 20px);
        }
        .card:not(:last-child) {
            margin-right: 30px;
        }
        }
        @media (width: 1100px) {
        .card {
            flex-basis: calc(25% - 30px);
        }
        .card:not(:last-child) {
            margin-right: 40px;
        }
        }
        /* FOOTER STYLES
        –––––––––––––––––––––––––––––––––––––––––––––––––– */
        .page-footer {
        position: fixed;
        right: 0;
        bottom: 50px;
        display: flex;
        align-items: center;
        padding: 5px;
        z-index: 1;
        }
        .page-footer a {
        display: flex;
        margin-left: 4px;
        }
        </style>
        """,unsafe_allow_html=True)

with tab1:
    text = "Every year, approximately 1.5 lakh people dies on India roads, which translate, on an average, into 1130 accidents and 422 deaths every day or 47 accidents and 18 deaths every hour."
    st.info(text)
    size=(1100,500)
    image_paths = ['Images/accident_onedeath.jpg', 'Images/bike.jpg','Images/vehicle.jpg', 'Images/truck.jpg','Images/sch_bus.jpg','Images/accident4.jpg']
    delay = 3
    image_placeholder = st.empty()
    index = 0
    st.markdown("<br><hr>", unsafe_allow_html=True)
    st.markdown(
        """
        <style>
        .icon-container {
            bottom: 20px;
            right: 20px;
        }
        .icon-container a {
            margin-left: 10px;
        }
        </style>
        """,unsafe_allow_html=True)
    while(10):
        path = image_paths[index]
        image = Image.open(path)
        image = image.resize(size)
        image_placeholder.image(image)
        time.sleep(delay)
        image_placeholder.empty()
        index = (index + 1) % len(image_paths)








