import streamlit as st
from PIL import Image
import extra_streamlit_components as stx
from IPython.display import IFrame
from streamlit.components.v1 import components

st.set_page_config(page_title="Safety", page_icon="💞", layout="wide")
logo = Image.open("Images/logo.jpg")
st.image(logo, width=100)
gif_url = "https://ewscripps.brightspotcdn.com/dims4/default/7c4e68f/2147483647/strip/true/crop/800x450+0+0/resize/320x180!/quality/90/?url=http%3A%2F%2Fewscripps-brightspot.s3.amazonaws.com%2F3d%2F82%2Fcd5de20740f5b50f843fab609997%2Fcaraccidentgif-storyblocks.gif"
st.sidebar.image(gif_url, use_column_width=True)

tab1,tab2,tab3=st.tabs(["Rules","Services","Videos"])
image_visible = True


with tab1:
    image_visible = False
    chosen_id = stx.tab_bar(data=[
    stx.TabBarItemData(id="tab1", title="📝", description="Safety Rules"),
    stx.TabBarItemData(id="tab2", title="🚲", description="Rules for cars"),
    stx.TabBarItemData(id="tab3", title="🚗", description="Rules for bikes"),
    stx.TabBarItemData(id="tab4", title="👶🏻", description="Rules for kids"),
    stx.TabBarItemData(id="tab5", title="🛣️", description="Issues on road"),
    stx.TabBarItemData(id="tab6", title="👨‍👩‍👧‍👦", description="Rules for everyone")
    ])
    if chosen_id=="tab1":
        with open("Safety Rules/Rules.txt", "r") as f:
            text_content = f.read()
        st.text(text_content)
    elif chosen_id=="tab2":
        with open("Safety Rules/car_rules.txt", "r") as f:
            text_content = f.read()
        st.text(text_content)
    elif chosen_id=="tab3":
        with open("Safety Rules/bike_rules.txt", "r") as f:
            text_content = f.read()
        st.text(text_content)
    elif chosen_id=="tab4":
        with open("Safety Rules/kid_rules.txt", "r") as f:
            text_content = f.read()
        st.text(text_content)
    elif chosen_id=="tab5":
        with open("Safety Rules/road_issues.txt", "r") as f:
            text_content = f.read()
        st.text(text_content)
    elif chosen_id=="tab6":
        st.write(f"""
    <style>
        .streamlit-text-container {{
            max-width: 300px;
        }}
    </style>
    """, unsafe_allow_html=True)
        with open("Safety Rules/rules_everyone.txt", "r") as f:
            text_content = f.read()
        st.text(text_content)

with tab2:

    st.markdown("""
    <html>
    <head>
        <title>Services</title>
    <style>
        .container {
            display: flex;
            overflow-x: scroll;
            gap: 20px;
            margin:20px
        }
        .card {
            background-color: black;
                box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
                width: 300px;
                height: 400px;
                display: flex;
                flex-direction: column;
                align-items: center;
                padding: 20px;
                text-align: center;
                position: relative;
        }
        .card img {
            width: 250px;
            height: 200px;
            object-fit: contain;
            margin-bottom: 20px;
        }
        .card h3 {
            font-size: 20px;
            font-weight: bold;
            text-decoration: none;
            margin-bottom: 10px;
        }
        .card p {
            font-size: 16px;
            line-height: 1.5;
            margin-bottom: 20px;
            text-decoration: none;
            color: white;
        }
        a {
            text-decoration: none;
            color: white;
        }
    </style>
    </head>
    <body>
        <h3>License Related</h3>
        <div class="container">
            <a href="https://parivahan.gov.in/parivahan//en/content/learners-license" target="_blank">
                <div class="card">
                    <img src="https://parivahan.gov.in/parivahan//sites/default/files/images/v-learners-license-services.png" alt="Image 1">
                    <h3>Drivers/ Learners License</h3>
                    <p>License registration on your fingertips</p>
                </div>
            </a>
            <a href="https://parivahan.gov.in/parivahan//en/content/permanent-license" target="_blank">
                <div class="card">
                    <img src="https://parivahan.gov.in/parivahan//sites/default/files/images/v-driving-school-license.png" alt="Image 2">
                    <h3>Driving License</h3>
                    <p>One place for application of Driving School License.</p>
                </div>
            </a>
            <a href="https://parivahan.gov.in/parivahan//en/content/renewal" target="_blank">
                <div class="card">
                    <img src="https://parivahan.gov.in/parivahan//sites/default/files/images/v-driving-license.png" alt="Image 3">
                    <h3>Renewal</h3>
                    <p>Renew the Driving License.</p>
                </div>
            </a>
            </div>
        </div>
        <h3>Vehicle Related</h3>
        <div class="container">
            <a href="https://parivahan.gov.in/parivahan//en/content/temporary-registration" target="_blank">
                <div class="card">
                    <img src="https://parivahan.gov.in/parivahan//sites/default/files/images/v-new-vehicle-registration.png" alt="Image 1">
                    <h3>Temporary Registration</h3>
                    <p>Apply for temporary registration of a motor vehicle.</p>
                </div>
            </a>
            <a href="https://parivahan.gov.in/parivahan//en/content/permanent-registration" target="_blank">
                <div class="card">
                    <img src="https://parivahan.gov.in/parivahan//sites/default/files/images/v-permit-services.png" alt="Image 2">
                    <h3>Permanant Registration</h3>
                    <p>If temporarily registered then apply before expires.</p>
                </div>
            </a>
            <a href="https://parivahan.gov.in/parivahan//en/content/renewal-rc" target="_blank">
                <div class="card">
                    <img src="https://parivahan.gov.in/parivahan/sites/default/files/images/v-vehicle-registration.png" alt="Image 3">
                    <h3>Renewal of RC</h3>
                    <p>Renew before the date of its expiry of the registration.</p>
                </div>
            </a>
            </div>
        </div>
        <h3>Manufacturer Related Services</h3>
        <div class="container">
            <a href="https://vahan.parivahan.gov.in/vltdmaker/vahan/welcome.xhtml" target="_blank">
                <div class="card">
                    <img src="https://parivahan.gov.in/parivahan//sites/default/files/images/v-vltd-icon.png" alt="Image 1">
                    <h3>VLTD Maker</h3>
                    <p>Track, Analyze and Optimize Vehicle through location.</p>
                </div>
            </a>
            <a href="https://vahan.parivahan.gov.in/sldmaker/vahan/welcome.xhtml" target="_blank">
                <div class="card">
                    <img src="https://parivahan.gov.in/parivahan//sites/default/files/images/v-sld.png" alt="Image 2">
                    <h3>SLD Maker</h3>
                    <p>Speed Limiting Device Ecosystem.</p>
                </div>
            </a>
            <a href="https://vahan.parivahan.gov.in/cngmaker/vahan/welcome.xhtml" target="_blank">
                <div class="card">
                    <img src="https://parivahan.gov.in/parivahan//sites/default/files/images/v-cng-maker.png" alt="Image 3">
                    <h3>CNG Maker</h3>
                    <p>Make way for cleaner, greener fuel.</p>
                </div>
            </a>
            <a href="https://vahan.parivahan.gov.in/makermodel/vahan/welcome.xhtml" target="_blank">
                <div class="card">
                    <img src="https://parivahan.gov.in/parivahan//sites/default/files/images/v-homologation.png" alt="Image 3">
                    <h3>Homologation</h3>
                    <p>Complete Life Cycle of a vehicle.</p>
                </div>
            </a>
            </div>
        </div>
    </body>
    </html>
    """, unsafe_allow_html=True)






    


with tab3:
    image_visible = False
    kids_videos = [
    "https://www.youtube.com/embed/_NeEF1fwT4k",
    "https://www.youtube.com/embed/SJeqR9fsOPM","https://www.youtube.com/embed/aT61nwd5U-s","https://www.youtube.com/embed/PpEWVxNj2xo"]
    short_films=["https://www.youtube.com/embed/ep0DIE1PM-o",
    "https://www.youtube.com/embed/9M9ptAWbHE0","https://www.youtube.com/embed/ZGhBHncf66E","https://www.youtube.com/embed/H8QiaXYONNM"]
    incidents=["https://www.youtube.com/embed/aSJ_L7UyA3I","https://www.youtube.com/embed/t7911kgJJZc","https://www.youtube.com/embed/NCJrXUE-7ag",
               "https://www.youtube.com/embed/r_hblamyjyU"]
    
    media_html = ""
    for vid in kids_videos:
        video_html = f'<iframe width="560" height="315" src="{vid}" frameborder="0" allowfullscreen></iframe>'
        media_html += f'<div style="border: 1px solid black; padding: 10px; margin: 10px">{video_html}</div>'

    st.markdown(f"""<div style="display: flex; overflow-x: scroll; padding: 10px 0;">{media_html}</div>""", unsafe_allow_html=True)
    media_html = ""
    for vid1 in short_films:
        video_html = f'<iframe width="560" height="315" src="{vid1}" frameborder="0" allowfullscreen></iframe>'
        media_html += f'<div style="border: 1px solid black; padding: 10px; margin: 10px">{video_html}</div>'

    st.markdown(f"""<div style="display: flex; overflow-x: scroll; padding: 10px 0;">{media_html}</div>""", unsafe_allow_html=True)
    media_html = ""
    for vid1 in incidents:
        video_html = f'<iframe width="560" height="315" src="{vid1}" frameborder="0" allowfullscreen></iframe>'
        media_html += f'<div style="border: 1px solid black; padding: 10px; margin: 10px">{video_html}</div>'

    st.markdown(f"""<div style="display: flex; overflow-x: scroll; padding: 10px 0;">{media_html}</div>""", unsafe_allow_html=True)


st.sidebar.write("More info : https://morth.nic.in/")
st.markdown("<br><hr>", unsafe_allow_html=True)
st.markdown("<h20 style='text-align: centre; color:white; font-size: 10px;font-family: Comic Sans MS;'>This app was created as a part of the course project for Data Analysis and Visualization (DAVL).</h20>", unsafe_allow_html=True)
st.markdown("<h25 style='text-align: centre; color:white;font-size: 8px; font-family: Comic Sans MS;'>Made with ❤️</h25>", unsafe_allow_html=True)
