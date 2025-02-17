import streamlit as st
from PIL import Image

st.set_page_config(page_title="Predictions", page_icon="📊", layout="wide")
logo = Image.open("Images/logo.jpg")
st.image(logo, width=100)
gif_url = "https://ewscripps.brightspotcdn.com/dims4/default/7c4e68f/2147483647/strip/true/crop/800x450+0+0/resize/320x180!/quality/90/?url=http%3A%2F%2Fewscripps-brightspot.s3.amazonaws.com%2F3d%2F82%2Fcd5de20740f5b50f843fab609997%2Fcaraccidentgif-storyblocks.gif"
st.sidebar.image(gif_url, use_column_width=True)
tab1, tab2,tab3,tab4,tab5,tab6,tab7,tab8= st.tabs(["Yearly Analysis","Black Spots","Cause","Weather","Gender and Age","License and Vehicle","Road and Junction","Traffic control and Collision"])
with tab1:
     #st.markdown("""<iframe title="PB_page1" width="1140" height="541.25" src="https://app.powerbi.com/reportEmbed?reportId=5b7274ea-a37b-43db-aeab-2f9ba76ce24b&autoAuth=true&ctid=da780102-affb-4105-8d1a-34ddb1b9fe7c" frameborder="0" allowFullScreen="true"></iframe>""",unsafe_allow_html=True)
     power_bi_url = "https://app.powerbi.com/reportEmbed?reportId=5b7274ea-a37b-43db-aeab-2f9ba76ce24b&autoAuth=true&ctid=da780102-affb-4105-8d1a-34ddb1b9fe7c"

     embedded_url = power_bi_url + "&toolbarHidden=true&navContentPaneEnabled=false&filterPaneEnabled=false"

     st.markdown(f'<div style="display: flex; justify-content: center;"><iframe width="1000" height="541.25" src="{embedded_url}" frameborder="0" allowFullScreen="true"></iframe></div>', unsafe_allow_html=True)
with tab2:
     #st.markdown("""<iframe title="PB_page1" width="1140" height="541.25" src="https://app.powerbi.com/reportEmbed?reportId=5b7274ea-a37b-43db-aeab-2f9ba76ce24b&autoAuth=true&ctid=da780102-affb-4105-8d1a-34ddb1b9fe7c" frameborder="0" allowFullScreen="true"></iframe>""",unsafe_allow_html=True)
     power_bi_url = "https://app.powerbi.com/reportEmbed?reportId=cd8a3604-9c09-497a-aa05-cd5b2ce15c62&autoAuth=true&ctid=da780102-affb-4105-8d1a-34ddb1b9fe7c"

     embedded_url = power_bi_url + "&toolbarHidden=true&navContentPaneEnabled=false&filterPaneEnabled=false"

     st.markdown(f'<div style="display: flex; justify-content: center;"><iframe width="1000" height="541.25" src="{embedded_url}" frameborder="0" allowFullScreen="true"></iframe></div>', unsafe_allow_html=True)
with tab3:
     #st.markdown("""<iframe title="PB_page1" width="1140" height="541.25" src="https://app.powerbi.com/reportEmbed?reportId=5b7274ea-a37b-43db-aeab-2f9ba76ce24b&autoAuth=true&ctid=da780102-affb-4105-8d1a-34ddb1b9fe7c" frameborder="0" allowFullScreen="true"></iframe>""",unsafe_allow_html=True)
     power_bi_url = "https://app.powerbi.com/reportEmbed?reportId=480a875d-e925-4902-aa8e-014c0c7c119a&autoAuth=true&ctid=da780102-affb-4105-8d1a-34ddb1b9fe7c"

     embedded_url = power_bi_url + "&toolbarHidden=true&navContentPaneEnabled=false&filterPaneEnabled=false"

     st.markdown(f'<div style="display: flex; justify-content: center;"><iframe width="1000" height="541.25" src="{embedded_url}" frameborder="0" allowFullScreen="true"></iframe></div>', unsafe_allow_html=True)
with tab4:
     #st.markdown("""<iframe title="PB_page1" width="1140" height="541.25" src="https://app.powerbi.com/reportEmbed?reportId=5b7274ea-a37b-43db-aeab-2f9ba76ce24b&autoAuth=true&ctid=da780102-affb-4105-8d1a-34ddb1b9fe7c" frameborder="0" allowFullScreen="true"></iframe>""",unsafe_allow_html=True)
     power_bi_url = "https://app.powerbi.com/reportEmbed?reportId=a8b7e535-3de2-496d-97a3-a4d16458e7ef&autoAuth=true&ctid=da780102-affb-4105-8d1a-34ddb1b9fe7c"

     embedded_url = power_bi_url + "&toolbarHidden=true&navContentPaneEnabled=false&filterPaneEnabled=false"

     st.markdown(f'<div style="display: flex; justify-content: center;"><iframe width="1000" height="541.25" src="{embedded_url}" frameborder="0" allowFullScreen="true"></iframe></div>', unsafe_allow_html=True)
with tab5:
     #st.markdown("""<iframe title="PB_page1" width="1140" height="541.25" src="https://app.powerbi.com/reportEmbed?reportId=5b7274ea-a37b-43db-aeab-2f9ba76ce24b&autoAuth=true&ctid=da780102-affb-4105-8d1a-34ddb1b9fe7c" frameborder="0" allowFullScreen="true"></iframe>""",unsafe_allow_html=True)
     power_bi_url = "https://app.powerbi.com/reportEmbed?reportId=41bf9a18-b21c-4ce4-9ab0-74011f6a57bb&autoAuth=true&ctid=da780102-affb-4105-8d1a-34ddb1b9fe7c"

     embedded_url = power_bi_url + "&toolbarHidden=true&navContentPaneEnabled=false&filterPaneEnabled=false"

     st.markdown(f'<div style="display: flex; justify-content: center;"><iframe width="1000" height="541.25" src="{embedded_url}" frameborder="0" allowFullScreen="true"></iframe></div>', unsafe_allow_html=True)
with tab6:
     #st.markdown("""<iframe title="PB_page1" width="1140" height="541.25" src="https://app.powerbi.com/reportEmbed?reportId=5b7274ea-a37b-43db-aeab-2f9ba76ce24b&autoAuth=true&ctid=da780102-affb-4105-8d1a-34ddb1b9fe7c" frameborder="0" allowFullScreen="true"></iframe>""",unsafe_allow_html=True)
     power_bi_url = "https://app.powerbi.com/reportEmbed?reportId=f22803b7-7082-47a1-bfad-ce8c0b3c951c&autoAuth=true&ctid=da780102-affb-4105-8d1a-34ddb1b9fe7c"

     embedded_url = power_bi_url + "&toolbarHidden=true&navContentPaneEnabled=false&filterPaneEnabled=false"

     st.markdown(f'<div style="display: flex; justify-content: center;"><iframe width="1000" height="541.25" src="{embedded_url}" frameborder="0" allowFullScreen="true"></iframe></div>', unsafe_allow_html=True)
with tab7:
     #st.markdown("""<iframe title="PB_page1" width="1140" height="541.25" src="https://app.powerbi.com/reportEmbed?reportId=5b7274ea-a37b-43db-aeab-2f9ba76ce24b&autoAuth=true&ctid=da780102-affb-4105-8d1a-34ddb1b9fe7c" frameborder="0" allowFullScreen="true"></iframe>""",unsafe_allow_html=True)
     power_bi_url = "https://app.powerbi.com/reportEmbed?reportId=956a75fb-8128-4973-b154-41441834f2a6&autoAuth=true&ctid=da780102-affb-4105-8d1a-34ddb1b9fe7c"

     embedded_url = power_bi_url + "&toolbarHidden=true&navContentPaneEnabled=false&filterPaneEnabled=false"

     st.markdown(f'<div style="display: flex; justify-content: center;"><iframe width="1000" height="541.25" src="{embedded_url}" frameborder="0" allowFullScreen="true"></iframe></div>', unsafe_allow_html=True)
with tab8:
     #st.markdown("""<iframe title="PB_page1" width="1140" height="541.25" src="https://app.powerbi.com/reportEmbed?reportId=5b7274ea-a37b-43db-aeab-2f9ba76ce24b&autoAuth=true&ctid=da780102-affb-4105-8d1a-34ddb1b9fe7c" frameborder="0" allowFullScreen="true"></iframe>""",unsafe_allow_html=True)
     power_bi_url = "https://app.powerbi.com/reportEmbed?reportId=be93637f-1694-4695-a085-1e3137297acb&autoAuth=true&ctid=da780102-affb-4105-8d1a-34ddb1b9fe7c"

     embedded_url = power_bi_url + "&toolbarHidden=true&navContentPaneEnabled=false&filterPaneEnabled=false"

     st.markdown(f'<div style="display: flex; justify-content: center;"><iframe width="1000" height="541.25" src="{embedded_url}" frameborder="0" allowFullScreen="true"></iframe></div>', unsafe_allow_html=True)
st.sidebar.write("More info : https://morth.nic.in/")
st.markdown("<br><hr>", unsafe_allow_html=True)
st.markdown("<h20 style='text-align: centre; color:white; font-size: 10px;font-family: Comic Sans MS;'>This app was created as a part of the course project for Data Analysis and Visualization (DAVL).</h20>", unsafe_allow_html=True)
st.markdown("<h25 style='text-align: centre; color:white;font-size: 8px; font-family: Comic Sans MS;'>Made with ❤️</h25>", unsafe_allow_html=True)

