import streamlit as st
import extra_streamlit_components as stx
from streamlit_folium import folium_static
from PIL import Image
import pandas as pd
import folium

st.set_page_config(page_title="Predictions", page_icon="📊", layout="wide")
logo = Image.open("Images/logo.jpg")
st.image(logo, width=100)
df2=pd.read_excel("Predictions/statewise_injured.xlsx")
df3=pd.read_excel("Predictions/statewise_Killed.xlsx")
df1=pd.read_excel("Predictions/statewise_Accidents.xlsx")
data1=pd.read_excel("Accident Dataset/2018-2021 data.xlsx")
state_names = df1["State/UT"].unique()
years = df1.columns[1:-1].tolist()
selected_state = st.sidebar.selectbox("Select a state", state_names)
gif_url = "https://ewscripps.brightspotcdn.com/dims4/default/7c4e68f/2147483647/strip/true/crop/800x450+0+0/resize/320x180!/quality/90/?url=http%3A%2F%2Fewscripps-brightspot.s3.amazonaws.com%2F3d%2F82%2Fcd5de20740f5b50f843fab609997%2Fcaraccidentgif-storyblocks.gif"
st.sidebar.image(gif_url, use_column_width=True)


if st.button("Open State Excel"):
    if st.button("Close State Excel"):
        data1 = None
    st.write(data1)
    if data1 is not None:
        st.write(data1)



st.header("Statewise Road Safety Data")
chosen_id = stx.tab_bar(data=[
    stx.TabBarItemData(id="tab1", title="Accidents🤕", description="Unexpected events"),
    stx.TabBarItemData(id="tab2", title="Injured🏥", description="Taken care of"),
    stx.TabBarItemData(id="tab3", title="Killed🩸", description="Missed out")])

placeholder = st.container()

if chosen_id=="tab1":
    if selected_state != "All":
        df1 = df1[df1['State/UT'] == selected_state]
    st.write(df1)
    st.subheader("Map Visualization")
    m = folium.Map(location=[20.5937, 78.9629], zoom_start=4,tiles='CartoDB dark_matter')
    df_locations = pd.read_csv("Accident Dataset/statewise_loactions.csv")
    marker_icon = folium.features.CustomIcon(
        'Images/marker_icon.png',
        icon_size=(30, 30)
    )

    for index, row in df_locations.iterrows():
        if row['State/UT'] == selected_state:
            state = df1.columns[1:-1].tolist()
            acc_22 = round(df1[df1['State/UT'] == selected_state][state[0]].values[0],2)
            acc_23 = round(df1[df1['State/UT'] == selected_state][state[1]].values[0],2)
            acc_24 = round(df1[df1['State/UT'] == selected_state][state[2]].values[0],2)
            acc_25 = round(df1[df1['State/UT'] == selected_state][state[3]].values[0],2)
            acc_26 = round(df1[df1['State/UT'] == selected_state][state[4]].values[0],2)
            tooltip = folium.features.GeoJsonTooltip(fields=['State/UT', 'Accidents'],
                                            aliases=['State:', 'Accidents:'],
                                            localize=True)
            popup_html=f"<b>State:</b> {selected_state}<br><b>Accidents 2022:</b> {acc_22}<br><b>Accidents 2023:</b> {acc_23}<br><b>Accidents 2024:</b> {acc_24}<br><b>Accidents 2025:</b> {acc_25}<br><b>Accidents 2026:</b> {acc_26}"
            popup = folium.Popup(popup_html, max_width=300)
            folium.Marker([row['latitude'], row['longitude']],
                        icon=folium.Icon(color='red', icon='pushpin'),
                        tooltip=tooltip,popup=popup).add_to(m)
            
                        
        else:
            folium.Marker([row['latitude'], row['longitude']],
                        icon=None,
                        tooltip=row['State/UT']).add_to(m)
    folium_static(m)


elif chosen_id=="tab2":
    if selected_state != "All":
        df2 = df2[df2['State/UT'] == selected_state]
    st.write(df2)
    st.subheader("Map Visualization")
    m = folium.Map(location=[20.5937, 78.9629], zoom_start=4,tiles='CartoDB dark_matter')
    df_locations = pd.read_csv("Accident Dataset/statewise_loactions.csv")
    marker_icon = folium.features.CustomIcon(
        'Images/marker_icon.png',
        icon_size=(30, 30)
    )

    for index, row in df_locations.iterrows():
        if row['State/UT'] == selected_state:
            state = df2.columns[1:-1].tolist()
            acc_22 = round(df2[df2['State/UT'] == selected_state][state[0]].values[0],2)
            acc_23 = round(df2[df2['State/UT'] == selected_state][state[1]].values[0],2)
            acc_24 = round(df2[df2['State/UT'] == selected_state][state[2]].values[0],2)
            acc_25 = round(df2[df2['State/UT'] == selected_state][state[3]].values[0],2)
            acc_26 = round(df2[df2['State/UT'] == selected_state][state[4]].values[0],2)
            tooltip = folium.features.GeoJsonTooltip(fields=['State/UT', 'Accidents'],
                                            aliases=['State:', 'Accidents:'],
                                            localize=True)
            popup_html=f"<b>State:</b> {selected_state}<br><b>Accidents 2022:</b> {acc_22}<br><b>Accidents 2023:</b> {acc_23}<br><b>Accidents 2024:</b> {acc_24}<br><b>Accidents 2025:</b> {acc_25}<br><b>Accidents 2026:</b> {acc_26}"
            popup = folium.Popup(popup_html, max_width=300)
            folium.Marker([row['latitude'], row['longitude']],
                        icon=folium.Icon(color='red', icon='pushpin'),
                        tooltip=tooltip,popup=popup).add_to(m)
            
                        
        else:
            folium.Marker([row['latitude'], row['longitude']],
                        icon=None,
                        tooltip=row['State/UT']).add_to(m)
    folium_static(m)
elif chosen_id=="tab3":
    if selected_state != "All":
        df3 = df3[df3['State/UT'] == selected_state]
    st.write(df3)
    st.subheader("Map Visualization")
    m = folium.Map(location=[20.5937, 78.9629], zoom_start=4,tiles='CartoDB dark_matter')
    df_locations = pd.read_csv("Accident Dataset/statewise_loactions.csv")
    marker_icon = folium.features.CustomIcon(
        'Images/marker_icon.png',
        icon_size=(30, 30)
    )

    for index, row in df_locations.iterrows():
        if row['State/UT'] == selected_state:
            state = df3.columns[1:-1].tolist()
            acc_22 = round(df3[df3['State/UT'] == selected_state][state[0]].values[0],2)
            acc_23 = round(df3[df3['State/UT'] == selected_state][state[1]].values[0],2)
            acc_24 = round(df3[df3['State/UT'] == selected_state][state[2]].values[0],2)
            acc_25 = round(df3[df3['State/UT'] == selected_state][state[3]].values[0],2)
            acc_26 = round(df3[df3['State/UT'] == selected_state][state[4]].values[0],2)
            tooltip = folium.features.GeoJsonTooltip(fields=['State/UT', 'Accidents'],
                                            aliases=['State:', 'Accidents:'],
                                            localize=True)
            popup_html=f"<b>State:</b> {selected_state}<br><b>Accidents 2022:</b> {acc_22}<br><b>Accidents 2023:</b> {acc_23}<br><b>Accidents 2024:</b> {acc_24}<br><b>Accidents 2025:</b> {acc_25}<br><b>Accidents 2026:</b> {acc_26}"
            popup = folium.Popup(popup_html, max_width=300)
            folium.Marker([row['latitude'], row['longitude']],
                        icon=folium.Icon(color='red', icon='pushpin'),
                        tooltip=tooltip,popup=popup).add_to(m)
            
                        
        else:
            folium.Marker([row['latitude'], row['longitude']],
                        icon=None,
                        tooltip=row['State/UT']).add_to(m)
    folium_static(m)
else:
    placeholder = st.empty()

st.sidebar.write("More info : https://morth.nic.in/")
st.markdown("<br><hr>", unsafe_allow_html=True)
st.markdown("<h20 style='text-align: centre; color:white; font-size: 10px;font-family: Comic Sans MS;'>This app was created as a part of the course project for Data Analysis and Visualization (DAVL).</h20>", unsafe_allow_html=True)
st.markdown("<h25 style='text-align: centre; color:white;font-size: 8px; font-family: Comic Sans MS;'>Made with ❤️</h25>", unsafe_allow_html=True)


