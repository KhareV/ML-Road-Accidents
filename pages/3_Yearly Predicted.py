import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
from io import BytesIO
import base64
from reportlab.lib.pagesizes import LETTER
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.lib.units import inch
from reportlab.graphics.shapes import Drawing
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from reportlab.lib import colors


st.set_page_config(page_title="Predictions", page_icon="📊", layout="wide")
logo = Image.open("Images/logo.jpg")
st.image(logo, width=100)
gif_url = "https://ewscripps.brightspotcdn.com/dims4/default/7c4e68f/2147483647/strip/true/crop/800x450+0+0/resize/320x180!/quality/90/?url=http%3A%2F%2Fewscripps-brightspot.s3.amazonaws.com%2F3d%2F82%2Fcd5de20740f5b50f843fab609997%2Fcaraccidentgif-storyblocks.gif"
st.sidebar.image(gif_url, use_column_width=True)

def generate_pdf(year_data, selected_year):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=LETTER)
    c.setTitle("Road Accident Report")

    # Add a border around the page
    c.rect(0.5*inch, 0.75*inch,7.5*inch, 10*inch)
    # Add a title
    c.setFont("Helvetica-Bold", 16)
    c.drawCentredString(4.25*inch, 9.5*inch, "Road Accident Report")

    # Add a subtitle with the selected year
    c.setFont("Helvetica", 12)
    c.drawCentredString(4.25*inch, 9*inch, f"Report for the year {selected_year}")

    # Add a table with the statistics
    table_data = [
        ["Statistic", "Value"],
        ["Total Number of Road Accidents (in numbers)", f"{round(year_data['Total Number of Road Accidents (in numbers)'].values[0],2)}"],
        ["Total Number of Persons Killed (in numbers)", f"{round(year_data['Total Number of Persons Killed (in numbers)'].values[0],2)}"],
        ["Total Number of Persons Injured (in numbers)", f"{round(year_data['Total Number of Persons Injured (in numbers)'].values[0],2)}"]
    ]
    table_style = [
        ('BACKGROUND', (0,0), (-1,0), colors.grey),
        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
        ('ALIGN', (0,0), (-1,0), 'CENTER'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,0), 12),
        ('BOTTOMPADDING', (0,0), (-1,0), 12),
        ('BACKGROUND', (0,1), (-1,-1), colors.beige),
        ('TEXTCOLOR', (0,1), (-1,-1), colors.black),
        ('ALIGN', (0,1), (-1,-1), 'CENTER'),
        ('FONTNAME', (0,1), (-1,-1), 'Helvetica'),
        ('FONTSIZE', (0,1), (-1,-1), 10),
        ('BOTTOMPADDING', (0,1), (-1,-1), 10),
        ('GRID', (0,0), (-1,-1), 1, colors.black),
    ]
    c.drawCentredString(4.25*inch, 8.5*inch, "Statistics")
    table_width = 6.5*inch
    table_x = (8.5*inch - table_width) / 2
    from reportlab.lib.pagesizes import landscape, A4
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
    data = [["Statistic", "Value"],
        ["Total Number of Road Accidents (in numbers)", f"{round(year_data['Total Number of Road Accidents (in numbers)'].values[0],2)}"],
        ["Total Number of Persons Killed (in numbers)", f"{round(year_data['Total Number of Persons Killed (in numbers)'].values[0],2)}"],
        ["Total Number of Persons Injured (in numbers)", f"{round(year_data['Total Number of Persons Injured (in numbers)'].values[0],2)}"]]
    table = Table(data)
    table.setStyle(TableStyle(table_style))
    table.wrapOn(c, 0, 0)
    table.drawOn(c, 2*inch, 6.5*inch)
    report_data = "In the year "+str(selected_year)+" a total of "+str(round(year_data['Total Number of Road Accidents (in numbers)'].values[0],2))+" road accidents are likely to happen which may result, in the loss of "+str(round(year_data['Total Number of Persons Killed (in numbers)'].values[0],2))+" lives. "
    report_data+="However accidents do not always result in loss of lives but may, rather lead to fatal injuries. The total number of persons predicted to be injured in road accidents, is "+str(round(year_data['Total Number of Persons Injured (in numbers)'].values[0],2))+"."
    c.setFont("Helvetica", 12)
    lines = report_data.split(",")

    # Set the y-coordinate for the first line
    y = 6*inch
    x=1*inch
    # Loop through the lines and draw them on the canvas
    for line in lines:
        c.drawString(x, y, line)
        y -= 0.25*inch
        x=0.75*inch
    #c.drawCentredString(3.75*inch, 6*inch, report_data)
    # Add a chart with the data
    data = [round(year_data['Total Number of Road Accidents (in numbers)'].values[0],2),
        round(year_data['Total Number of Persons Killed (in numbers)'].values[0],2),
        round(year_data['Total Number of Persons Injured (in numbers)'].values[0],2)
    ]
    drawing = Drawing(400, 300)
    chart = VerticalBarChart()
    chart.width = 350
    chart.height = 150
    chart.data = [data]
    chart.x = 50
    chart.y = 0
    chart.strokeColor = colors.black
    chart.valueAxis.valueMin = 0
    chart.categoryAxis.labels.boxAnchor = 'ne'
    chart.categoryAxis.labels.dx = 8
    chart.categoryAxis.labels.dy = -2
    chart.categoryAxis.labels.angle = 45
    chart.categoryAxis.categoryNames = ["Total Number of Road Accidents", "Total Number of Persons Killed", "Total Number of Persons Injured"]
    drawing.add(chart)
    c.drawCentredString(4*inch, 4.75*inch, "Statistics Chart")
    drawing.wrapOn(c, 0, 0)
    drawing.drawOn(c, 1*inch, 2*inch)

    c.showPage()
    c.save()

    buffer.seek(0)
    b64=base64.b64encode(buffer.getvalue()).decode('utf-8')
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="report.pdf">📋Download PDF</a>'
    return href



st.markdown("<h4 style='text-align: left; color:white; font-family: Comic Sans MS;'>Predicting Road Accidents</h4>", unsafe_allow_html=True)
st.markdown("<h6 style='text-align: left; color:white;font-size: 15px; font-family: Comic Sans MS;'>How Data Analysis Can Help Save Lives</h6>", unsafe_allow_html=True)

#tab1, tab2= st.tabs(["OverAll Prediction🔮", "Statewise prediction🗺️"])

data=pd.read_excel("Predictions/predicted_values.xlsx")
data1=pd.read_excel("Accident Dataset/1970-2021 data.xlsx")
if st.button("Open Excel"):
    if st.button("Close Excel"):
        data1 = None
    if data1 is not None:
        st.write(data1)
years = sorted(data['Year'].unique().tolist(), reverse=False)
years.insert(0, "All")
selected_year = st.selectbox("Select a year", years)
if selected_year == "All":
    filtered_data = data
else:
    filtered_data = data[data['Year'] == selected_year]
st.write(filtered_data)

if selected_year != "All":
    st.markdown(
    f"""
    <h1 style='text-align: left; color: cyan;font-family: Comic Sans MS; font-size: 20px;'>Report for Year {selected_year}</h1>
    """,
    unsafe_allow_html=True,
)
    #st.markdown(f"## Report for Year {selected_year}")
    year_data = filtered_data[filtered_data['Year'] == selected_year]
    report_data1 = f"In the year {selected_year}, it is predicted that:"
    report_data1 += f"\n- {round(year_data['Total Number of Road Accidents (in numbers)'].values[0],2)} road accidents will occur"
    report_data1 += f"\n- {round(year_data['Total Number of Persons Killed (in numbers)'].values[0],2)} people may lose their lives"
    report_data1 += f"\n- {round(year_data['Total Number of Persons Injured (in numbers)'].values[0],2)} people may be injured"
    st.markdown(report_data1)
    # generate PDF report
    st.markdown(
    f"""
    <h1 style='text-align: left; color: cyan;font-family: Comic Sans MS; font-size: 20px;'>Generate PDF report📄</h1>
    """,
    unsafe_allow_html=True,
)
    #st.markdown("## Generate PDF report")
    href = generate_pdf(year_data, selected_year)
    st.markdown(href, unsafe_allow_html=True)

    if st.checkbox("Show historical data", False):
        st.markdown(
    f"""
    <h1 style='text-align: left; color: cyan;font-family: Comic Sans MS; font-size: 20px;'>Historical Data📅</h1>
    """,
    unsafe_allow_html=True,
)
        st.write(data1)
    # plot historical data
    st.markdown(
    f"""
    <h1 style='text-align: left; color: cyan;font-family: Comic Sans MS; font-size: 20px;'>Historical Data Visualization📈</h1>
    """,
    unsafe_allow_html=True,
)
    #st.markdown(f"## Historical Data Visualization")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(data1["Year"], data1["Total Number of Road Accidents (in numbers)"], label="Road Accidents")
    ax.plot(data1["Year"], data1["Total Number of Persons Killed (in numbers)"], label="Persons Killed")
    ax.plot(data1["Year"], data1["Total Number of Persons Injured (in numbers)"], label="Persons Injured")
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of incidents")
    ax.set_title("Historical Data for Road Accidents in India")
    ax.legend()
    st.pyplot(fig)
    st.markdown(
    f"""
    <h1 style='text-align: left; color: cyan;font-family: Comic Sans MS; font-size: 20px;'>Model Information ℹ️</h1>
    """,
    unsafe_allow_html=True,
)
    #st.markdown("## Model Information")
    model_info = """

    The model used is a Time Series Forecasting model using Auto Regressive Integrated Moving Average (ARIMA) method.
    The dataset used is the '1970-2021 data.xlsx' file which contains historical data on Road Accidents in India.
    The model was trained on data upto the year 2019 and tested on data for the year 2020 and 2021.
    The predicted values for the years 2022-2031 are stored in the 'predicted_values.xlsx' file.
    The model takes into account the seasonality and trend in the data to make predictions.
    """
    st.markdown(model_info)


st.sidebar.write("More info : https://morth.nic.in/")
st.markdown("<br><hr>", unsafe_allow_html=True)
st.markdown("<h20 style='text-align: centre; color:white; font-size: 10px;font-family: Comic Sans MS;'>This app was created as a part of the course project for Data Analysis and Visualization (DAVL).</h20>", unsafe_allow_html=True)
st.markdown("<h25 style='text-align: centre; color:white;font-size: 8px; font-family: Comic Sans MS;'>Made with ❤️</h25>", unsafe_allow_html=True)

