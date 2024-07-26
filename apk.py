import pandas as pd
import plotly.graph_objects as go
import pymongo
import streamlit as st
from darts import TimeSeries
from darts.models import LightGBMModel
import pickle
from darts.utils.missing_values import fill_missing_values
import base64



# URI MongoDB
URI = "mongodb+srv://paradisaea:09071992@paradisaea.1sgdpuv.mongodb.net/?retryWrites=true&w=majority&appName=Paradisaea"
client = pymongo.MongoClient(URI)
db = client["paradisaea"]
collection = db["tes2"]

# Load model using pickle
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

add_bg_from_local('genshin .jpeg')


def ambil_data_terakhir():
    data_list = list(collection.find({}))
    data_raw = pd.DataFrame(data_list).drop("_id", axis=1)
    data_raw["timestamp"] = pd.to_datetime(data_raw["timestamp"]).dt.round("15min")
    data = (
        data_raw.groupby("timestamp")
        .agg(
            {
                "temperature": "mean",
                "humidity": "mean",
                # Asumsi komponen ketiga adalah "pressure"
                "fan": "max"
            }
        )
        .reset_index()
    )
    data = data.tail(20)  # ambil 20 data terakhir
    series = TimeSeries.from_dataframe(
        data,
        time_col="timestamp",
        fill_missing_dates=True,
        freq="15min",
    )
    series = fill_missing_values(series, fill="auto")
    return series

    


def lakukan_forecast(series, hours_ahead):
    steps = hours_ahead * 4  # 4 steps per hour karena frekuensi 15 menit
    predicted = model.predict(
        steps,
        series=series,
    )
    return predicted

# Judul utama
# st.title("Temperature & Humidity Monitoring")


font_css = '''
<style>
@import url('https://fonts.googleapis.com/css2?family=EB+Garamond&display=swap');

body {
    font-family: 'EB Garamond', serif;
    color: blue; /* Ganti warna teks di sini */
}

h1, h2, h3, h4, h5, h6 {
    color: #E0FFFF; /* Ganti warna teks untuk header di sini */
}
</style>
'''

st.markdown(font_css, unsafe_allow_html=True)


# Sidebar dengan pilihan halaman
st.sidebar.title("Menu")
page = st.sidebar.selectbox("Pilih halaman:", ["Data Terbaru", "Prediksi 1 Jam", "Prediksi 2 Jam"])

# Ambil data terbaru dan lakukan prediksi
data = ambil_data_terakhir()
predicted_1_hour = lakukan_forecast(data, 1)
predicted_2_hours = lakukan_forecast(data, 2)

if "matikan_kipas" not in st.session_state:
    st.session_state.matikan_kipas = False



elif page == "Data Terbaru":
    st.header("Data Terbaru")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Temperature Sekarang", f"{data['temperature'].values()[0][0]:.2f} C")
    with col2:
        st.metric("Kelembapan Sekarang", f"{data['humidity'].values()[0][0]:.2f} %")

    if data["temperature"].values()[0] < 25 and not st.session_state.matikan_kipas:
        st.warning("Suhu dingin, matikan Kipas!", icon="🥶")
        if st.button("Matikan Kipas"):
            st.session_state.matikan_kipas = True
            st.experimental_rerun()
            

    if st.session_state.matikan_kipas:
        st.success("Kipas Telah dimatikan")
        st.snow()

elif page == "Prediksi 1 Jam":
    st.header("Prediksi Suhu 1 Jam Ke Depan")
    

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=data.time_index,
            y=data["temperature"].values().flatten(),
            mode="lines",
            name="Temperature",
            line=dict(color="blue"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=predicted_1_hour.time_index,
            y=predicted_1_hour["temperature"].values().flatten(),
            mode="lines",
            name="Temperature (Predict)",
            line=dict(color="blue", dash="dash"),
        )
    )
    
     # Ubah background chart menjadi transparan
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    st.plotly_chart(fig)

elif page == "Prediksi 2 Jam":
    st.header("Prediksi Suhu 2 Jam Ke Depan")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=data.time_index,
            y=data["temperature"].values().flatten(),
            mode="lines",
            name="Temperature",
            line=dict(color="blue"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=predicted_2_hours.time_index,
            y=predicted_2_hours["temperature"].values().flatten(),
            mode="lines",
            name="Temperature (Predict)",
            line=dict(color="blue", dash="dash"),
        )
    )
    
     # Ubah background chart menjadi transparan
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    st.plotly_chart(fig)
    
    