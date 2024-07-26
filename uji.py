import pandas as pd
import plotly.graph_objects as go
import pymongo
import requests
import streamlit as st
from darts import TimeSeries
from darts.models import LightGBMModel
import pickle
from darts.utils.missing_values import fill_missing_values
import base64

# URL Flask endpoint
FLASK_URL = "http://192.168.1.75:5000/"

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
            background-image: url(data:image/{"jpeg"};base64,{encoded_string.decode()});
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

def ambil_data_sejarah():
    data_list = list(collection.find({}))
    data_raw = pd.DataFrame(data_list).drop("_id", axis=1)
    data_raw["timestamp"] = pd.to_datetime(data_raw["timestamp"]).dt.round("15min")
    data = (
        data_raw.groupby("timestamp")
        .agg(
            {
                "temperature": "mean",
                "humidity": "mean",
                "fan": "max"
            }
        )
        .reset_index()
    )
    return data

def lakukan_forecast(series, hours_ahead):
    steps = hours_ahead * 4  # 4 steps per hour karena frekuensi 15 menit
    predicted = model.predict(
        steps,
        series=series,
    )
    return predicted

font_css = '''
<style>
@import url('https://fonts.googleapis.com/css2?family=EB+Garamond&display=swap');

body {
    font-family: 'EB Garamond', serif;
    color: blue;
}

h1, h2, h3, h4, h5, h6 {
    color: #E0FFFF;
    font-weight: bold;
}
</style>
'''

st.markdown(font_css, unsafe_allow_html=True)

# Sidebar dengan pilihan halaman
page = st.sidebar.selectbox("Pilih halaman:", ["Home Page", "Dashboard Utama", "Data Terbaru", "Prediksi 1 Jam", "Prediksi 2 Jam", "History Temperature", "Statistik dan Laporan"])

if "matikan_kipas" not in st.session_state:
    st.session_state.matikan_kipas = False

# Ambil data terbaru dan lakukan prediksi
data = ambil_data_terakhir()
predicted_1_hour = lakukan_forecast(data, 1)
predicted_2_hours = lakukan_forecast(data, 2)

def write_command_to_file(command):
    try:
        with open("command.txt", "w") as f:
            f.write(command)
        return {"status": "success", "message": f"Kipas {command}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

if page == "Home Page":
    st.header("Selamat Datang di Website Kami")
    st.header("Silakan atur suhu sesuka Anda")
    
    # Input untuk mengatur suhu
    suhu_baru = st.number_input("Masukkan suhu yang diinginkan (dalam derajat Celsius):", min_value=0.0, max_value=100.0, step=0.1)
    
    if st.button("Atur Suhu"):
        # Simpan suhu yang diatur ke database MongoDB
        collection.update_one({}, {"$set": {"temperature": suhu_baru}}, upsert=True)
        st.success(f"Suhu berhasil diatur ke {suhu_baru:.2f} °C")

    # Command On/Off untuk kipas
    if st.button("Hidupkan Kipas"):
        response = write_command_to_file("ON")
        if response["status"] == "success":
            st.success("Kipas dihidupkan")
        else:
            st.error(f"Gagal menghidupkan kipas: {response['message']}")
        
    if st.button("Matikan Kipas"):
        response = write_command_to_file("OFF")
        if response["status"] == "success":
            st.success("Kipas dimatikan")
        else:
            st.error(f"Gagal mematikan kipas: {response['message']}")

elif page == "Dashboard Utama":
    st.header("Dashboard Utama")
    st.subheader("Status Sistem")

    # Menampilkan status terkini dari sistem pendingin
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Suhu Ruangan", f"{data['temperature'].values()[-1][0]:.2f} °C")
    with col2:
        st.metric("Kelembapan Ruangan", f"{data['humidity'].values()[-1][0]:.2f} %")
    with col3:
        kipas_status = "Aktif" if data['fan'].values()[-1] == 1 else "Tidak Aktif"
        st.metric("Status Kipas", kipas_status)

    st.subheader("Kontrol Utama")
    
    # Kontrol untuk menghidupkan atau mematikan sistem pendingin
    if st.button("Hidupkan Kipas"):
        response = write_command_to_file("ON")
        if response["status"] == "success":
            st.success("Kipas dihidupkan")
        else:
            st.error(f"Gagal menghidupkan kipas: {response['message']}")
        
    if st.button("Matikan Kipas"):
        response = write_command_to_file("OFF")
        if response["status"] == "success":
            st.success("Kipas dimatikan")
        else:
            st.error(f"Gagal mematikan kipas: {response['message']}")

    # Input untuk mengatur suhu target
    suhu_target = st.number_input("Masukkan suhu target (dalam derajat Celsius):", min_value=0.0, max_value=100.0, step=0.1)
    
    if st.button("Atur Suhu Target"):
        collection.update_one({}, {"$set": {"temperature_target": suhu_target}}, upsert=True)
        st.success(f"Suhu target berhasil diatur ke {suhu_target:.2f} °C")

elif page == "Data Terbaru":
    st.header("Data Terbaru")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Temperature Sekarang", f"{data['temperature'].values()[0][0]:.2f} C")
    with col2:
        st.metric("Kelembapan Sekarang", f"{data['humidity'].values()[0][0]:.2f} %")

    if data["temperature"].values()[0] < 25 and not st.session_state.matikan_kipas:
        st.warning("Suhu dingin, matikan Kipas!", icon="🥶")
        # Matikan = st.button("Matikan Kipas")
        # if Matikan:
        #     st.success("Kipas telah dimatikan")
            
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
            name="Predicted Temperature",
            line=dict(color="red", dash="dash"),
        )
    )
    fig.update_layout(
        title="Prediksi Suhu 1 Jam Ke Depan",
        xaxis_title="Waktu",
        yaxis_title="Temperature",
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
            name="Predicted Temperature",
            line=dict(color="red", dash="dash"),
        )
    )
    fig.update_layout(
        title="Prediksi Suhu 2 Jam Ke Depan",
        xaxis_title="Waktu",
        yaxis_title="Temperature",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig)

elif page == "History Temperature":
    st.header("History Suhu")

    data_history = ambil_data_sejarah()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=data_history["timestamp"],
            y=data_history["temperature"],
            mode="lines",
            name="Temperature",
            line=dict(color="blue"),
        )
    )
    fig.update_layout(
        title="History Temperature",
        xaxis_title="Waktu",
        yaxis_title="Temperature",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig)
    st.dataframe(data_history, use_container_width=True)  # Menampilkan dataframe dengan lebar penuh


elif page == "Statistik dan Laporan":
    st.header("Statistik dan Laporan")
    
    # Bagian Grafik Suhu
    st.subheader("Grafik Suhu")
    data_history = ambil_data_sejarah()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=data_history["timestamp"],
            y=data_history["temperature"],
            mode="lines",
            name="Temperature",
            line=dict(color="blue"),
        )
    )
    fig.update_layout(
        title="Grafik Suhu",
        xaxis_title="Waktu",
        yaxis_title="Temperature",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig)
    
    
    # Placeholder untuk informasi konsumsi energi
    st.subheader("Konsumsi Energi")
    st.write("Informasi konsumsi energi akan ditambahkan di sini.")
