import streamlit as st
import pandas as pd
import plotly.express as px
import time
import json
import joblib
import requests
import paho.mqtt.client as mqtt
import queue
from datetime import datetime
import warnings
import os

# ==========================================
# 1. KONFIGURASI HALAMAN & GLOBAL
# ==========================================
st.set_page_config(
    page_title="The Avicenna Dashboard",
    page_icon="🏫",
    layout="wide"
)

# Suppress warning sklearn
warnings.filterwarnings("ignore")

# --- KONFIGURASI KREDENSIAL ---
MQTT_BROKER = "broker.hivemq.com" # Menggunakan public broker yang sama dengan ESP32
MQTT_PORT = 1883
TOPIC_SENSOR = "kelas/data"      # Disesuaikan dengan ESP32 (topic publish)
TOPIC_CONTROL = "kelas/kontrol"  # (Tidak digunakan karena kontrol manual dihapus)
MODEL_FILENAME = "FinalProject3_KNN_Avicenna.pkl"
OWM_API_KEY = "4b031f7ed240d398ab4b7696d2361d97"
CITY_NAME = "Sukabumi,ID"
CSV_LOG_FILE = "live_data_dashboard.csv"

# ==========================================
# 2. FUNGSI-FUNGSI LOGIKA (ML & API)
# ==========================================

# Load Model ML
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_FILENAME):
        return "MISSING"
    try:
        model = joblib.load(MODEL_FILENAME)
        return model
    except:
        return None

# Load Queue MQTT
@st.cache_resource
def get_data_queue():
    return queue.Queue()

# Fungsi Cek Cuaca (Return Suhu & Deskripsi)
@st.cache_data(ttl=600) 
def get_weather_cached():
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={CITY_NAME}&appid={OWM_API_KEY}&units=metric&lang=id"
        response = requests.get(url, timeout=5).json()
        if response.get("cod") == 200:
            temp = response["main"]["temp"]
            desc = response["weather"][0]["description"]
            return temp, desc
        return None, None
    except:
        return None, None

# Fungsi Simpan ke CSV
def save_to_csv(data_dict):
    df_new = pd.DataFrame([data_dict])
    if not os.path.exists(CSV_LOG_FILE):
        df_new.to_csv(CSV_LOG_FILE, index=False)
    else:
        df_new.to_csv(CSV_LOG_FILE, mode='a', header=False, index=False)

# ==========================================
# 3. SETUP MQTT CLIENT
# ==========================================

def on_message(client, userdata, msg):
    try:
        payload = msg.payload.decode()
        # Debugging: print payload yang masuk
        print(f"RAW MQTT: {payload}")
        
        data = json.loads(payload)
        
        # Mapping data JSON dari ESP32 (suhu/kelembaban) ke format dashboard
        # ESP32 mengirim: {"suhu": 25.0, "kelembaban": 60.0}
        # Dashboard butuh: temp, hum, lux (lux kita set default 0 jika tidak ada)
        processed_data = {
            "temp": data.get("temp", 0),
            "hum": data.get("hum", 0),
            "lux": data.get("lux", 0) # Default 0 karena DHT22 tidak baca cahaya
        }
        
        q = get_data_queue()
        q.put(processed_data)
    except Exception as e:
        print(f"Error parsing MQTT: {e}")

@st.cache_resource
def start_mqtt_client():
    client = mqtt.Client()
    client.on_message = on_message
    try:
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
        client.subscribe(TOPIC_SENSOR)
        client.loop_start()
        return client
    except Exception as e:
        st.error(f"Gagal koneksi MQTT: {e}")
        return None

# ==========================================
# 4. TAMPILAN DASHBOARD (SIDEBAR & MAIN)
# ==========================================

# --- SIDEBAR CONFIGURATION ---
with st.sidebar:
    st.image("https://wp-umbrella.com/wp-content/uploads/2021/06/monitoring.png", width=100)
    st.title("The Avicenna")
    st.subheader("Panel Monitoring")
    
    # Info Cuaca
    st.divider()
    st.write(f"📍 **Lokasi:** {CITY_NAME}")
    temp_out_raw, desc_out = get_weather_cached()
    
    if temp_out_raw:
        st.metric("🌤️ Cuaca Luar", f"{temp_out_raw}°C", desc_out.title())
    else:
        st.warning("Gagal memuat cuaca")

    # Inisialisasi MQTT & Model
    model = load_model()
    client = start_mqtt_client()

    # --- BAGIAN KONTROL MANUAL DIHAPUS ---

    # Status Koneksi MQTT
    st.divider()
    if client:
        st.success(f"🟢 MQTT Terhubung\nBroker: {MQTT_BROKER}")
    else:
        st.error("🔴 MQTT Terputus")
        
    # Download Data
    st.divider()
    st.write("📂 **Log Data**")
    if os.path.exists(CSV_LOG_FILE):
        with open(CSV_LOG_FILE, "rb") as f:
            st.download_button("Unduh CSV", f, file_name="log_avicenna.csv", mime="text/csv")

# --- MAIN PAGE ---
st.title("🏫 Class Monitoring System")
st.markdown("Dashboard monitoring suhu, kelembaban, dan kenyamanan kelas secara real-time.")
st.divider()

# Inisialisasi Queue & Session State
data_queue = get_data_queue()

if "df_live" not in st.session_state:
    st.session_state.df_live = pd.DataFrame(columns=["Timestamp", "Temp_In", "Hum_In", "Lux_In", "Temp_Out", "Prediction"])

# Container UI Auto-Refresh
placeholder = st.empty()

# ==========================================
# 5. LOOP UTAMA
# ==========================================
while True:
    # 1. Proses Data Baru dari MQTT
    while not data_queue.empty():
        raw_data = data_queue.get()
        
        # Parsing
        temp = float(raw_data.get("temp", 0))
        hum = float(raw_data.get("hum", 0))
        lux = int(raw_data.get("lux", 0))
        ts = datetime.now().strftime("%H:%M:%S")
        
        # Ambil Cuaca (dari cache sidebar tadi)
        t_out = temp_out_raw if temp_out_raw else 25.0
        
        # Prediksi ML
        prediction = "Normal"
        if model and model != "MISSING":
            try:
                # Pastikan input sesuai dengan fitur model (misal temp & hum)
                pred = model.predict([[temp, hum]])[0]
                prediction = str(pred)
            except:
                prediction = "Error ML"
        elif model == "MISSING":
            prediction = "No Model"

        # Data Row Dictionary
        new_row_dict = {
            "Timestamp": ts,
            "Temp_In": temp,
            "Hum_In": hum,
            "Lux_In": lux,
            "Temp_Out": t_out,
            "Prediction": prediction
        }

        # Simpan ke CSV Lokal
        save_to_csv(new_row_dict)

        # Update Session State (untuk Grafik)
        st.session_state.df_live = pd.concat([st.session_state.df_live, pd.DataFrame([new_row_dict])], ignore_index=True)

    # 2. Render Tampilan
    df = st.session_state.df_live
    
    with placeholder.container():
        if df.empty:
            st.info(f"📡 Menunggu data sensor dari topic '{TOPIC_SENSOR}'...")
            time.sleep(1)
            continue

        last_row = df.iloc[-1]
        status = last_row['Prediction']

        # --- ALERT KHUSUS JIKA PANAS ---
        if status == "Panas":
            st.error(f"🔥 **PERINGATAN SUHU TINGGI!** Suhu Kelas mencapai {last_row['Temp_In']}°C.")
        
        # --- METRIK UTAMA ---
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🌡️ Suhu Kelas", f"{last_row['Temp_In']} °C", f"{last_row['Temp_In'] - last_row['Temp_Out']:.1f}°C vs Luar")
        c2.metric("💧 Kelembaban", f"{last_row['Hum_In']} %")
        c3.metric("💡 Cahaya", f"{last_row['Lux_In']} Lux")
        
        # Ikon Status AI
        if status == "Panas": 
            c4.metric("🧠 Status AI", status, "Bahaya", delta_color="inverse")
        elif status == "Dingin":
            c4.metric("🧠 Status AI", status, "Dingin", delta_color="normal")
        else:
            c4.metric("🧠 Status AI", status, "Nyaman")

        # --- GRAFIK TERPISAH (DETAIL) ---
        st.subheader("📈 Analisis Real-time")
        unique_key = int(time.time() * 1000000) # Key unik untuk refresh
        
        # Ambil 50 data terakhir agar grafik tidak terlalu padat
        df_chart = df.tail(50)

        # 1. Grafik Suhu (Temp In & Out Gabung) - Full Width
        fig_temp = px.line(df_chart, x="Timestamp", y=["Temp_In", "Temp_Out"], 
                           title="Suhu Kelas vs Suhu Luar (°C)", markers=True,
                           color_discrete_map={"Temp_In": "#FF4B4B", "Temp_Out": "#FFA15A"})
        fig_temp.update_layout(height=350, hovermode="x unified")
        st.plotly_chart(fig_temp, width="stretch", key=f"chart_temp_{unique_key}")

        # 2. Grafik Kelembaban & Lux - Side by Side
        col_g1, col_g2 = st.columns(2)
        
        with col_g1:
            fig_hum = px.area(df_chart, x="Timestamp", y="Hum_In", 
                              title="Kelembaban Udara (%)", markers=True,
                              color_discrete_sequence=["#00CC96"])
            fig_hum.update_layout(height=300, hovermode="x unified")
            st.plotly_chart(fig_hum, width="stretch", key=f"chart_hum_{unique_key}")

        with col_g2:
            fig_lux = px.line(df_chart, x="Timestamp", y="Lux_In", 
                              title="Intensitas Cahaya (Lux)", markers=True,
                              color_discrete_sequence=["#FFC107"])
            fig_lux.update_layout(height=300, hovermode="x unified")
            st.plotly_chart(fig_lux, width="stretch", key=f"chart_lux_{unique_key}")

        # --- TABEL DATA ---
        with st.expander("Lihat Data Log Terakhir"):
            st.dataframe(df.sort_index(ascending=False).head(5))

    time.sleep(1)