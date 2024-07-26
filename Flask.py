from datetime import datetime
import pandas as pd
import pymongo
import pytz
from flask import Flask, jsonify, request

app = Flask(__name__)

COMMAND_FILE = 'command.txt'


URI = "mongodb+srv://paradisaea:09071992@paradisaea.1sgdpuv.mongodb.net/?retryWrites=true&w=majority&appName=Paradisaea"
client = pymongo.MongoClient(URI)
db = client["paradisaea"]
coll = db["tes3"]

@app.route("/")
def root_route():
    """Route untuk menunjukkan data yang sudah tersimpan"""
    data = [x for x in coll.find()]
    df = pd.DataFrame.from_dict(data)
    return df.to_html(), 200

@app.route("/temp/all")
def get_temp_all():
    data = [x["temperature"] for x in coll.find()]
    return jsonify(data)

@app.route("/temp/avg")
def get_temp_avg():
    data = [x["temperature"] for x in coll.find()]
    total = sum(data)
    jumlah_data = len(data)
    avg = total / jumlah_data
    return jsonify({"average": avg})

@app.route("/hum/all")
def get_hum_all():
    data = [x["humidity"] for x in coll.find()]
    return jsonify(data)

@app.route("/hum/avg")
def get_hum_avg():
    data = [x["humidity"] for x in coll.find()]
    total = sum(data)
    jumlah_data = len(data)
    avg = total / jumlah_data
    return jsonify({"average": avg})

@app.route("/submit", methods=["POST"])
def submit_post():
    """Route untuk memasukkan data menggunakan POST, data dimasukkan sebagai JSON di body pesan."""
    timestamp = datetime.now(tz=pytz.timezone("Asia/Jakarta")).strftime(
        "%d/%m/%Y %H:%M:%S"
    )
    temp = float(request.get_json()["temp"])
    hum = float(request.get_json()["hum"])
    fan = int(request.get_json()["fan"])
    new_data = {
        "timestamp": timestamp,
        "temperature": temp,
        "humidity": hum,
        "fan": fan,
    }
    coll.insert_one(new_data.copy())
    return jsonify(new_data)

@app.route("/fan_command", methods=["GET"])
def get_fan_command():
    """Route untuk membaca command on/off dari file"""
    try:
        with open("command.txt", "r") as f:
            command = f.read().strip()
        return jsonify({"command": command})
    except FileNotFoundError:
        return jsonify({"command": "OFF"})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
    
    