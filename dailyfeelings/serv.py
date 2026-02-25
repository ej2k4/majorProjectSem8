from flask import Flask, request, jsonify
from flask_cors import CORS
import json, os

app = Flask(__name__)
CORS(app)

FILE = "feelings.json"

@app.route("/save-feeling", methods=["POST"])
def save_feeling():
    entry = request.get_json()
    
    # Load existing data
    if os.path.exists(FILE):
        with open(FILE, "r") as f:
            data = json.load(f)
    else:
        data = []
    
    # Append new entry and save
    data.append(entry)
    with open(FILE, "w") as f:
        json.dump(data, f, indent=2)
    
    return jsonify({"status": "saved"})

if __name__ == "__main__":
    app.run(port=5000)