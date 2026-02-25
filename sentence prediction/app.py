from flask import Flask, request, jsonify
from flask_cors import CORS
import sqlite3
from datetime import date

app = Flask(__name__)
CORS(app)

def init_db():
    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS feelings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            child_id TEXT,
            mood TEXT,
            reason TEXT,
            wish TEXT,
            date TEXT
        )
    """)

    conn.commit()
    conn.close()

init_db()

@app.route("/api/feelings", methods=["POST"])
def save_feeling():
    data = request.json

    child_id = data.get("child_id")
    mood = data.get("mood")
    reason = data.get("reason")
    wish = data.get("wish")
    today = str(date.today())

    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()

    cursor.execute(
        "SELECT * FROM feelings WHERE child_id=? AND date=?",
        (child_id, today)
    )

    if cursor.fetchone():
        conn.close()
        return jsonify({"message": "Already submitted today"}), 400

    cursor.execute("""
        INSERT INTO feelings (child_id, mood, reason, wish, date)
        VALUES (?, ?, ?, ?, ?)
    """, (child_id, mood, reason, wish, today))

    conn.commit()
    conn.close()

    return jsonify({"message": "Feeling saved successfully"})

@app.route("/api/feelings/<child_id>", methods=["GET"])
def get_feelings(child_id):
    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()

    cursor.execute(
        "SELECT * FROM feelings WHERE child_id=?",
        (child_id,)
    )

    rows = cursor.fetchall()
    conn.close()

    results = []

    for row in rows:
        results.append({
            "id": row[0],
            "child_id": row[1],
            "mood": row[2],
            "reason": row[3],
            "wish": row[4],
            "date": row[5]
        })

    return jsonify(results)

if __name__ == "__main__":
    app.run(debug=True, port=5000)