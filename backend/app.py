from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from modules.predict_future import predict_future_match, get_teams_and_referees
from modules.predict_history import predict_history_matchup

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "http://localhost:5173"}})

# Endpoint untuk mendapatkan daftar tim dan wasit
@app.route('/api/teams', methods=['GET'])
def teams_endpoint():
    teams, referees = get_teams_and_referees()
    if teams and referees:
        return jsonify({"teams": teams, "referees": referees})
    return jsonify({"error": "Could not load teams and referees"}), 500

# Endpoint untuk prediksi pertandingan baru
@app.route('/api/predict/future', methods=['POST'])
def predict_future():
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid input"}), 400

    try:
        home_team = data['home_team']
        away_team = data['away_team']
        match_date = datetime.strptime(data['match_date'], '%Y-%m-%d')
        referee = data.get('referee', 'Michael Oliver')
        odds = data.get('odds') # Bisa None

        result = predict_future_match(home_team, away_team, match_date, referee, odds)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Endpoint untuk prediksi berdasarkan riwayat
@app.route('/api/predict/history', methods=['GET'])
def predict_history():
    home_team = request.args.get('home_team')
    away_team = request.args.get('away_team')

    if not home_team or not away_team:
        return jsonify({"error": "home_team and away_team parameters are required"}), 400

    try:
        result = predict_history_matchup(home_team, away_team)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)