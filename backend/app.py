import os
import sys
from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

app = Flask(__name__)

allowed_origins = os.getenv("ALLOWED_ORIGINS", "*")
if allowed_origins == "*" or not allowed_origins:
    CORS(app)
else:
    # allow only frontend origin for API routes
    CORS(app, resources={r"/api/*": {"origins": allowed_origins}})

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

@app.route('/api/referees', methods=['GET'])
def referees_endpoint():
    """
    Return only referees list used by frontend.
    Uses modules.predict_future.get_teams_and_referees to load data.
    """
    try:
        teams, referees = get_teams_and_referees()
        if referees:
            referees_list = [str(r) for r in referees]
            print(f"Loaded {len(referees_list)} referees") 
            return jsonify({"referees": referees_list})
        print("No referees loaded")
        return jsonify({"error": "Could not load referees"}), 500
    except Exception as e:
        print("Error in /api/referees:", str(e))
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)