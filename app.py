from flask import Flask, request, jsonify
from flask_cors import CORS
from epl_prediction import EPLPredictor
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Initialize predictor
predictor = EPLPredictor()
predictor.load_data()
predictor.prepare_features()
predictor.train_models()

@app.route('/api/teams', methods=['GET'])
def get_teams():
    return jsonify(list(sorted(predictor.teams)))

@app.route('/api/predict', methods=['POST'])
def predict_match():
    data = request.json
    home_team = data.get('home_team')
    away_team = data.get('away_team')
    match_date = data.get('match_date')
    
    if match_date:
        match_date = datetime.strptime(match_date, "%Y-%m-%d")
    
    prediction = predictor.predict_match(home_team, away_team, match_date)
    return jsonify(prediction)

if __name__ == '__main__':
    app.run(debug=True)