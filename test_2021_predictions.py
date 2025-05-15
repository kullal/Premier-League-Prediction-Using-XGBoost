import pandas as pd
import numpy as np
from epl_prediction import EPLPredictor
from datetime import datetime
import matplotlib.pyplot as plt

def main():
    print("Premier League 2021 Season Predictions")
    print("=======================================")
    
    # Initialize and train the model
    predictor = EPLPredictor()
    predictor.load_data()
    predictor.prepare_features()
    predictor.train_models()
    
    # Define some key 2021 season matches to predict
    matches_2021 = [
        {"home": "Arsenal", "away": "Chelsea", "date": "2021-08-22"},
        {"home": "Manchester United", "away": "Leeds", "date": "2021-08-14"},
        {"home": "Tottenham", "away": "Manchester City", "date": "2021-08-15"},
        {"home": "Liverpool", "away": "Burnley", "date": "2021-08-21"},
        {"home": "West Ham", "away": "Leicester", "date": "2021-08-23"}
    ]
    
    # Fix team names that might differ from our dataset
    team_mapping = {
        "Manchester United": "Man United",
        "Manchester City": "Man City",
    }
    
    # Create a results dataframe
    results = []
    
    # Predict each match
    for i, match in enumerate(matches_2021):
        # Map team names if needed
        home_team = team_mapping.get(match["home"], match["home"])
        away_team = team_mapping.get(match["away"], match["away"])
        match_date = datetime.strptime(match["date"], "%Y-%m-%d")
        
        print(f"\nPredicting {i+1}/{len(matches_2021)}: {home_team} vs {away_team} ({match_date.strftime('%Y-%m-%d')})")
        
        # Make prediction
        prediction = predictor.predict_match(home_team, away_team, match_date)
        
        # Display results
        print(f"Match: {prediction['match']}")
        
        print("Result Probabilities:")
        for outcome, prob in prediction['match_result'].items():
            print(f"  {outcome}: {prob}")
        
        print("Goal Timing Probabilities:")
        print(f"  First Half: {prediction['goal_timing']['first_half_goals']}")
        print(f"  Second Half: {prediction['goal_timing']['second_half_goals']}")
        
        # Store results
        result_probs = {k: float(v.strip('%'))/100 for k, v in prediction['match_result'].items()}
        most_likely_result = max(result_probs, key=result_probs.get)
        
        results.append({
            "home_team": home_team,
            "away_team": away_team,
            "date": match_date,
            "predicted_result": most_likely_result,
            "home_win_prob": result_probs.get("Home Team Win", 0),
            "draw_prob": result_probs.get("Draw", 0),
            "away_win_prob": result_probs.get("Away Team Win", 0),
            "first_half_goals_prob": float(prediction['goal_timing']['first_half_goals'].strip('%'))/100,
            "second_half_goals_prob": float(prediction['goal_timing']['second_half_goals'].strip('%'))/100
        })
    
    # Create a dataframe of results
    results_df = pd.DataFrame(results)
    print("\n2021 Season Predictions Summary:")
    print(results_df[["home_team", "away_team", "predicted_result", "home_win_prob", "draw_prob", "away_win_prob"]])
    
    # Visualize all predictions in a single figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Set up bar positions
    bar_width = 0.25
    x = np.arange(len(results_df))
    
    # Plot bars for each outcome
    ax.bar(x - bar_width, results_df["home_win_prob"], bar_width, label="Home Win", color="green")
    ax.bar(x, results_df["draw_prob"], bar_width, label="Draw", color="gray")
    ax.bar(x + bar_width, results_df["away_win_prob"], bar_width, label="Away Win", color="blue")
    
    # Customize plot
    ax.set_xlabel("Match")
    ax.set_ylabel("Probability")
    ax.set_title("Premier League 2021 Match Predictions")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{row.home_team} vs {row.away_team}" for _, row in results_df.iterrows()], rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main() 