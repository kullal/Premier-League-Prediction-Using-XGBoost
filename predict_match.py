import sys
from epl_prediction import EPLPredictor
from datetime import datetime

def main():
    # Check if teams are provided as command-line arguments
    if len(sys.argv) < 3:
        print("Usage: python predict_match.py <home_team> <away_team>")
        print("Example: python predict_match.py Arsenal 'Man City'")
        
        # List available teams
        predictor = EPLPredictor()
        predictor.load_data()
        print("\nAvailable teams:")
        for team in sorted(predictor.teams):
            print(f"- {team}")
        return
    
    # Get team names from command-line arguments
    home_team = sys.argv[1]
    away_team = sys.argv[2]
    
    # Initialize the predictor
    print(f"Initializing Premier League prediction system...")
    predictor = EPLPredictor()
    predictor.load_data()
    predictor.prepare_features()
    predictor.train_models()
    
    # Make prediction
    print(f"Predicting match outcome: {home_team} vs {away_team}")
    prediction = predictor.predict_match(home_team, away_team)
    
    # Display results
    print("\n========== MATCH PREDICTION ==========")
    print(f"Match: {prediction['match']}")
    
    print("\nResult Probabilities:")
    for outcome, prob in prediction['match_result'].items():
        print(f"  {outcome}: {prob}")
    
    print("\nGoal Timing Probabilities:")
    print(f"  First Half: {prediction['goal_timing']['first_half_goals']}")
    print(f"  Second Half: {prediction['goal_timing']['second_half_goals']}")
    
    # Visualize prediction
    print("\nGenerating visualization...")
    predictor.visualize_prediction(prediction)
    
    # Show feature importance
    print("\nShowing feature importance...")
    predictor.feature_importance()

if __name__ == "__main__":
    main() 