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
    
    # Check if date is provided as an optional argument
    match_date = None
    if len(sys.argv) > 3:
        try:
            match_date = datetime.strptime(sys.argv[3], "%Y-%m-%d")
            print(f"Using prediction date: {match_date.strftime('%Y-%m-%d')}")
        except ValueError:
            print(f"Invalid date format. Using current date instead.")
    
    # Initialize the predictor
    print(f"Initializing Premier League prediction system...")
    predictor = EPLPredictor()
    predictor.load_data()
    predictor.prepare_features()
    predictor.train_models()
    
    # Make prediction
    print(f"Predicting match outcome: {home_team} vs {away_team}")
    prediction = predictor.predict_match(home_team, away_team, match_date)
    
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
    
    # Optional: evaluate model performance
    if len(sys.argv) > 4 and sys.argv[4].lower() == 'evaluate':
        print("\nEvaluating model performance...")
        predictor.evaluate_model()

if __name__ == "__main__":
    main()