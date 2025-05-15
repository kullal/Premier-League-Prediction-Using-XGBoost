# Premier League Prediction System Using XGBoost

This project provides a machine learning system that predicts English Premier League (EPL) match outcomes using XGBoost. The system uses historical data from 2010-2020 seasons to predict:

1. Match outcomes (home win, draw, away win)
2. Goal timing distribution (first half vs second half goals)

## Features

- Predict match results between any two Premier League teams
- Visualize prediction probabilities
- Analyze which features have the most impact on predictions
- Uses XGBoost for high prediction accuracy

## Installation

1. Clone this repository
2. Install required packages:

```bash
pip install -r requirements.txt
```

## Usage

### Command-Line Interface

The easiest way to use the system is with the command-line interface:

```bash
python predict_match.py <home_team> <away_team>
```

For example:
```bash
python predict_match.py Arsenal "Man City"
```

If you run the script without arguments, it will display a list of all available teams:
```bash
python predict_match.py
```

### 2021 Season Predictions

To run predictions on specific matches from the 2021 season:

```bash
python test_2021_predictions.py
```

This script:
- Predicts outcomes for 5 key matches from the beginning of the 2021 season
- Displays individual match predictions and probabilities
- Shows a summary table of all predictions
- Generates a visualization comparing all match predictions

### Running the Example Script

You can also run the example prediction script:

```bash
python epl_prediction.py
```

This will:
- Load the EPL dataset
- Train XGBoost prediction models
- Predict and visualize the outcome of a sample match (Arsenal vs Man City)

### Using in Your Own Code

To predict different matches, you can import the `EPLPredictor` class in your own code:

```python
from epl_prediction import EPLPredictor

# Initialize and train the model
predictor = EPLPredictor()
predictor.load_data()
predictor.prepare_features()
predictor.train_models()

# Predict a custom match
prediction = predictor.predict_match("Liverpool", "Chelsea")

# Display prediction
print(f"Match: {prediction['match']}")
print("\nResult Probabilities:")
for outcome, prob in prediction['match_result'].items():
    print(f"  {outcome}: {prob}")

print("\nGoal Timing Probabilities:")
print(f"  First Half: {prediction['goal_timing']['first_half_goals']}")
print(f"  Second Half: {prediction['goal_timing']['second_half_goals']}")

# Visualize prediction
predictor.visualize_prediction(prediction)
```

## Dataset

The system uses the EPL dataset from seasons 2010-2020, which includes:
- Match results
- Team statistics (shots, corners, fouls, etc.)
- Goal timing information (halftime vs full-time)

## How It Works

The prediction system uses two XGBoost models:
1. A multi-class classifier for predicting match outcomes (home win, draw, away win)
2. A binary classifier for predicting whether more goals will be scored in the first or second half

Features used for prediction include:
- Team identities
- Match timing (month, day of week)
- Historical team performance statistics 