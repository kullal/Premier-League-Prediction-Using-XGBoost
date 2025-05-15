import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

class EPLPredictor:
    def __init__(self):
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model_result = None
        self.model_goals_time = None
        self.team_encoder = None
        self.teams = None
        self.result_mapping = {'HomeWin': 'Home Team Win', 'Draw': 'Draw', 'AwayWin': 'Away Team Win'}
        
    def load_data(self, file_path="Dataset EPL 2010-2020/epl-allseasons-matchstats.csv"):
        """Load the EPL dataset from CSV"""
        self.data = pd.read_csv(file_path)
        print(f"Loaded data with {self.data.shape[0]} matches from seasons {self.data['Season'].unique()}")
        
        # Convert date string to datetime
        self.data['Date'] = pd.to_datetime(self.data['Date'])
        
        # Get unique teams
        home_teams = self.data['HomeTeam'].unique()
        away_teams = self.data['AwayTeam'].unique()
        self.teams = np.unique(np.concatenate([home_teams, away_teams]))
        
        # Encode team names
        self.team_encoder = LabelEncoder()
        self.team_encoder.fit(self.teams)
        
        return self
    
    def prepare_features(self):
        """Prepare features for the model"""
        # Create features for match outcome prediction
        df = self.data.copy()
        
        # Encode teams
        df['HomeTeam_encoded'] = self.team_encoder.transform(df['HomeTeam'])
        df['AwayTeam_encoded'] = self.team_encoder.transform(df['AwayTeam'])
        
        # Extract month and day of week as features
        df['Month'] = df['Date'].dt.month
        df['DayOfWeek'] = df['Date'].dt.dayofweek
        
        # Create features for the model
        features = [
            'HomeTeam_encoded', 'AwayTeam_encoded', 'Month', 'DayOfWeek',
            'HomeShots', 'HomeShotsOnTarget', 'HomeCorners', 'HomeFouls',
            'HomeYellowCards', 'AwayShots', 'AwayShotsOnTarget', 'AwayCorners', 
            'AwayFouls', 'AwayYellowCards'
        ]
        
        # Add head-to-head features (could be expanded with your head-to-head data)
        # For this example, we'll use basic features
        
        # Split data
        X = df[features]
        y_result = df['FullTime']  # Match result
        
        # For goal time prediction, create a simple binary feature
        # If more goals were scored in first half (1) or second half (0)
        df['MoreGoalsFirstHalf'] = ((df['HomeGoalsHalftime'] + df['AwayGoalsHalftime']) > 
                                    (df['HomeGoals'] + df['AwayGoals'] - 
                                     df['HomeGoalsHalftime'] - df['AwayGoalsHalftime'])).astype(int)
        
        y_goals_time = df['MoreGoalsFirstHalf']
        
        # Split into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y_result, test_size=0.2, random_state=42)
        
        # Split data for goals time prediction
        self.X_train_goals, self.X_test_goals, self.y_train_goals, self.y_test_goals = train_test_split(
            X, y_goals_time, test_size=0.2, random_state=42)
        
        return self
    
    def train_models(self):
        """Train the XGBoost models"""
        # Model for match result prediction
        self.model_result = xgb.XGBClassifier(
            objective='multi:softprob',
            num_class=3,
            learning_rate=0.05,
            max_depth=6,
            n_estimators=100,
            subsample=0.8,
            random_state=42
        )
        
        # Convert categorical output to numeric for XGBoost
        result_encoder = LabelEncoder()
        y_train_encoded = result_encoder.fit_transform(self.y_train)
        
        self.model_result.fit(
            self.X_train, 
            y_train_encoded,
            eval_set=[(self.X_train, y_train_encoded)],
            verbose=False
        )
        
        # Model for goal time prediction
        self.model_goals_time = xgb.XGBClassifier(
            objective='binary:logistic',
            learning_rate=0.05,
            max_depth=6,
            n_estimators=100,
            subsample=0.8,
            random_state=42
        )
        
        self.model_goals_time.fit(
            self.X_train_goals, 
            self.y_train_goals,
            eval_set=[(self.X_train_goals, self.y_train_goals)],
            verbose=False
        )
        
        print("Models trained successfully!")
        return self
    
    def get_team_stats(self, team, is_home=True):
        """Get average statistics for a team"""
        if is_home:
            team_data = self.data[self.data['HomeTeam'] == team]
            shots = team_data['HomeShots'].mean()
            shots_on_target = team_data['HomeShotsOnTarget'].mean()
            corners = team_data['HomeCorners'].mean()
            fouls = team_data['HomeFouls'].mean()
            yellow_cards = team_data['HomeYellowCards'].mean()
        else:
            team_data = self.data[self.data['AwayTeam'] == team]
            shots = team_data['AwayShots'].mean()
            shots_on_target = team_data['AwayShotsOnTarget'].mean()
            corners = team_data['AwayCorners'].mean()
            fouls = team_data['AwayFouls'].mean()
            yellow_cards = team_data['AwayYellowCards'].mean()
            
        return {
            'shots': shots,
            'shots_on_target': shots_on_target,
            'corners': corners,
            'fouls': fouls,
            'yellow_cards': yellow_cards
        }
    
    def predict_match(self, home_team, away_team, date=None):
        """Predict the outcome of a match between home_team and away_team"""
        if date is None:
            date = datetime.now()
        
        # Check if teams exist in our data
        if home_team not in self.teams or away_team not in self.teams:
            missing = []
            if home_team not in self.teams:
                missing.append(home_team)
            if away_team not in self.teams:
                missing.append(away_team)
            print(f"Warning: {', '.join(missing)} not found in training data. Using similar team statistics.")
        
        # Get team encodings
        home_encoded = self.team_encoder.transform([home_team])[0] if home_team in self.teams else 0
        away_encoded = self.team_encoder.transform([away_team])[0] if away_team in self.teams else 0
        
        # Get average stats for both teams
        home_stats = self.get_team_stats(home_team, is_home=True) if home_team in self.teams else {
            'shots': 13, 'shots_on_target': 7, 'corners': 6, 'fouls': 12, 'yellow_cards': 2
        }
        
        away_stats = self.get_team_stats(away_team, is_home=False) if away_team in self.teams else {
            'shots': 10, 'shots_on_target': 5, 'corners': 4, 'fouls': 12, 'yellow_cards': 2
        }
        
        # Prepare input for prediction
        X_pred = pd.DataFrame({
            'HomeTeam_encoded': [home_encoded],
            'AwayTeam_encoded': [away_encoded],
            'Month': [date.month],
            'DayOfWeek': [date.weekday()],
            'HomeShots': [home_stats['shots']],
            'HomeShotsOnTarget': [home_stats['shots_on_target']],
            'HomeCorners': [home_stats['corners']],
            'HomeFouls': [home_stats['fouls']],
            'HomeYellowCards': [home_stats['yellow_cards']],
            'AwayShots': [away_stats['shots']],
            'AwayShotsOnTarget': [away_stats['shots_on_target']],
            'AwayCorners': [away_stats['corners']],
            'AwayFouls': [away_stats['fouls']],
            'AwayYellowCards': [away_stats['yellow_cards']]
        })
        
        # Predict match result
        result_probs = self.model_result.predict_proba(X_pred)[0]
        result_labels = ['HomeWin', 'Draw', 'AwayWin']  # Order may vary based on encoder
        result_dict = {result_labels[i]: result_probs[i] for i in range(len(result_labels))}
        
        # Predict goal timing
        goals_time_prob = self.model_goals_time.predict_proba(X_pred)[0]
        
        # Organize results
        prediction = {
            'match': f"{home_team} vs {away_team}",
            'match_result': {
                self.result_mapping[label]: f"{prob*100:.1f}%" 
                for label, prob in result_dict.items()
            },
            'goal_timing': {
                'first_half_goals': f"{goals_time_prob[1]*100:.1f}%",
                'second_half_goals': f"{goals_time_prob[0]*100:.1f}%"
            }
        }
        
        return prediction
    
    def visualize_prediction(self, prediction):
        """Visualize prediction results"""
        # Extract data
        match = prediction['match']
        
        # Match result probabilities
        result_labels = list(prediction['match_result'].keys())
        result_probs = [float(p.strip('%'))/100 for p in prediction['match_result'].values()]
        
        # Goal timing probabilities
        goal_labels = ['First Half', 'Second Half']
        goal_probs = [
            float(prediction['goal_timing']['first_half_goals'].strip('%'))/100,
            float(prediction['goal_timing']['second_half_goals'].strip('%'))/100,
        ]
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot match result probabilities
        ax1.bar(result_labels, result_probs, color=['green', 'gray', 'blue'])
        ax1.set_title(f'Match Result Prediction: {match}')
        ax1.set_ylabel('Probability')
        ax1.set_ylim(0, 1)
        for i, v in enumerate(result_probs):
            ax1.text(i, v + 0.02, f'{v:.1%}', ha='center')
        
        # Plot goal timing probabilities
        ax2.bar(goal_labels, goal_probs, color=['orange', 'purple'])
        ax2.set_title('Goal Timing Prediction')
        ax2.set_ylabel('Probability')
        ax2.set_ylim(0, 1)
        for i, v in enumerate(goal_probs):
            ax2.text(i, v + 0.02, f'{v:.1%}', ha='center')
        
        plt.tight_layout()
        plt.show()
        
    def feature_importance(self):
        """Show feature importance for the result prediction model"""
        fig, ax = plt.subplots(figsize=(10, 6))
        xgb.plot_importance(self.model_result, ax=ax)
        plt.title('Feature Importance for Match Result Prediction')
        plt.tight_layout()
        plt.show()


# Example usage
if __name__ == "__main__":
    predictor = EPLPredictor()
    predictor.load_data()
    predictor.prepare_features()
    predictor.train_models()
    
    # Predict a match
    prediction = predictor.predict_match("Arsenal", "Man City")
    print("\nMatch Prediction:")
    print(f"Match: {prediction['match']}")
    print("\nResult Probabilities:")
    for outcome, prob in prediction['match_result'].items():
        print(f"  {outcome}: {prob}")
    
    print("\nGoal Timing Probabilities:")
    print(f"  First Half: {prediction['goal_timing']['first_half_goals']}")
    print(f"  Second Half: {prediction['goal_timing']['second_half_goals']}")
    
    # Visualize prediction
    predictor.visualize_prediction(prediction)
    
    # Show feature importance
    predictor.feature_importance() 