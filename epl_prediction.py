import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

class EPLPredictor:
    def __init__(self): 
        self.teams = set()
        self.matches_df = None
        self.features_df = None
        self.result_model = None
        self.goal_timing_model = None
        self.label_encoder = LabelEncoder()
        
    def load_data(self):
        """Load and prepare the EPL match data from 2020-2024"""
        print("Loading EPL match data...")
        
        # Load the new dataset (2020-2024)
        self.matches_df = pd.read_csv('Dataset EPL 2020-2024/matches.csv')
        
        # Clean and prepare the data
        self.matches_df = self.matches_df.dropna(subset=['result', 'opponent'])
        
        # Extract team names
        self.teams = set(self.matches_df['team'].unique()) | set(self.matches_df['opponent'].unique())
        print(f"Loaded data for {len(self.teams)} teams")
        
    def prepare_features(self):
        """Prepare features for the prediction models"""
        print("Preparing features...")
        
        # Create features dataframe
        features = []
        
        # Group by team to calculate team-specific stats
        team_stats = self.matches_df.groupby('team')
        
        for team in self.teams:
            # Get team's matches
            team_matches = self.matches_df[self.matches_df['team'] == team]
            
            # Calculate team performance metrics
            win_rate = team_matches[team_matches['result'] == 'W'].shape[0] / max(1, team_matches.shape[0])
            home_win_rate = team_matches[(team_matches['venue'] == 'Home') & (team_matches['result'] == 'W')].shape[0] / max(1, team_matches[team_matches['venue'] == 'Home'].shape[0])
            away_win_rate = team_matches[(team_matches['venue'] == 'Away') & (team_matches['result'] == 'W')].shape[0] / max(1, team_matches[team_matches['venue'] == 'Away'].shape[0])
            
            # Calculate offensive and defensive stats
            avg_goals_scored = team_matches['gf'].mean()
            avg_goals_conceded = team_matches['ga'].mean()
            avg_shots = team_matches['sh'].mean() if 'sh' in team_matches.columns else 0
            avg_shots_on_target = team_matches['sot'].mean() if 'sot' in team_matches.columns else 0
            
            # Calculate form metrics (last 5 matches)
            recent_matches = team_matches.sort_values('date', ascending=False).head(5)
            recent_win_rate = recent_matches[recent_matches['result'] == 'W'].shape[0] / max(1, recent_matches.shape[0])
            
            # Add possession stats
            avg_possession = team_matches['poss'].mean() if 'poss' in team_matches.columns else 50
            
            # Add team features
            features.append({
                'team': team,
                'win_rate': win_rate,
                'home_win_rate': home_win_rate,
                'away_win_rate': away_win_rate,
                'avg_goals_scored': avg_goals_scored,
                'avg_goals_conceded': avg_goals_conceded,
                'avg_shots': avg_shots,
                'avg_shots_on_target': avg_shots_on_target,
                'avg_possession': avg_possession,
                'recent_form': recent_win_rate
            })
        
        self.features_df = pd.DataFrame(features)
        print(f"Prepared features for {self.features_df.shape[0]} teams")
        
    def _prepare_match_features(self, home_team, away_team, match_date=None):
        """Prepare features for a specific match"""
        if home_team not in self.teams or away_team not in self.teams:
            raise ValueError(f"Team not found. Available teams: {sorted(self.teams)}")
        
        # Get team features
        home_features = self.features_df[self.features_df['team'] == home_team].iloc[0]
        away_features = self.features_df[self.features_df['team'] == away_team].iloc[0]
        
        # Create match features
        match_features = {
            'home_win_rate': home_features['win_rate'],
            'away_win_rate': away_features['win_rate'],
            'home_team_home_win_rate': home_features['home_win_rate'],
            'away_team_away_win_rate': away_features['away_win_rate'],
            'home_team_avg_goals': home_features['avg_goals_scored'],
            'away_team_avg_goals': away_features['avg_goals_scored'],
            'home_team_avg_conceded': home_features['avg_goals_conceded'],
            'away_team_avg_conceded': away_features['avg_goals_conceded'],
            'home_team_avg_shots': home_features['avg_shots'],
            'away_team_avg_shots': away_features['avg_shots'],
            'home_team_avg_shots_on_target': home_features['avg_shots_on_target'],
            'away_team_avg_shots_on_target': away_features['avg_shots_on_target'],
            'home_team_avg_possession': home_features['avg_possession'],
            'away_team_avg_possession': away_features['avg_possession'],
            'home_team_recent_form': home_features['recent_form'],
            'away_team_recent_form': away_features['recent_form'],
            'goal_diff_home': home_features['avg_goals_scored'] - away_features['avg_goals_conceded'],
            'goal_diff_away': away_features['avg_goals_scored'] - home_features['avg_goals_conceded'],
            'shot_efficiency_home': home_features['avg_shots_on_target'] / max(1, home_features['avg_shots']),
            'shot_efficiency_away': away_features['avg_shots_on_target'] / max(1, away_features['avg_shots'])
        }
        
        # Add head-to-head features if we have match data
        if match_date:
            match_date = pd.to_datetime(match_date)
            
            # Get previous matches between these teams
            h2h_matches = self.matches_df[
                ((self.matches_df['team'] == home_team) & (self.matches_df['opponent'] == away_team)) |
                ((self.matches_df['team'] == away_team) & (self.matches_df['opponent'] == home_team))
            ]
            
            # Filter to only include matches before the prediction date
            h2h_matches = h2h_matches[pd.to_datetime(h2h_matches['date']) < match_date]
            
            if not h2h_matches.empty:
                # Calculate head-to-head stats
                home_wins = h2h_matches[
                    (h2h_matches['team'] == home_team) & 
                    (h2h_matches['venue'] == 'Home') & 
                    (h2h_matches['result'] == 'W')
                ].shape[0]
                
                away_wins = h2h_matches[
                    (h2h_matches['team'] == away_team) & 
                    (h2h_matches['venue'] == 'Home') & 
                    (h2h_matches['result'] == 'W')
                ].shape[0]
                
                draws = h2h_matches.shape[0] - home_wins - away_wins
                
                match_features['h2h_home_win_rate'] = home_wins / max(1, h2h_matches.shape[0])
                match_features['h2h_away_win_rate'] = away_wins / max(1, h2h_matches.shape[0])
                match_features['h2h_draw_rate'] = draws / max(1, h2h_matches.shape[0])
            else:
                # No head-to-head data available
                match_features['h2h_home_win_rate'] = 0.33
                match_features['h2h_away_win_rate'] = 0.33
                match_features['h2h_draw_rate'] = 0.34
        
        return pd.DataFrame([match_features])
    
    def train_models(self):
        """Train the prediction models"""
        print("Training prediction models...")
        
        # Prepare training data
        X_train = []
        y_result = []
        y_goal_timing = []
        
        # Create training examples from historical matches
        for _, match in self.matches_df.iterrows():
            home_team = match['team'] if match['venue'] == 'Home' else match['opponent']
            away_team = match['opponent'] if match['venue'] == 'Home' else match['team']
            
            try:
                # Prepare features for this match
                match_features = self._prepare_match_features(home_team, away_team)
                
                # Determine match result (from home team perspective)
                if match['venue'] == 'Home':
                    result = 'home_win' if match['result'] == 'W' else ('draw' if match['result'] == 'D' else 'away_win')
                else:
                    result = 'away_win' if match['result'] == 'W' else ('draw' if match['result'] == 'D' else 'home_win')
                
                # Add to training data
                X_train.append(match_features.iloc[0].values)
                y_result.append(result)
                
                # Determine goal timing (using first half/second half goals if available)
                # Jika data waktu gol tersedia, gunakan itu, jika tidak gunakan pendekatan probabilistik
                if 'hg' in match and 'ag' in match:  # Jika ada data gol babak pertama
                    first_half_goals = match['hg']
                    total_goals = match['gf'] + match['ga']
                    y_goal_timing.append(0 if first_half_goals > (total_goals / 2) else 1)
                else:
                    # Gunakan pendekatan probabilistik berdasarkan pola umum
                    y_goal_timing.append(np.random.choice([0, 1], p=[0.4, 0.6]))  # Lebih banyak gol di babak kedua
                
            except Exception as e:
                # Skip matches with missing data
                continue
        
        # Convert to numpy arrays
        X_train = np.array(X_train)
        
        # Encode result labels
        self.label_encoder.fit(y_result)
        y_result_encoded = self.label_encoder.transform(y_result)
        
        # Train result prediction model with improved parameters
        self.result_model = xgb.XGBClassifier(
            objective='multi:softprob',
            num_class=3,
            learning_rate=0.03,
            max_depth=7,
            n_estimators=200,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1,
            scale_pos_weight=1,
            seed=42
        )
        self.result_model.fit(X_train, y_result_encoded)
        
        # Train goal timing model with improved parameters
        self.goal_timing_model = xgb.XGBClassifier(
            objective='binary:logistic',
            learning_rate=0.03,
            max_depth=5,
            n_estimators=100,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=2,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1,
            scale_pos_weight=1,
            seed=42
        )
        self.goal_timing_model.fit(X_train, y_goal_timing)
        
        print("Models trained successfully")
    
    def predict_match(self, home_team, away_team, match_date=None):
        """Predict the outcome of a match between two teams"""
        # Prepare features for the match
        match_features = self._prepare_match_features(home_team, away_team, match_date)
        
        # Predict match result
        result_probs = self.result_model.predict_proba(match_features)[0]
        result_classes = self.label_encoder.classes_
        
        # Format result probabilities
        result_dict = {}
        for i, result in enumerate(result_classes):
            if result == 'home_win':
                result_dict["Home Team Win"] = f"{result_probs[i]*100:.1f}%"
            elif result == 'away_win':
                result_dict["Away Team Win"] = f"{result_probs[i]*100:.1f}%"
            else:
                result_dict["Draw"] = f"{result_probs[i]*100:.1f}%"
        
        # Predict goal timing
        goal_timing_prob = self.goal_timing_model.predict_proba(match_features)[0]
        
        # Create prediction result
        prediction = {
            'match': f"{home_team} vs {away_team}",
            'match_result': result_dict,
            'goal_timing': {
                'first_half_goals': f"{goal_timing_prob[0]*100:.1f}%",
                'second_half_goals': f"{goal_timing_prob[1]*100:.1f}%"
            }
        }
        
        return prediction
    
    def visualize_prediction(self, prediction):
        """Visualize the match prediction"""
        # Extract probabilities
        home_win_prob = float(prediction['match_result'].get('Home Team Win', '0%').strip('%')) / 100
        draw_prob = float(prediction['match_result'].get('Draw', '0%').strip('%')) / 100
        away_win_prob = float(prediction['match_result'].get('Away Team Win', '0%').strip('%')) / 100
        
        # Create bar chart for match result
        plt.figure(figsize=(10, 6))
        
        # Plot match result probabilities
        plt.subplot(1, 2, 1)
        results = ['Home Win', 'Draw', 'Away Win']
        probs = [home_win_prob, draw_prob, away_win_prob]
        colors = ['green', 'gray', 'blue']
        
        plt.bar(results, probs, color=colors)
        plt.title(f'Match Result Prediction\n{prediction["match"]}')
        plt.ylabel('Probability')
        plt.ylim(0, 1)
        
        # Add probability labels
        for i, prob in enumerate(probs):
            plt.text(i, prob + 0.02, f'{prob:.1%}', ha='center')
        
        # Plot goal timing probabilities
        plt.subplot(1, 2, 2)
        first_half_prob = float(prediction['goal_timing']['first_half_goals'].strip('%')) / 100
        second_half_prob = float(prediction['goal_timing']['second_half_goals'].strip('%')) / 100
        
        plt.bar(['First Half', 'Second Half'], [first_half_prob, second_half_prob], color=['orange', 'purple'])
        plt.title('Goal Timing Prediction')
        plt.ylabel('Probability')
        plt.ylim(0, 1)
        
        # Add probability labels
        plt.text(0, first_half_prob + 0.02, f'{first_half_prob:.1%}', ha='center')
        plt.text(1, second_half_prob + 0.02, f'{second_half_prob:.1%}', ha='center')
        
        plt.tight_layout()
        plt.show()
    
    def feature_importance(self):
        """Display feature importance for the result prediction model"""
        if self.result_model is None:
            print("Model not trained yet")
            return
        
        # Get feature importance
        importance = self.result_model.feature_importances_
        
        # Get feature names
        feature_names = list(self._prepare_match_features(list(self.teams)[0], list(self.teams)[1]).columns)
        
        # Create DataFrame for visualization
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance
        }).sort_values('Importance', ascending=False)
        
        # Plot feature importance
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=importance_df)
        plt.title('Feature Importance for Match Result Prediction')
        plt.tight_layout()
        plt.show()

# Example usage
if __name__ == "__main__":
    # Initialize predictor
    predictor = EPLPredictor()
    predictor.load_data()
    predictor.prepare_features()
    predictor.train_models()
    
    # Evaluate model performance
    def evaluate_model(self, test_size=0.2):
        """Evaluate model performance using train-test split"""
        if self.result_model is None:
            print("Model not trained yet")
            return
        
        # Prepare data
        X = []
        y_result = []
        
        for _, match in self.matches_df.iterrows():
            home_team = match['team'] if match['venue'] == 'Home' else match['opponent']
            away_team = match['opponent'] if match['venue'] == 'Home' else match['team']
            
            try:
                # Prepare features
                match_features = self._prepare_match_features(home_team, away_team)
                
                # Determine result
                if match['venue'] == 'Home':
                    result = 'home_win' if match['result'] == 'W' else ('draw' if match['result'] == 'D' else 'away_win')
                else:
                    result = 'away_win' if match['result'] == 'W' else ('draw' if match['result'] == 'D' else 'home_win')
                
                X.append(match_features.iloc[0].values)
                y_result.append(result)
                
            except Exception as e:
                continue
        
        # Split data
        X = np.array(X)
        y_result_encoded = self.label_encoder.transform(y_result)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_result_encoded, test_size=test_size, random_state=42
        )
        
        # Train model on training data
        model = xgb.XGBClassifier(
            objective='multi:softprob',
            num_class=3,
            learning_rate=0.03,
            max_depth=7,
            n_estimators=200,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1,
            scale_pos_weight=1,
            seed=42
        )
        model.fit(X_train, y_train)
        
        # Evaluate on test data
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
        
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Model Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=self.label_encoder.classes_))
        
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=self.label_encoder.classes_,
                    yticklabels=self.label_encoder.classes_)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.show()
        
        return accuracy
    
    # Make a sample prediction
    home_team = "Arsenal"
    away_team = "Man City"
    prediction = predictor.predict_match(home_team, away_team)
    
    # Display prediction
    print(f"\nPrediction for {home_team} vs {away_team}:")
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