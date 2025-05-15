import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

class EPLPredictor:
    def __init__(self): 
        self.teams = set()
        self.matches_df = None
        self.features_df = None
        self.result_model = None
        self.goal_timing_model = None
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        
    def load_data(self):
        """Load and prepare the EPL match data"""
        print("Loading EPL match data...")
        
        try:
            # Coba load dataset baru (2020-2024)
            self.matches_df = pd.read_csv('Dataset EPL 2020-2024/matches.csv')
            print("Loaded dataset from 2020-2024")
        except FileNotFoundError:
            # Jika tidak ada, buat folder dan berikan pesan
            import os
            os.makedirs('Dataset EPL 2020-2024', exist_ok=True)
            print("Dataset folder created. Please add match data to 'Dataset EPL 2020-2024/matches.csv'")
            # Buat dataframe kosong dengan kolom yang diperlukan
            self.matches_df = pd.DataFrame(columns=['date', 'team', 'opponent', 'venue', 'result', 'gf', 'ga', 'sh', 'sot', 'poss'])
        
        # Clean and prepare the data
        if not self.matches_df.empty:
            self.matches_df = self.matches_df.dropna(subset=['result', 'opponent'])
            
            # Convert date to datetime
            if 'date' in self.matches_df.columns:
                self.matches_df['date'] = pd.to_datetime(self.matches_df['date'])
            
            # Extract team names
            self.teams = set(self.matches_df['team'].unique()) | set(self.matches_df['opponent'].unique())
            print(f"Loaded data for {len(self.teams)} teams")
        else:
            print("No data available. Please add match data.")
        
    def prepare_features(self):
        """Prepare features for the prediction models with enhanced metrics"""
        print("Preparing features...")
        
        # Create features dataframe
        features = []
        
        # Group by team to calculate team-specific stats
        if not self.matches_df.empty:
            for team in self.teams:
                # Get team's matches
                team_matches = self.matches_df[self.matches_df['team'] == team]
                
                # Calculate team performance metrics
                win_rate = team_matches[team_matches['result'] == 'W'].shape[0] / max(1, team_matches.shape[0])
                draw_rate = team_matches[team_matches['result'] == 'D'].shape[0] / max(1, team_matches.shape[0])
                loss_rate = team_matches[team_matches['result'] == 'L'].shape[0] / max(1, team_matches.shape[0])
                
                home_matches = team_matches[team_matches['venue'] == 'Home']
                away_matches = team_matches[team_matches['venue'] == 'Away']
                
                home_win_rate = home_matches[home_matches['result'] == 'W'].shape[0] / max(1, home_matches.shape[0])
                away_win_rate = away_matches[away_matches['result'] == 'W'].shape[0] / max(1, away_matches.shape[0])
                
                # Calculate offensive and defensive stats
                avg_goals_scored = team_matches['gf'].mean()
                avg_goals_conceded = team_matches['ga'].mean()
                home_avg_goals = home_matches['gf'].mean() if not home_matches.empty else 0
                away_avg_goals = away_matches['gf'].mean() if not away_matches.empty else 0
                
                # Advanced metrics
                goal_diff = avg_goals_scored - avg_goals_conceded
                points_per_game = (win_rate * 3 + draw_rate * 1)
                
                # Shot metrics
                avg_shots = team_matches['sh'].mean() if 'sh' in team_matches.columns else 0
                avg_shots_on_target = team_matches['sot'].mean() if 'sot' in team_matches.columns else 0
                shot_accuracy = avg_shots_on_target / max(1, avg_shots)
                shot_conversion = avg_goals_scored / max(1, avg_shots_on_target)
                
                # Calculate form metrics (last 5 matches)
                recent_matches = team_matches.sort_values('date', ascending=False).head(5)
                recent_win_rate = recent_matches[recent_matches['result'] == 'W'].shape[0] / max(1, recent_matches.shape[0])
                recent_points = (recent_matches[recent_matches['result'] == 'W'].shape[0] * 3 + 
                                recent_matches[recent_matches['result'] == 'D'].shape[0]) / max(1, recent_matches.shape[0])
                
                # Add possession stats
                avg_possession = team_matches['poss'].mean() if 'poss' in team_matches.columns else 50
                
                # Add team features
                features.append({
                    'team': team,
                    'win_rate': win_rate,
                    'draw_rate': draw_rate,
                    'loss_rate': loss_rate,
                    'home_win_rate': home_win_rate,
                    'away_win_rate': away_win_rate,
                    'avg_goals_scored': avg_goals_scored,
                    'avg_goals_conceded': avg_goals_conceded,
                    'home_avg_goals': home_avg_goals,
                    'away_avg_goals': away_avg_goals,
                    'goal_diff': goal_diff,
                    'points_per_game': points_per_game,
                    'avg_shots': avg_shots,
                    'avg_shots_on_target': avg_shots_on_target,
                    'shot_accuracy': shot_accuracy,
                    'shot_conversion': shot_conversion,
                    'avg_possession': avg_possession,
                    'recent_form': recent_win_rate,
                    'recent_points': recent_points
                })
            
            self.features_df = pd.DataFrame(features)
            print(f"Prepared features for {self.features_df.shape[0]} teams")
        else:
            print("No data available for feature preparation")
            self.features_df = pd.DataFrame(columns=['team'])
        
    def _prepare_match_features(self, home_team, away_team, match_date=None):
        """Prepare features for a specific match with enhanced metrics"""
        if home_team not in self.teams or away_team not in self.teams:
            raise ValueError(f"Team not found. Available teams: {sorted(self.teams)}")
        
        # Get team features
        home_features = self.features_df[self.features_df['team'] == home_team].iloc[0]
        away_features = self.features_df[self.features_df['team'] == away_team].iloc[0]
        
        # Create match features
        match_features = {
            'home_win_rate': home_features['win_rate'],
            'away_win_rate': away_features['win_rate'],
            'home_draw_rate': home_features['draw_rate'],
            'away_draw_rate': away_features['draw_rate'],
            'home_team_home_win_rate': home_features['home_win_rate'],
            'away_team_away_win_rate': away_features['away_win_rate'],
            'home_team_avg_goals': home_features['avg_goals_scored'],
            'away_team_avg_goals': away_features['avg_goals_scored'],
            'home_team_home_avg_goals': home_features['home_avg_goals'],
            'away_team_away_avg_goals': away_features['away_avg_goals'],
            'home_team_avg_conceded': home_features['avg_goals_conceded'],
            'away_team_avg_conceded': away_features['avg_goals_conceded'],
            'home_team_goal_diff': home_features['goal_diff'],
            'away_team_goal_diff': away_features['goal_diff'],
            'home_team_points_per_game': home_features['points_per_game'],
            'away_team_points_per_game': away_features['points_per_game'],
            'home_team_avg_shots': home_features['avg_shots'],
            'away_team_avg_shots': away_features['avg_shots'],
            'home_team_avg_shots_on_target': home_features['avg_shots_on_target'],
            'away_team_avg_shots_on_target': away_features['avg_shots_on_target'],
            'home_team_shot_accuracy': home_features['shot_accuracy'],
            'away_team_shot_accuracy': away_features['shot_accuracy'],
            'home_team_shot_conversion': home_features['shot_conversion'],
            'away_team_shot_conversion': away_features['shot_conversion'],
            'home_team_avg_possession': home_features['avg_possession'],
            'away_team_avg_possession': away_features['avg_possession'],
            'home_team_recent_form': home_features['recent_form'],
            'away_team_recent_form': away_features['recent_form'],
            'home_team_recent_points': home_features['recent_points'],
            'away_team_recent_points': away_features['recent_points'],
            'form_diff': home_features['recent_form'] - away_features['recent_form'],
            'points_diff': home_features['points_per_game'] - away_features['points_per_game'],
            'goal_diff_home': home_features['avg_goals_scored'] - away_features['avg_goals_conceded'],
            'goal_diff_away': away_features['avg_goals_scored'] - home_features['avg_goals_conceded'],
            'possession_diff': home_features['avg_possession'] - away_features['avg_possession']
        }
        
        # Add head-to-head features if we have match data
        if match_date and not self.matches_df.empty:
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
                
                # Calculate average goals in h2h matches
                h2h_home_goals = h2h_matches[h2h_matches['team'] == home_team]['gf'].mean()
                h2h_away_goals = h2h_matches[h2h_matches['team'] == away_team]['gf'].mean()
                match_features['h2h_home_avg_goals'] = h2h_home_goals if not np.isnan(h2h_home_goals) else 1.0
                match_features['h2h_away_avg_goals'] = h2h_away_goals if not np.isnan(h2h_away_goals) else 0.5
            else:
                # No head-to-head data available
                match_features['h2h_home_win_rate'] = 0.45  # Home advantage
                match_features['h2h_away_win_rate'] = 0.30
                match_features['h2h_draw_rate'] = 0.25
                match_features['h2h_home_avg_goals'] = 1.5
                match_features['h2h_away_avg_goals'] = 1.0
        else:
            # No match date or no data
            match_features['h2h_home_win_rate'] = 0.45  # Home advantage
            match_features['h2h_away_win_rate'] = 0.30
            match_features['h2h_draw_rate'] = 0.25
            match_features['h2h_home_avg_goals'] = 1.5
            match_features['h2h_away_avg_goals'] = 1.0
        
        return pd.DataFrame([match_features])
    
    def train_models(self):
        """Train the prediction models with optimized parameters"""
        print("Training prediction models...")
        
        if self.matches_df.empty:
            print("No data available for training")
            return
        
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
                match_features = self._prepare_match_features(home_team, away_team, match['date'] if 'date' in match else None)
                
                # Determine match result (from home team perspective)
                if match['venue'] == 'Home':
                    result = 'home_win' if match['result'] == 'W' else ('draw' if match['result'] == 'D' else 'away_win')
                else:
                    result = 'away_win' if match['result'] == 'W' else ('draw' if match['result'] == 'D' else 'home_win')
                
                # Add to training data
                X_train.append(match_features.iloc[0].values)
                y_result.append(result)
                
                # Determine goal timing (using first half/second half goals if available)
                if 'hg' in match and 'ag' in match:  # Jika ada data gol babak pertama
                    first_half_goals = match['hg']
                    total_goals = match['gf'] + match['ga']
                    y_goal_timing.append(0 if first_half_goals > (total_goals / 2) else 1)
                else:
                    # Gunakan pendekatan probabilistik berdasarkan pola umum
                    y_goal_timing.append(np.random.choice([0, 1], p=[0.4, 0.6]))  # Lebih banyak gol di babak kedua
                
            except Exception as e:
                # Skip matches with missing data
                print(f"Skipping match due to error: {e}")
                continue
        
        if not X_train:
            print("No valid training examples could be created")
            return
            
        # Convert to numpy arrays
        X_train = np.array(X_train)
        
        # Scale features for better model performance
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Encode result labels
        self.label_encoder.fit(y_result)
        y_result_encoded = self.label_encoder.transform(y_result)
        
        # Split data for validation
        X_train_split, X_val, y_result_split, y_val_result = train_test_split(
            X_train_scaled, y_result_encoded, test_size=0.2, random_state=42
        )
        
        # Optimized parameters for result prediction model
        self.result_model = xgb.XGBClassifier(
            objective='multi:softprob',
            num_class=3,
            learning_rate=0.01,
            max_depth=6,
            n_estimators=300,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            gamma=0.2,
            reg_alpha=0.2,
            reg_lambda=1.5,
            scale_pos_weight=1,
            seed=42,
            use_label_encoder=False,
            eval_metric='mlogloss'
        )
        
        # Train with early stopping - PERBAIKAN: Hapus early_stopping_rounds
        self.result_model.fit(
            X_train_split, y_result_split,
            eval_set=[(X_val, y_val_result)],
            verbose=True
        )
        
        # Evaluate model
        val_preds = self.result_model.predict(X_val)
        val_accuracy = accuracy_score(y_val_result, val_preds)
        print(f"Validation accuracy: {val_accuracy:.4f}")
        
        # Train goal timing model with optimized parameters
        X_train_goal, X_val_goal, y_train_goal, y_val_goal = train_test_split(
            X_train_scaled, y_goal_timing, test_size=0.2, random_state=42
        )
        
        self.goal_timing_model = xgb.XGBClassifier(
            objective='binary:logistic',
            learning_rate=0.01,
            max_depth=4,
            n_estimators=200,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=2,
            gamma=0.1,
            reg_alpha=0.2,
            reg_lambda=1,
            scale_pos_weight=1,
            seed=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        
        # Train with early stopping - PERBAIKAN: Hapus early_stopping_rounds
        self.goal_timing_model.fit(
            X_train_goal, y_train_goal,
            eval_set=[(X_val_goal, y_val_goal)],
            verbose=True
        )
        
        print("Models trained successfully")
    
    def predict_match(self, home_team, away_team, match_date=None):
        """Predict the outcome of a match between two teams"""
        if self.result_model is None or self.goal_timing_model is None:
            print("Models not trained yet. Training models now...")
            self.train_models()
            if self.result_model is None:
                return {
                    'match': f"{home_team} vs {away_team}",
                    'match_result': {'Home Team Win': '33.3%', 'Draw': '33.3%', 'Away Team Win': '33.3%'},
                    'goal_timing': {'first_half_goals': '40.0%', 'second_half_goals': '60.0%'}
                }
        
        # Prepare features for the match
        try:
            match_features = self._prepare_match_features(home_team, away_team, match_date)
            
            # Scale features
            match_features_scaled = self.scaler.transform(match_features)
            
            # Predict match result
            result_probs = self.result_model.predict_proba(match_features_scaled)[0]
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
            goal_timing_prob = self.goal_timing_model.predict_proba(match_features_scaled)[0]
            
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
        except Exception as e:
            print(f"Error predicting match: {e}")
            return {
                'match': f"{home_team} vs {away_team}",
                'match_result': {'Home Team Win': '33.3%', 'Draw': '33.3%', 'Away Team Win': '33.3%'},
                'goal_timing': {'first_half_goals': '40.0%', 'second_half_goals': '60.0%'}
            }
    
    def visualize_prediction(self, prediction):
        """Visualize the match prediction with enhanced graphics"""
        # Extract probabilities
        home_win_prob = float(prediction['match_result'].get('Home Team Win', '0%').strip('%')) / 100
        draw_prob = float(prediction['match_result'].get('Draw', '0%').strip('%')) / 100
        away_win_prob = float(prediction['match_result'].get('Away Team Win', '0%').strip('%')) / 100
        
        # Create bar chart for match result
        plt.figure(figsize=(12, 8))
        
        # Plot match result probabilities
        plt.subplot(1, 2, 1)
        results = ['Home Win', 'Draw', 'Away Win']
        probs = [home_win_prob, draw_prob, away_win_prob]
        colors = ['#2ecc71', '#95a5a6', '#3498db']
        
        bars = plt.bar(results, probs, color=colors, width=0.6)
        plt.title(f'Match Result Prediction\n{prediction["match"]}', fontsize=14, fontweight='bold')
        plt.ylabel('Probability', fontsize=12)
        plt.ylim(0, 1)
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        
        # Add probability labels
        for bar, prob in zip(bars, probs):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{prob:.1%}', ha='center', fontsize=12, fontweight='bold')
        
        # Plot goal timing probabilities
        plt.subplot(1, 2, 2)
        first_half_prob = float(prediction['goal_timing']['first_half_goals'].strip('%')) / 100
        second_half_prob = float(prediction['goal_timing']['second_half_goals'].strip('%')) / 100
        
        bars = plt.bar(['First Half', 'Second Half'], [first_half_prob, second_half_prob], 
                      color=['#e67e22', '#9b59b6'], width=0.6)
        plt.title('Goal Timing Prediction', fontsize=14, fontweight='bold')
        plt.ylabel('Probability', fontsize=12)
        plt.ylim(0, 1)
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        
        # Add probability labels
        for bar, prob in zip(bars, [first_half_prob, second_half_prob]):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{prob:.1%}', ha='center', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.show()
    
    def feature_importance(self):
        """Visualize feature importance for the prediction models"""
        if self.result_model is None:
            print("Models not trained yet")
            return
            
        # Get feature importance
        importance = self.result_model.feature_importances_
        
        # Get feature names from the first match features
        if not self.matches_df.empty:
            home_team = self.matches_df['team'].iloc[0]
            away_team = self.matches_df['opponent'].iloc[0]
            match_features = self._prepare_match_features(home_team, away_team)
            feature_names = match_features.columns
            
            # Plot feature importance
            plt.figure(figsize=(12, 8))
            sorted_idx = np.argsort(importance)
            plt.barh(range(len(sorted_idx)), importance[sorted_idx], align='center')
            plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])
            plt.title('Feature Importance for Match Result Prediction', fontsize=14, fontweight='bold')
            plt.xlabel('Importance', fontsize=12)
            plt.tight_layout()
            plt.show()
        else:
            print("No data available for feature importance visualization")

# Kode di luar kelas untuk menjalankan contoh prediksi
if __name__ == "__main__":
    # Inisialisasi prediktor
    predictor = EPLPredictor()
    predictor.load_data()
    predictor.prepare_features()
    predictor.train_models()
    
    # Prediksi pertandingan contoh
    home_team = "Arsenal"
    away_team = "Man City"
    prediction = predictor.predict_match(home_team, away_team)
    
    # Tampilkan hasil prediksi
    print(f"\nPrediksi Pertandingan: {home_team} vs {away_team}")
    print("\nProbabilitas Hasil:")
    for outcome, prob in prediction['match_result'].items():
        print(f"  {outcome}: {prob}")
    
    print("\nProbabilitas Waktu Gol:")
    print(f"  Babak Pertama: {prediction['goal_timing']['first_half_goals']}")
    print(f"  Babak Kedua: {prediction['goal_timing']['second_half_goals']}")
    
    # Visualisasi prediksi
    predictor.visualize_prediction(prediction)
    
    # Tampilkan feature importance
    predictor.feature_importance()