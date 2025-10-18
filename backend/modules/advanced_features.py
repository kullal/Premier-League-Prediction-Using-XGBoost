"""
Advanced Feature Engineering Module
Provides additional features for better prediction accuracy:
- Head-to-head (H2H) history
- Venue-specific performance
- Rest days between matches
- Seasonal form trends
- Goal difference momentum
- Recent wins/losses streaks
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def clean_feature_name(col_name):
    """Cleans a column name to be XGBoost compatible."""
    new_name = str(col_name).replace('<', '_lt_').replace('>', '_gt_').replace('=', '_eq_')
    new_name = new_name.replace('[', '_').replace(']', '_').replace(' ', '_')
    new_name = "".join(char for char in new_name if char.isalnum() or char == '_')
    new_name = new_name.strip('_')
    if not new_name:
        new_name = 'unnamed_col_fallback'
    return new_name


def get_h2h_features(home_team, away_team, date_of_match, historical_df, max_h2h=10):
    """
    Calculate head-to-head statistics between two teams.
    
    Args:
        home_team (str): Home team name
        away_team (str): Away team name
        date_of_match (datetime): Date of the match
        historical_df (pd.DataFrame): Historical match data
        max_h2h (int): Maximum number of H2H matches to consider
    
    Returns:
        dict: H2H features
    """
    features = {}
    
    # Get all H2H matches before this date
    h2h_matches = historical_df[
        (((historical_df['HomeTeam'] == home_team) & (historical_df['AwayTeam'] == away_team)) |
         ((historical_df['HomeTeam'] == away_team) & (historical_df['AwayTeam'] == home_team))) &
        (historical_df['Date'] < date_of_match)
    ].sort_values(by='Date', ascending=False).head(max_h2h)
    
    if len(h2h_matches) == 0:
        # No H2H history
        features['H2H_Home_Wins'] = 0
        features['H2H_Draws'] = 0
        features['H2H_Away_Wins'] = 0
        features['H2H_Home_Goals_Avg'] = 0
        features['H2H_Away_Goals_Avg'] = 0
        features['H2H_Total_Matches'] = 0
    else:
        home_wins = 0
        draws = 0
        away_wins = 0
        home_goals_total = 0
        away_goals_total = 0
        
        for _, match in h2h_matches.iterrows():
            if match['HomeTeam'] == home_team:
                home_goals_total += match['FTHG']
                away_goals_total += match['FTAG']
                if match['FTHG'] > match['FTAG']:
                    home_wins += 1
                elif match['FTHG'] == match['FTAG']:
                    draws += 1
                else:
                    away_wins += 1
            else:  # Away team was home in that match
                home_goals_total += match['FTAG']
                away_goals_total += match['FTHG']
                if match['FTAG'] > match['FTHG']:
                    home_wins += 1
                elif match['FTAG'] == match['FTHG']:
                    draws += 1
                else:
                    away_wins += 1
        
        features['H2H_Home_Wins'] = home_wins
        features['H2H_Draws'] = draws
        features['H2H_Away_Wins'] = away_wins
        features['H2H_Home_Goals_Avg'] = home_goals_total / len(h2h_matches)
        features['H2H_Away_Goals_Avg'] = away_goals_total / len(h2h_matches)
        features['H2H_Total_Matches'] = len(h2h_matches)
    
    return features


def get_venue_specific_performance(team_name, is_home, date_of_match, historical_df, window=10):
    """
    Calculate team's performance at specific venue (home or away).
    
    Args:
        team_name (str): Team name
        is_home (bool): True if home venue, False if away
        date_of_match (datetime): Date of the match
        historical_df (pd.DataFrame): Historical match data
        window (int): Number of recent venue-specific matches
    
    Returns:
        dict: Venue-specific features
    """
    features = {}
    prefix = 'Home_Venue' if is_home else 'Away_Venue'
    
    if is_home:
        venue_matches = historical_df[
            (historical_df['HomeTeam'] == team_name) &
            (historical_df['Date'] < date_of_match)
        ].sort_values(by='Date', ascending=False).head(window)
        
        if len(venue_matches) > 0:
            wins = (venue_matches['FTHG'] > venue_matches['FTAG']).sum()
            draws = (venue_matches['FTHG'] == venue_matches['FTAG']).sum()
            losses = (venue_matches['FTHG'] < venue_matches['FTAG']).sum()
            goals_scored = venue_matches['FTHG'].mean()
            goals_conceded = venue_matches['FTAG'].mean()
        else:
            wins = draws = losses = 0
            goals_scored = goals_conceded = 0
    else:
        venue_matches = historical_df[
            (historical_df['AwayTeam'] == team_name) &
            (historical_df['Date'] < date_of_match)
        ].sort_values(by='Date', ascending=False).head(window)
        
        if len(venue_matches) > 0:
            wins = (venue_matches['FTAG'] > venue_matches['FTHG']).sum()
            draws = (venue_matches['FTAG'] == venue_matches['FTHG']).sum()
            losses = (venue_matches['FTAG'] < venue_matches['FTHG']).sum()
            goals_scored = venue_matches['FTAG'].mean()
            goals_conceded = venue_matches['FTHG'].mean()
        else:
            wins = draws = losses = 0
            goals_scored = goals_conceded = 0
    
    features[f'{prefix}_Wins_L{window}'] = wins
    features[f'{prefix}_Draws_L{window}'] = draws
    features[f'{prefix}_Losses_L{window}'] = losses
    features[f'{prefix}_Goals_Scored_Avg'] = goals_scored
    features[f'{prefix}_Goals_Conceded_Avg'] = goals_conceded
    features[f'{prefix}_Win_Rate'] = wins / window if window > 0 else 0
    
    return features


def get_rest_days_feature(team_name, date_of_match, historical_df):
    """
    Calculate rest days since last match for a team.
    
    Args:
        team_name (str): Team name
        date_of_match (datetime): Date of current match
        historical_df (pd.DataFrame): Historical match data
    
    Returns:
        int: Days since last match
    """
    last_match = historical_df[
        ((historical_df['HomeTeam'] == team_name) | (historical_df['AwayTeam'] == team_name)) &
        (historical_df['Date'] < date_of_match)
    ].sort_values(by='Date', ascending=False).head(1)
    
    if len(last_match) == 0:
        return 7  # Default 1 week if no history
    
    rest_days = (date_of_match - last_match.iloc[0]['Date']).days
    return min(rest_days, 30)  # Cap at 30 days


def get_winning_streak_feature(team_name, date_of_match, historical_df, window=5):
    """
    Calculate current winning/losing streak.
    
    Args:
        team_name (str): Team name
        date_of_match (datetime): Date of match
        historical_df (pd.DataFrame): Historical match data
        window (int): Max number of recent matches to check
    
    Returns:
        dict: Streak features
    """
    features = {}
    
    recent_matches = historical_df[
        ((historical_df['HomeTeam'] == team_name) | (historical_df['AwayTeam'] == team_name)) &
        (historical_df['Date'] < date_of_match)
    ].sort_values(by='Date', ascending=False).head(window)
    
    if len(recent_matches) == 0:
        features['Win_Streak'] = 0
        features['Unbeaten_Streak'] = 0
        features['Loss_Streak'] = 0
        return features
    
    win_streak = 0
    unbeaten_streak = 0
    loss_streak = 0
    
    for _, match in recent_matches.iterrows():
        if match['HomeTeam'] == team_name:
            if match['FTHG'] > match['FTAG']:
                win_streak += 1
                unbeaten_streak += 1
                loss_streak = 0
            elif match['FTHG'] == match['FTAG']:
                unbeaten_streak += 1
                win_streak = 0
                loss_streak = 0
            else:
                loss_streak += 1
                win_streak = 0
                unbeaten_streak = 0
                break
        else:
            if match['FTAG'] > match['FTHG']:
                win_streak += 1
                unbeaten_streak += 1
                loss_streak = 0
            elif match['FTAG'] == match['FTHG']:
                unbeaten_streak += 1
                win_streak = 0
                loss_streak = 0
            else:
                loss_streak += 1
                win_streak = 0
                unbeaten_streak = 0
                break
    
    features['Win_Streak'] = win_streak
    features['Unbeaten_Streak'] = unbeaten_streak
    features['Loss_Streak'] = loss_streak
    
    return features


def get_goal_difference_momentum(team_name, date_of_match, historical_df, window=5):
    """
    Calculate goal difference trend over recent matches.
    
    Args:
        team_name (str): Team name
        date_of_match (datetime): Date of match
        historical_df (pd.DataFrame): Historical match data
        window (int): Number of recent matches
    
    Returns:
        dict: Goal difference features
    """
    features = {}
    
    recent_matches = historical_df[
        ((historical_df['HomeTeam'] == team_name) | (historical_df['AwayTeam'] == team_name)) &
        (historical_df['Date'] < date_of_match)
    ].sort_values(by='Date', ascending=False).head(window)
    
    if len(recent_matches) == 0:
        features['GD_Momentum'] = 0
        features['GD_Avg_Recent'] = 0
        return features
    
    goal_diffs = []
    for _, match in recent_matches.iterrows():
        if match['HomeTeam'] == team_name:
            gd = match['FTHG'] - match['FTAG']
        else:
            gd = match['FTAG'] - match['FTHG']
        goal_diffs.append(gd)
    
    # Calculate trend (positive if improving, negative if declining)
    if len(goal_diffs) >= 3:
        recent_gd = np.mean(goal_diffs[:3])
        older_gd = np.mean(goal_diffs[3:]) if len(goal_diffs) > 3 else recent_gd
        momentum = recent_gd - older_gd
    else:
        momentum = 0
    
    features['GD_Momentum'] = momentum
    features['GD_Avg_Recent'] = np.mean(goal_diffs)
    
    return features


def calculate_prediction_confidence(probabilities):
    """
    Calculate confidence score based on probability distribution.
    Higher entropy = lower confidence, lower entropy = higher confidence.
    
    Args:
        probabilities (array): Array of class probabilities [away_win, draw, home_win]
    
    Returns:
        dict: Confidence metrics
    """
    # Ensure probabilities sum to 1
    probs = np.array(probabilities) / np.sum(probabilities)
    
    # Calculate entropy (0 = certain, log(3) ≈ 1.1 = uniform/uncertain)
    epsilon = 1e-10
    entropy = -np.sum(probs * np.log(probs + epsilon))
    max_entropy = np.log(3)  # Maximum entropy for 3 classes
    
    # Normalize to 0-1 scale (1 = high confidence, 0 = low confidence)
    confidence_score = 1 - (entropy / max_entropy)
    
    # Get margin (difference between top 2 probabilities)
    sorted_probs = np.sort(probs)[::-1]
    margin = sorted_probs[0] - sorted_probs[1]
    
    # Categorize confidence
    if confidence_score > 0.7 or margin > 0.3:
        confidence_level = "High"
    elif confidence_score > 0.4 or margin > 0.15:
        confidence_level = "Medium"
    else:
        confidence_level = "Low"
    
    return {
        "confidence_score": float(confidence_score),
        "confidence_level": confidence_level,
        "entropy": float(entropy),
        "margin": float(margin)
    }


def get_all_advanced_features(home_team, away_team, date_of_match, historical_df):
    """
    Combine all advanced features into one dictionary.
    
    Args:
        home_team (str): Home team name
        away_team (str): Away team name
        date_of_match (datetime): Date of match
        historical_df (pd.DataFrame): Historical match data
    
    Returns:
        dict: All advanced features with cleaned names
    """
    all_features = {}
    
    # H2H features
    h2h_feats = get_h2h_features(home_team, away_team, date_of_match, historical_df)
    for k, v in h2h_feats.items():
        all_features[clean_feature_name(k)] = v
    
    # Venue-specific performance
    home_venue_feats = get_venue_specific_performance(home_team, True, date_of_match, historical_df)
    away_venue_feats = get_venue_specific_performance(away_team, False, date_of_match, historical_df)
    
    for k, v in home_venue_feats.items():
        all_features[clean_feature_name(f'Home_{k}')] = v
    for k, v in away_venue_feats.items():
        all_features[clean_feature_name(f'Away_{k}')] = v
    
    # Rest days
    all_features['Home_Rest_Days'] = get_rest_days_feature(home_team, date_of_match, historical_df)
    all_features['Away_Rest_Days'] = get_rest_days_feature(away_team, date_of_match, historical_df)
    
    # Winning streaks
    home_streak = get_winning_streak_feature(home_team, date_of_match, historical_df)
    away_streak = get_winning_streak_feature(away_team, date_of_match, historical_df)
    
    for k, v in home_streak.items():
        all_features[clean_feature_name(f'Home_{k}')] = v
    for k, v in away_streak.items():
        all_features[clean_feature_name(f'Away_{k}')] = v
    
    # Goal difference momentum
    home_gd = get_goal_difference_momentum(home_team, date_of_match, historical_df)
    away_gd = get_goal_difference_momentum(away_team, date_of_match, historical_df)
    
    for k, v in home_gd.items():
        all_features[clean_feature_name(f'Home_{k}')] = v
    for k, v in away_gd.items():
        all_features[clean_feature_name(f'Away_{k}')] = v
    
    return all_features


def get_dynamic_default_odds(home_team, away_team, date_of_match, historical_df):
    """
    Calculate dynamic default odds based on historical team performance.
    Better than fixed 2.0/3.0/4.0 defaults.
    
    Args:
        home_team (str): Home team name
        away_team (str): Away team name
        date_of_match (datetime): Date of match
        historical_df (pd.DataFrame): Historical match data
    
    Returns:
        dict: Dynamic odds estimates
    """
    # Get recent form (last 10 matches)
    home_matches = historical_df[
        ((historical_df['HomeTeam'] == home_team) | (historical_df['AwayTeam'] == home_team)) &
        (historical_df['Date'] < date_of_match)
    ].sort_values(by='Date', ascending=False).head(10)
    
    away_matches = historical_df[
        ((historical_df['HomeTeam'] == away_team) | (historical_df['AwayTeam'] == away_team)) &
        (historical_df['Date'] < date_of_match)
    ].sort_values(by='Date', ascending=False).head(10)
    
    # Calculate win rates
    home_wins = 0
    home_draws = 0
    for _, match in home_matches.iterrows():
        if match['HomeTeam'] == home_team:
            if match['FTHG'] > match['FTAG']:
                home_wins += 1
            elif match['FTHG'] == match['FTAG']:
                home_draws += 1
        else:
            if match['FTAG'] > match['FTHG']:
                home_wins += 1
            elif match['FTAG'] == match['FTHG']:
                home_draws += 1
    
    away_wins = 0
    away_draws = 0
    for _, match in away_matches.iterrows():
        if match['HomeTeam'] == away_team:
            if match['FTHG'] > match['FTAG']:
                away_wins += 1
            elif match['FTHG'] == match['FTAG']:
                away_draws += 1
        else:
            if match['FTAG'] > match['FTHG']:
                away_wins += 1
            elif match['FTAG'] == match['FTHG']:
                away_draws += 1
    
    home_total = len(home_matches) if len(home_matches) > 0 else 1
    away_total = len(away_matches) if len(away_matches) > 0 else 1
    
    home_win_prob = (home_wins / home_total) * 0.6 + 0.1  # Home advantage
    away_win_prob = (away_wins / away_total) * 0.4
    draw_prob = 1.0 - home_win_prob - away_win_prob
    draw_prob = max(0.15, min(0.35, draw_prob))  # Constrain draw probability
    
    # Normalize
    total = home_win_prob + draw_prob + away_win_prob
    home_win_prob /= total
    draw_prob /= total
    away_win_prob /= total
    
    # Convert probabilities to odds (with margin)
    margin = 1.05  # 5% bookmaker margin
    odds_home = margin / home_win_prob if home_win_prob > 0 else 10.0
    odds_draw = margin / draw_prob if draw_prob > 0 else 10.0
    odds_away = margin / away_win_prob if away_win_prob > 0 else 10.0
    
    # Clamp odds to realistic ranges
    odds_home = max(1.2, min(15.0, odds_home))
    odds_draw = max(2.5, min(10.0, odds_draw))
    odds_away = max(1.5, min(20.0, odds_away))
    
    return {
        "B365H": round(odds_home, 2),
        "B365D": round(odds_draw, 2),
        "B365A": round(odds_away, 2),
        "BSH": round(odds_home, 2),
        "BSD": round(odds_draw, 2),
        "BSA": round(odds_away, 2),
        "BWH": round(odds_home, 2),
        "BWD": round(odds_draw, 2),
        "BWA": round(odds_away, 2),
        "PSH": round(odds_home, 2),
        "PSD": round(odds_draw, 2),
        "PSA": round(odds_away, 2),
        "MaxH": round(odds_home * 1.05, 2),
        "MaxD": round(odds_draw * 1.05, 2),
        "MaxA": round(odds_away * 1.05, 2),
        "AvgH": round(odds_home, 2),
        "AvgD": round(odds_draw, 2),
        "AvgA": round(odds_away, 2)
    }
