import pandas as pd
import numpy as np
import joblib
import os
import glob
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
import matplotlib.pyplot as plt
import seaborn as sns

def load_all_data():
    """Load all available data for comprehensive cross-validation"""
    # Print current working directory for debugging
    current_dir = os.getcwd()
    print(f"Current working directory: {current_dir}")
    
    # Use relative path from script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..'))
    data_dir = os.path.join(project_root, 'data', 'Dataset EPL New')
    print(f"Looking for CSV files in: {os.path.abspath(data_dir)}")
    
    # Mendapatkan list file secara manual tanpa menggunakan glob karena terkendala spasi
    try:
        if os.path.exists(data_dir):
            files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.csv')]
        else:
            files = []
    except Exception as e:
        print(f"Error listing directory: {e}")
        files = []
    
    print(f"Found {len(files)} dataset files: {files}")
    
    # If no files found, try potential alternative paths
    if not files:
        alternative_paths = [
            os.path.join(project_root, 'data', 'Dataset EPL New'),
            os.path.join(current_dir, 'data', 'Dataset EPL New'),
            os.path.join(os.path.dirname(current_dir), 'data', 'Dataset EPL New'),
            os.path.abspath(os.path.join('..', '..', 'data', 'Dataset EPL New')),
            os.path.abspath(os.path.join('..', 'data', 'Dataset EPL New')),
            os.path.abspath('data/Dataset EPL New')
        ]
        
        for alt_path in alternative_paths:
            print(f"Trying alternative path: {alt_path}")
            try:
                if os.path.exists(alt_path):
                    files = [os.path.join(alt_path, f) for f in os.listdir(alt_path) if f.endswith('.csv')]
                    if files:
                        print(f"Found {len(files)} dataset files in: {alt_path}")
                        data_dir = alt_path  # Update data_dir to the successful path
                        break
                else:
                    print(f"Path does not exist: {alt_path}")
            except Exception as e:
                print(f"Error listing alternative directory {alt_path}: {e}")
    
    if not files:
        raise FileNotFoundError(f"No dataset files found. Please check the path and file names.")
    
    # Baca semua file CSV dan gabungkan
    dataframes = []
    for file in files:
        try:
            print(f"Reading {file}...")
            df = pd.read_csv(file)
            dataframes.append(df)
            print(f"Successfully read {file}, shape: {df.shape}")
        except Exception as e:
            print(f"Error reading {file}: {e}")
    
    if not dataframes:
        raise ValueError("No data could be loaded from the CSV files.")
    
    df = pd.concat(dataframes, ignore_index=True)
    
    print(f"Combined dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Process categorical columns and target like in training script
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    if 'FTR' in categorical_cols:
        categorical_cols.remove('FTR')
    
    for col in categorical_cols:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    
    # Target encoding
    if 'FTR' in df.columns:
        df['FTR'] = df['FTR'].map({'H': 0, 'D': 1, 'A': 2})
    
    # Drop rows with missing target
    df = df[df['FTR'].notnull()]
    
    # Prepare features and target
    X = df.drop(['FTR'], axis=1)
    y = df['FTR']
    
    # Clean column names
    X.columns = [str(col).replace('[', '_').replace(']', '_').replace('<', '_').replace('>', '_') for col in X.columns]
    
    # Process numeric columns like in training script
    numeric_data = []
    for col in X.columns:
        try:
            converted_series = pd.to_numeric(X[col], errors='coerce')
            numeric_data.append(converted_series)
        except:
            pass
    
    X_clean = pd.concat(numeric_data, axis=1)
    
    # Fill NaN values with median
    X_clean = X_clean.fillna(X_clean.median())
    
    # Align target with cleaned features
    y_clean = y.loc[X_clean.index]
    
    return X_clean, y_clean, df

def perform_cross_validation(X, y, n_splits=5):
    """Perform cross-validation to get more reliable model performance estimates"""
    print(f"\nPerforming {n_splits}-fold cross-validation...")
    
    # Create the model with the same parameters as in training
    model = xgb.XGBClassifier(
        objective='multi:softmax', 
        num_class=3, 
        eval_metric='mlogloss', 
        use_label_encoder=False
    )
    
    # Define cross-validation strategy
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Perform cross-validation for accuracy
    cv_accuracy = cross_val_score(model, X, y, cv=kf, scoring='accuracy')
    
    print("\nCross-validation results:")
    print(f"Mean Accuracy: {cv_accuracy.mean()*100:.2f}% (±{cv_accuracy.std()*100:.2f}%)")
    print(f"Individual fold accuracies: {[f'{acc*100:.2f}%' for acc in cv_accuracy]}")
    
    # Get predictions for each fold for more detailed analysis
    y_pred = cross_val_predict(model, X, y, cv=kf)
    
    # Classification report
    print("\nClassification Report (across all CV folds):")
    print(classification_report(y, y_pred, target_names=['Home Win', 'Draw', 'Away Win']))
    
    # Confusion matrix
    cm = confusion_matrix(y, y_pred)
    print("\nConfusion Matrix (across all CV folds):")
    print(cm)
    
    # Visualize confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Home Win', 'Draw', 'Away Win'],
                yticklabels=['Home Win', 'Draw', 'Away Win'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Cross-Validation Confusion Matrix')
    plt.tight_layout()
    plt.savefig('cv_confusion_matrix.png')
    print("Cross-validation confusion matrix saved to 'cv_confusion_matrix.png'")
    
    # Get class-wise metrics
    class_names = ['Home Win', 'Draw', 'Away Win']
    class_metrics = []
    
    for i, class_name in enumerate(class_names):
        # For each class, calculate metrics
        true_pos = cm[i, i]
        false_pos = cm[:, i].sum() - true_pos
        false_neg = cm[i, :].sum() - true_pos
        true_neg = cm.sum() - (true_pos + false_pos + false_neg)
        
        precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
        recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        class_metrics.append({
            'Class': class_name,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1,
            'Support': cm[i, :].sum()
        })
    
    metrics_df = pd.DataFrame(class_metrics)
    print("\nClass-wise metrics:")
    print(metrics_df.to_string(index=False))
    
    # Visualize class-wise metrics
    plt.figure(figsize=(10, 6))
    metrics_df.set_index('Class')[['Precision', 'Recall', 'F1 Score']].plot(kind='bar')
    plt.title('Class-wise Metrics')
    plt.ylabel('Score')
    plt.ylim(0, 1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('class_metrics.png')
    print("Class-wise metrics visualization saved to 'class_metrics.png'")
    
    return cv_accuracy, y_pred, metrics_df

def perform_leave_one_season_out_cv():
    """Use each season as a test set to evaluate how well the model generalizes to new seasons"""
    # Print current working directory for debugging
    current_dir = os.getcwd()
    print(f"Current working directory: {current_dir}")
    
    # Use relative path from script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..'))
    data_dir = os.path.join(project_root, 'data', 'Dataset EPL New')
    print(f"Looking for CSV files in: {os.path.abspath(data_dir)}")
    
    # Mendapatkan list file secara manual tanpa menggunakan glob karena terkendala spasi
    try:
        if os.path.exists(data_dir):
            files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.csv')]
        else:
            files = []
    except Exception as e:
        print(f"Error listing directory: {e}")
        files = []
    
    print(f"Found {len(files)} dataset files: {files}")
    
    # If no files found, try potential alternative paths
    if not files:
        alternative_paths = [
            os.path.join(project_root, 'data', 'Dataset EPL New'),
            os.path.join(current_dir, 'data', 'Dataset EPL New'),
            os.path.join(os.path.dirname(current_dir), 'data', 'Dataset EPL New'),
            os.path.abspath(os.path.join('..', '..', 'data', 'Dataset EPL New')),
            os.path.abspath(os.path.join('..', 'data', 'Dataset EPL New')),
            os.path.abspath('data/Dataset EPL New')
        ]
        
        for alt_path in alternative_paths:
            print(f"Trying alternative path: {alt_path}")
            try:
                if os.path.exists(alt_path):
                    files = [os.path.join(alt_path, f) for f in os.listdir(alt_path) if f.endswith('.csv')]
                    if files:
                        print(f"Found {len(files)} dataset files in: {alt_path}")
                        data_dir = alt_path  # Update data_dir to the successful path
                        break
                else:
                    print(f"Path does not exist: {alt_path}")
            except Exception as e:
                print(f"Error listing alternative directory {alt_path}: {e}")
    
    if not files:
        raise FileNotFoundError(f"No dataset files found. Please check the path and file names.")
    
    # Urutkan file berdasarkan nama
    files = sorted(files)
    
    if len(files) < 2:
        print("Need at least 2 seasons for leave-one-season-out CV. Skipping.")
        return None
        
    print(f"\nPerforming leave-one-season-out cross-validation with {len(files)} seasons...")
    
    results = []
    
    for test_file in files:
        season_name = os.path.basename(test_file).replace('.csv', '')
        print(f"\nUsing {season_name} as test set...")
        
        # Load test data
        test_df = pd.read_csv(test_file)
        
        # Load all other seasons as training data
        train_dfs = []
        for f in files:
            if f != test_file:
                train_dfs.append(pd.read_csv(f))
        
        train_df = pd.concat(train_dfs, ignore_index=True)
        
        # Process data (same as in training script)
        # Process categorical columns and target
        for df in [train_df, test_df]:
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
            if 'FTR' in categorical_cols:
                categorical_cols.remove('FTR')
            
            for col in categorical_cols:
                df[col] = LabelEncoder().fit_transform(df[col].astype(str))
            
            # Target encoding
            if 'FTR' in df.columns:
                df['FTR'] = df['FTR'].map({'H': 0, 'D': 1, 'A': 2})
            
            # Drop rows with missing target
            df = df[df['FTR'].notnull()]
        
        # Prepare features and targets
        X_train = train_df.drop(['FTR'], axis=1)
        y_train = train_df['FTR']
        X_test = test_df.drop(['FTR'], axis=1)
        y_test = test_df['FTR']
        
        # Clean column names
        for X in [X_train, X_test]:
            X.columns = [str(col).replace('[', '_').replace(']', '_').replace('<', '_').replace('>', '_') for col in X.columns]
        
        # Process numeric columns
        for X, name in [(X_train, 'train'), (X_test, 'test')]:
            numeric_data = []
            for col in X.columns:
                try:
                    converted_series = pd.to_numeric(X[col], errors='coerce')
                    numeric_data.append(converted_series)
                except:
                    pass
            
            if name == 'train':
                X_train_clean = pd.concat(numeric_data, axis=1)
                X_train_clean = X_train_clean.fillna(X_train_clean.median())
                y_train_clean = y_train.loc[X_train_clean.index]
            else:
                X_test_clean = pd.concat(numeric_data, axis=1)
                X_test_clean = X_test_clean.fillna(X_test_clean.median())
                y_test_clean = y_test.loc[X_test_clean.index]
        
        # Get common features
        common_features = [f for f in X_train_clean.columns if f in X_test_clean.columns]
        X_train_final = X_train_clean[common_features]
        X_test_final = X_test_clean[common_features]
        
        print(f"Training on {len(X_train_final)} matches with {len(common_features)} features")
        print(f"Testing on {len(X_test_final)} matches")
        
        # Train model
        model = xgb.XGBClassifier(
            objective='multi:softmax', 
            num_class=3, 
            eval_metric='mlogloss', 
            use_label_encoder=False
        )
        
        model.fit(X_train_final, y_train_clean)
        
        # Evaluate
        accuracy = model.score(X_test_final, y_test_clean)
        y_pred = model.predict(X_test_final)
        
        print(f"Accuracy on {season_name}: {accuracy*100:.2f}%")
        
        # Store results
        results.append({
            'Season': season_name,
            'Accuracy': accuracy,
            'Train Size': len(X_train_final),
            'Test Size': len(X_test_final)
        })
        
        # Classification report for this season
        print(f"\nClassification Report for {season_name}:")
        print(classification_report(y_test_clean, y_pred, target_names=['Home Win', 'Draw', 'Away Win']))
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    print("\nLeave-one-season-out cross-validation results:")
    print(results_df.to_string(index=False))
    print(f"Average accuracy across seasons: {results_df['Accuracy'].mean()*100:.2f}%")
    
    # Visualize results
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Season', y='Accuracy', data=results_df)
    plt.title('Accuracy by Season (Leave-One-Season-Out CV)')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    for i, row in enumerate(results_df.itertuples()):
        plt.text(i, row.Accuracy + 0.02, f'{row.Accuracy:.2f}', ha='center')
    plt.tight_layout()
    plt.savefig('season_cv_results.png')
    print("Season-wise CV results visualization saved to 'season_cv_results.png'")
    
    return results_df

def main():
    print("XGBoost Model Cross-Validation")
    print("==============================")
    
    # Load all data
    try:
        X, y, df = load_all_data()
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Perform k-fold cross-validation
    try:
        cv_results = perform_cross_validation(X, y)
    except Exception as e:
        print(f"Error during k-fold cross-validation: {e}")
    
    # Perform leave-one-season-out cross-validation
    try:
        season_cv_results = perform_leave_one_season_out_cv()
    except Exception as e:
        print(f"Error during leave-one-season-out cross-validation: {e}")
    
    print("\nCross-validation completed.")

if __name__ == "__main__":
    main() 