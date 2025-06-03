import os
import subprocess
import time
import sys

def run_command(command, description):
    """Run a command and print its output in real-time"""
    print(f"\n{'=' * 80}")
    print(f"RUNNING: {description}")
    print(f"{'=' * 80}\n")
    
    start_time = time.time()
    
    # Use shell=True for Windows compatibility
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        shell=True
    )
    
    # Print output in real-time
    for line in process.stdout:
        print(line, end='')
    
    process.wait()
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"\n{'=' * 80}")
    print(f"FINISHED: {description} in {duration:.2f} seconds")
    print(f"Return code: {process.returncode}")
    print(f"{'=' * 80}\n")
    
    return process.returncode

def main():
    print("Starting automated training process for Premier League Prediction Model")
    print("This script will run all training steps in sequence")
    
    # Determine the Python command to use (python or python3)
    python_cmd = "python"
    if sys.platform == "linux" or sys.platform == "darwin":
        try:
            subprocess.check_call(["python3", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            python_cmd = "python3"
        except:
            pass
    
    # Step 1: Data preprocessing
    preprocess_cmd = f"{python_cmd} Tuning_Training/preprocess_data.py"
    if run_command(preprocess_cmd, "Data Preprocessing") != 0:
        print("Error in preprocessing step. Stopping.")
        return
    
    # Step 2: Hyperparameter tuning
    tuning_cmd = f"{python_cmd} Tuning_Training/tune_xgboost_model.py"
    if run_command(tuning_cmd, "Hyperparameter Tuning") != 0:
        print("Error in hyperparameter tuning step. Stopping.")
        return
    
    # Step 3: Model training
    training_cmd = f"{python_cmd} Tuning_Training/train_xgboost_model.py"
    if run_command(training_cmd, "Model Training") != 0:
        print("Error in model training step. Stopping.")
        return
    
    print("\n✅ All training steps completed successfully!")
    print("The model has been trained and saved to models/xgboost_epl_model.json")
    print("\nYou can now use PredictFuture.py or PredictHistoryMatchup.py to make predictions")

if __name__ == "__main__":
    main() 