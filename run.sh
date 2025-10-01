#!/bin/bash
# Stori is cool 
# Orchestrates ML pipeline
 
# ------------------------ 
# Config and paths 
# ------------------------ 
CONFIG_FILE="config.yaml" 
INPUT_DATA="data/input.csv" 
OUTPUT_DIR="outputs" 

echo "Running pipeline..." 
echo "Config file: $CONFIG_FILE" 
echo "Input data: $INPUT_DATA" 
echo "Output dir: $OUTPUT_DIR" 

# ------------------------ 
# Step 1: Preprocess data 
# ------------------------ 
python src/preprocess.py --config "$CONFIG_FILE" --input "$INPUT_DATA" --output "$OUTPUT_DIR" 

# ------------------------ 
# Step 2: Initial hyperparameter tuning 
# ------------------------ 
python src/model_pipeline.py --config "$CONFIG_FILE" --input "$OUTPUT_DIR/processed_data_train.csv" --output "$OUTPUT_DIR" --stage 0 

# ------------------------ 
# Step 3: Feature selection 
# ------------------------ 
python src/feature_selection.py --config "$CONFIG_FILE" --input "$OUTPUT_DIR/processed_data_train.csv" --output "$OUTPUT_DIR" --model "$OUTPUT_DIR/model/final_model.pkl" 

# ------------------------ 
# Step 4: Final Bayesian hyperparameter tuning + retrain final model 
# ------------------------ 
python src/model_pipeline.py --config "$CONFIG_FILE" --input "$OUTPUT_DIR/processed_selected.csv" --output "$OUTPUT_DIR" --stage 1 

# ------------------------ 
# Step 5: Generate evaluation plots 
# ------------------------ 
python src/visualization.py --config "$CONFIG_FILE" --output "$OUTPUT_DIR" 

# ------------------------ 
# Step 6: Score new data 
# ------------------------ 
python src/scoring.py --config "$CONFIG_FILE" --input "data/new_data.csv" --output "$OUTPUT_DIR" 

echo "Pipeline finished. Artifacts stored in:" 
echo " Model -> $OUTPUT_DIR/model/" 
echo " Predictions -> $OUTPUT_DIR/predictions/" 
echo " Visualizations -> $OUTPUT_DIR/plots/"