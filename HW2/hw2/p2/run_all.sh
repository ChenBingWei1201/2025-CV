#!/bin/bash

# Activate the conda environment
conda activate cv_hw2_py38

# Set the dataset directory
DATASET_DIR="../hw2_data/p2_data"

# Get the model type from config.py
MODEL_TYPE=$(grep "model_type" config.py | cut -d "'" -f 2)
echo "Using model type: $MODEL_TYPE"

# First, train the model
echo "Training model..."
python p2_train.py --dataset_dir $DATASET_DIR

# Create checkpoint directory if it doesn't exist
mkdir -p checkpoint

# Copy the best model to the checkpoint directory
echo "Copying best model to checkpoint directory..."
LATEST_EXP=$(ls -t ./experiment | head -n 1)
cp ./experiment/$LATEST_EXP/model/model_best.pth ./checkpoint/${MODEL_TYPE}_best.pth

# Run inference
echo "Running inference..."
python p2_inference.py --test_datadir $DATASET_DIR/val --model_type $MODEL_TYPE --output_path ./output/pred_${MODEL_TYPE}.csv

# Evaluate the model
echo "Evaluating model..."
python p2_eval.py --csv_path ./output/pred_${MODEL_TYPE}.csv --annos_path $DATASET_DIR/val/annotations.json

echo "All done!"