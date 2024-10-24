#!/bin/bash

# Example call:
# ./run_inference_flow.sh \
#     /home/datduong/datduong_research_sda/PlayMaker/playmaker-ai-models
#     "20221205"
#     /home/datduong/datduong_research_sda/PlayMaker/playmaker-ai-models/forecasting/saved_models/TFT_2022_12_14-22_24_53/best_model

usage_mess="Usage: $0 <Path_to_parent_directory> <Inference_date> <Path_to_trained_model_directory>"

if (( $# != 3 ))
then
    echo "Not enough input arguments"
    echo $usage_mess
    exit 1
fi

PARENT_DIR=$1
INFERENCE_DATE=$2
MODEL_DIR=$3

echo "ROOT DIRECTORY: $PARENT_DIR"
echo "INFERENCE_DATE: $INFERENCE_DATE"
echo "MODEL_DIR: $MODEL_DIR"

export PYTHONPATH=$PARENT_DIR:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0

echo "Starting inference job at: `date`"
echo "Setting up environment"
cd "$PARENT_DIR"
source venv/bin/activate
echo "Initializing done!"

RESULT_DIR="$MODEL_DIR/inference_results/$INFERENCE_DATE"
MODEL_PREDICTION_SAVE_PATH="$RESULT_DIR/inference_predictions.pkl"
FINAL_RESULT_DIR="$PARENT_DIR/inference_results/$INFERENCE_DATE"

echo "RESULT_DIR: $RESULT_DIR"
echo "Creating directory for saving inference results"
mkdir -p $RESULT_DIR
mkdir -p $FINAL_RESULT_DIR


echo "Starting inference process"

python inference/run_inference.py \
    -id $INFERENCE_DATE -wd $MODEL_DIR -s $RESULT_DIR && \
python inference/generate_json_results.py \
    -id $INFERENCE_DATE -wd $MODEL_DIR -p $MODEL_PREDICTION_SAVE_PATH -s $RESULT_DIR

echo "Finished job with exit code $? at: `date`"
