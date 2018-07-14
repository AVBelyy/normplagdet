#!/bin/sh

# Example usage:
# ./run_adversarial_models.sh <path-to-PAN-dataset> train_m1.dump m1
# ./run_adversarial_models.sh <path-to-PAN-dataset> train_m2.dump m2

DATA_DIR=$1/
OUT_DET_DIR=$2
MODEL_TYPE=$3

TASKS_DIR=$DATA_DIR/tasks/05-summary-obfuscation/

# Run adversarial model
python3 adversarial_models.py $DATA_DIR $TASKS_DIR $OUT_DET_DIR $MODEL_TYPE

# Evaluate it
rm -rf predictions/
mkdir predictions/
python3 text_alignment_solution.py $TASKS_DIR/pairs $DATA_DIR/src/ $DATA_DIR/susp/ predictions/ $OUT_DET_DIR
echo "=== Plagdet results ==="
python3 plagdet_measures.py -p $TASKS_DIR/ -d predictions/
echo "=== Normplagdet results ==="
python3 normplagdet_measures.py -p $TASKS_DIR/ -d predictions/ -t $DATA_DIR/susp/ -s $DATA_DIR/src/
