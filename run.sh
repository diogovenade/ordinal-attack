#!/bin/bash
LOSSES="CrossEntropy BinomialUnimodal_CE CO2 UnimodalNet"
for LOSS in $LOSSES; do
    echo "train $LOSS"
    sbatch python3 train.py FFB $LOSS model-FFB-$LOSS.pth
done
for LOSS in $LOSSES; do
    echo "test $LOSS"
    sbatch python3 test.py FFB model-FFB-$LOSS.pth
done