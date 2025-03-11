#!/bin/bash
LOSSES="CrossEntropy BinomialUnimodal_CE CO2 UnimodalNet"
for LOSS in $LOSSES; do
    echo "test $LOSS"
    sbatch python3 test.py FFB model-FFB-$LOSS.pth
done