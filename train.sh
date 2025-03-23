#!/bin/bash
MODELS="DHCI FFB FGNET"
LOSSES="CrossEntropy BinomialUnimodal_CE CO2 UnimodalNet"

for MODEL in $MODELS; do
    for LOSS in $LOSSES; do
        sbatch python train.py $MODEL $LOSS models/model-$MODEL-$LOSS.pth
    done
done
