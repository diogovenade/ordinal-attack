#!/bin/bash
DATASETS="CARSDB CSAW_M FFB FOCUSPATH UTKFACE"
LOSSES="CrossEntropy BinomialUnimodal_CE CO2 UnimodalNet OrdinalEncoding ORD_ACL"

for DATASET in $DATASETS; do
    for LOSS in $LOSSES; do
        python train.py $DATASET $LOSS models/model-$DATASET-$LOSS.pth
    done
done