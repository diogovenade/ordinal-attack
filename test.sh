#!/bin/bash
DATASETS="CARSDB CSAW_M FFB FOCUSPATH UTKFACE"
LOSSES="CrossEntropy BinomialUnimodal_CE CO2 UnimodalNet OrdinalEncoding"
ATTACKS="GradientSignAttack"
EPSILONS="0.01 0.03 0.05 0.1 0.15 0.2 0.25 0.3"

echo "Dataset,Loss,Epsilon,Accuracy,OneOffAccuracy,MAE,QWK" > results.csv

for DATASET in $DATASETS; do
   for LOSS in $LOSSES; do
        python test.py $DATASET models/model-$DATASET-$LOSS.pth >> results.csv
    done
done

for DATASET in $DATASETS; do
    for LOSS in $LOSSES; do
        for ATTACK in $ATTACKS; do
            for EPSILON in $EPSILONS; do
                python test.py $DATASET models/model-$DATASET-$LOSS.pth --attack $ATTACK --epsilon $EPSILON >> results.csv
            done
        done
    done
done