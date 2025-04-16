#!/bin/bash
DATASETS="CARSDB CSAW_M FFB FOCUSPATH UTKFACE"
LOSSES="CrossEntropy BinomialUnimodal_CE CO2 UnimodalNet OrdinalEncoding"
ATTACKS="GradientSignAttack"
EPSILONS="0.01 0.03 0.05 0.1 0.15 0.2 0.25 0.3"
TARGETS=("next_class" "furthest_class")

echo "Dataset,Loss,Epsilon,Targeted,Target,Accuracy,OneOffAccuracy,MAE,QWK" > results.csv

for DATASET in $DATASETS; do
    for LOSS in $LOSSES; do
        # No attack
        python test.py $DATASET models/model-$DATASET-$LOSS.pth >> results.csv

        # Untargeted attacks
        for ATTACK in $ATTACKS; do
            for EPSILON in $EPSILONS; do
                python test.py $DATASET models/model-$DATASET-$LOSS.pth --attack $ATTACK --epsilon $EPSILON >> results.csv
            done
        done

        # Targeted attacks
        for ATTACK in $ATTACKS; do
            for EPSILON in $EPSILONS; do
                for TARGET in "${TARGETS[@]}"; do
                    python test.py $DATASET models/model-$DATASET-$LOSS.pth --attack $ATTACK --epsilon $EPSILON --targeted True --attack_target $TARGET >> results.csv
                done
            done
        done
    done
done