#!/bin/bash
DATASETS="CARSDB CSAW_M FFB FOCUSPATH UTKFACE"
LOSSES="CrossEntropy BinomialUnimodal_CE CO2 UnimodalNet OrdinalEncoding"
EPSILONS="0.01 0.03 0.05 0.1 0.15 0.2 0.25 0.3"
TARGETS=("next_class" "furthest_class")
ATTACK_LOSSES=("ModelLoss" "CrossEntropy")

echo "Attack,AttackLoss,Dataset,Loss,Epsilon,Targeted,Target,Accuracy,OneOffAccuracy,MAE,QWK" > results.csv

for DATASET in $DATASETS; do
    for LOSS in $LOSSES; do
        # No attack
        python test.py $DATASET models/model-$DATASET-$LOSS.pth >> results.csv
    done
done

for ATTACK_LOSS in "${ATTACK_LOSSES[@]}"; do
    for DATASET in $DATASETS; do
        for LOSS in $LOSSES; do
            # Untargeted attacks
            for EPSILON in $EPSILONS; do
                python test.py $DATASET models/model-$DATASET-$LOSS.pth --attack GSA --epsilon $EPSILON --attack_loss $ATTACK_LOSS >> results.csv
            done

            # Targeted attacks
            for EPSILON in $EPSILONS; do
                for TARGET in "${TARGETS[@]}"; do
                    python test.py $DATASET models/model-$DATASET-$LOSS.pth --attack GSA --epsilon $EPSILON --targeted True --attack_target $TARGET --attack_loss $ATTACK_LOSS >> results.csv
                done
            done
        done
    done
done

for DATASET in $DATASETS; do
    for LOSS in $LOSSES; do
        # Targeted attacks
        for EPSILON in $EPSILONS; do
            for TARGET in "${TARGETS[@]}"; do
                python test.py $DATASET models/model-$DATASET-$LOSS.pth --attack FFA --epsilon $EPSILON --targeted True --attack_target $TARGET --attack_loss $ATTACK_LOSS >> results.csv
            done
        done
    done
done