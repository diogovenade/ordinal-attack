#!/bin/bash
DATASETS="CARSDB CSAW_M FFB FOCUSPATH UTKFACE"
LOSSES="CrossEntropy BinomialUnimodal_CE CO2 UnimodalNet OrdinalEncoding ORD_ACL"
EPSILONS="0.005 0.02 0.045 0.08 0.125 0.18 0.245"
TARGETS=("next_class" "furthest_class")
ATTACK_LOSSES=("ModelLoss" "CrossEntropy")
ATTACKS=("GradientSignAttack" "LinfBasicIterativeAttack" "MomentumIterativeAttack")

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
            if [[ ("$LOSS" == "OrdinalEncoding" || "$LOSS" == "ORD_ACL") && "$ATTACK_LOSS" == "CrossEntropy" ]]; then
                continue
            fi
            for ATTACK in "${ATTACKS[@]}"; do
                # Untargeted attack
                for EPSILON in $EPSILONS; do
                    python test.py $DATASET models/model-$DATASET-$LOSS.pth --attack $ATTACK --epsilon $EPSILON --attack_loss $ATTACK_LOSS >> results.csv
                done

                # Targeted attack
                for EPSILON in $EPSILONS; do
                    for TARGET in "${TARGETS[@]}"; do
                        python test.py $DATASET models/model-$DATASET-$LOSS.pth --attack $ATTACK --epsilon $EPSILON --targeted True --attack_target $TARGET --attack_loss $ATTACK_LOSS >> results.csv
                    done
                done

            done
        done
    done
done

for DATASET in $DATASETS; do
    for LOSS in $LOSSES; do
        # Targeted attacks - FFA
        for EPSILON in $EPSILONS; do
            for TARGET in "${TARGETS[@]}"; do
                python test.py $DATASET models/model-$DATASET-$LOSS.pth --attack FFA --epsilon $EPSILON --targeted True --attack_target $TARGET >> results.csv
            done
        done
    done
done