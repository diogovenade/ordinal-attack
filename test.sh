#!/bin/bash
DATASETS="CARSDB CSAW_M FFB FOCUSPATH UTKFACE"
LOSSES="CrossEntropy BinomialUnimodal_CE CO2 UnimodalNet OrdinalEncoding"
ATTACKS="GradientSignAttack"
EPSILONS="0.01 0.03 0.05 0.1 0.15 0.2 0.25 0.3"

echo "Loss,Epsilon,Accuracy,F1,Precision,Recall,TP,TN,FP,FN" > xxx.csv

# CSV -> pgfplots (grafico)
# CSV -> pgfplotstable (tabela)

for DATASET in $DATASETS; do
   for LOSS in $LOSSES; do
        # epsilon=0
        python test.py $DATASET models/model-$DATASET-$LOSS.pth metrics/metrics-$DATASET-$LOSS.csv  # >> xxx.csv
    done
done

for DATASET in $DATASETS; do
    for LOSS in $LOSSES; do
        for ATTACK in $ATTACKS; do
            if [ "$ATTACK" == "GradientSignAttack" ]; then
                for EPSILON in $EPSILONS; do
                    python test.py $DATASET models/model-$DATASET-$LOSS.pth attacks/$ATTACK/eps$EPSILON/attack-$DATASET-$LOSS-$ATTACK-eps$EPSILON.csv --attack $ATTACK --epsilon $EPSILON
                done
            else
                python test.py $DATASET models/model-$DATASET-$LOSS.pth attacks/$ATTACK/attack-$DATASET-$LOSS-$ATTACK.csv --attack $ATTACK
            fi
        done
    done
done