#!/bin/bash
MODELS="DHCI FFB FGNET"
LOSSES="CrossEntropy BinomialUnimodal_CE CO2 UnimodalNet"
ATTACKS="GradientSignAttack"
EPSILONS="0.05 0.1 0.15 0.2 0.25 0.3"

for MODEL in $MODELS; do
    for LOSS in $LOSSES; do
        for ATTACK in $ATTACKS; do
            for EPSILON in $EPSILONS; do
                sbatch python test.py $MODEL models/model-$MODEL-$LOSS.pth attacks/$ATTACK/targeted/eps$EPSILON/attack-$MODEL-$LOSS-$ATTACK-eps$EPSILON.txt --attack $ATTACK --epsilon $EPSILON --targeted true
            done
        done
    done
done