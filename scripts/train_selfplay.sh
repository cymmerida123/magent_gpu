#!/bin/sh
env="SingleCombat"
scenario="selfplay"
model='F16'
algo="ppo"
exp="v1"
seed=5
device="cuda:1"

echo "env is ${env}, scenario is ${scenario}, model is ${model}, algo is ${algo}, exp is ${exp}, seed is ${seed}"
python train/train_F16sim.py \
    --env-name ${env} --algorithm-name ${algo} --scenario-name ${scenario} --model-name ${model} --experiment-name ${exp} \
    --seed ${seed} --device ${device} --n-training-threads 1 --n-rollout-threads 3000 --cuda \
    --log-interval 1 --save-interval 1 \
    --use-eval --eval-interval 10 --eval-episodes 5\
    --num-mini-batch 5 --buffer-size 1000 --num-env-steps 2.25e9 --use-selfplay --selfplay-algorithm "pfsp"\
    --lr 3e-4 --gamma 0.99 --ppo-epoch 16 --clip-params 0.2 --max-grad-norm 2 --entropy-coef 1e-3 \
    --hidden-size "128 128" --act-hidden-size "128 128" --recurrent-hidden-size 128 --recurrent-hidden-layers 1 --data-chunk-length 8