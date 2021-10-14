# Learning One Representation to Optimize All Rewards
This repo contains code for the paper 

[Learning One Representation to Optimize All Rewards.
Ahmed Touati, Yann Ollivier. NeurIPS 2021](https://arxiv.org/pdf/2103.07945.pdf)

## Install Requirements

```bash
pip install 'gym[atari]'
pip install torch
pip install opencv-python
# Baselines for Atari preprocessing
# Tensorflow is a dependency, but you don't need to install the GPU version
conda install tensorflow
pip install git+git://github.com/openai/baselines
# AtariARI (Atari Annotated RAM Interface)
pip install git+git://github.com/mila-iqia/atari-representation-learning.git
```

## Instruction to run the code
If you want to use GPU, just add the flag `--cuda`.
1. train **discrete maze**:
```bash
python grid_main.py \
    --agent FB \
    --n-cycles 25 \
    --n-test-rollouts 10 \
    --num-rollouts-per-cycle 4 \
    --update-eps 1 \
    --soft-update \
    --temp 200 \
    --seed 0 \
    --gamma 0.99 \
    --lr 0.0005 \
    --polyak 0.95 \
    --embed-dim 100 \
    --w-sampling cauchy_ball \
    --n-epochs 200 \
```
2. train **continuous maze**:
```bash
python continuous_main.py \
    --agent FB \
    --n-cycles 25 \
    --n-test-rollouts 10 \
    --num-rollouts-per-cycle 4 \
    --update-eps 1 \
    --soft-update \
    --temp 200 \
    --seed 0 \
    --gamma 0.99 \
    --lr 0.0005 \
    --polyak 0.95 \
    --embed-dim 100 \
    --w-sampling cauchy_ball \
    --n-epochs 200 \
```
3. train **atari**:
```bash
python atari_main.py \
    --agent FB \
    --n-cycles 25 \
    --n-test-rollouts 10 \
    --num-rollouts-per-cycle 2 \
    --update-eps 0.2 \
    --soft-update \
    --temp 200 \
    --seed 0 \
    --gamma 0.9 \
    --lr 0.0005 \
    --polyak 0.95 \
    --embed-dim 100 \
    --w-sampling cauchy_ball \
    --n-epochs 200 \
```
