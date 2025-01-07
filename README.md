# Reinforcement Learning with Feedback-modulated TD-STDP

This repository is the official implementation of the paper titled ["Reinforcement Learning with Feedback-modulated TD-STDP"](https://arxiv.org/abs/2008.13044).

## Requirements

Only BindsNet, Torch and gym are required to run the code.  To install requirements, run:

```setup
pip install -r requirements.txt
```

## Training

To train the model on Cartpole, run this command:

```train
python main.py -c config_cp.ini
```

To train the model on Cartpole (low firing rate version), run this command:

```
python main.py -c config_cp_lfr.ini
```

To train the model on LunarLander, run this command:

```
python main.py -c config_ll.ini
```

This will load the `config_cp.ini` or `config_cp_lft.ini ` or`config_ll.ini` in `config` folder to run the experiment. By default, 10 runs of training will be done. The result, including plot of learning curve, will be stored in `result` folder, and the trained model, `model_cp_std_0.pt` or `model_cp_lfr_0.pt` or`model_ll_std_0.pt`, will be stored in `model` folder. Edit the config file to adjust training setting. If gpu is not supported, set `gpu` in config file to `False`.

## Evaluation

To evaluate the trained model on Cartpole, run this command:

```eval
python main.py -c config_cp_test.ini
```

To evaluate the trained model on Cartpole (low firing rate version), run this command:

```
python main.py -c config_cp_lfr_test.ini
```

To evaluate the trained model on LunarLander, run this command:

```
python main.py -c config_ll_test.ini
```

This will load `model_cp_std_0.pt` or `model_cp_lfr_0.pt` or `model_ll_std_0.pt` in `model`folder and test the model for 100 episodes. By default, the episode will be visualized. Visualization of episode can be disabled by setting `test_vis` in config file to `False`.

## Pre-trained Models

The pre-trained model, `model_cp_std_0.pt` or `model_cp_lfr_0.pt`or `model_ll_std_0.pt`,  is already stored in `model` folder. 

## Results

Our model has the following result on CartPole and LunarLander. See paper for details of the result. 

|                                                      |  CartPole-v1   |  LunarLander-v2  |
| :--------------------------------------------------- | :------------: | :--------------: |
| First episode to achieve perfect score - Mean (Std.) | 57.00 (23.97)  |  383.80 (71.49)  |
| Episode required to solve the task  - Mean (Std.)    | 169.50 (23.47) | 2575.20 (666.51) |

## Baseline Models

To train the baseline model used in the paper on Cartpole, run this command:

```train
python baseline.py -e cp
```

To train the baseline model used in the paper on LunarLander, run this command:

```
python baseline.py -e ll
```

## Contributing

This software is licensed under the Apache License, version 2 ("ALv2").