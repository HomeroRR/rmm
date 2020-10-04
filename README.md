# RMM: A Recursive Mental Model for Dialogue Navigation

This repository contains code for the paper [**RMM: A Recursive Mental Model for Dialog Navigation**](https://arxiv.org/abs/2005.00728).

```bibtex
@inproceedings{romanroman:EMNLP-Findings20,
  title={RMM: A Recursive Mental Model for Dialog Navigation},
  author={Homero Roman Roman and Yonatan Bisk and Jesse Thomason and Asli Celikyilmaz and Jianfeng Gao},
  booktitle={Findings of the 2020 Conference on Empirical Methods in Natural Language Processing},
  year={2020}
}
```

## Installation / Build Instructions

This repository is built from the [Matterport3DSimulator](https://github.com/peteanderson80/Matterport3DSimulator) codebase. The original installation instructions are included at [`README_Matterport3DSimulator.md`](README_Matterport3DSimulator.md). In this document we outline the instructions necessary to work with the CVDN task.

We recommend using the mattersim [Dockerfile](Dockerfile) to install the simulator. The simulator can also be [built without docker](https://github.com/peteanderson80/Matterport3DSimulator#building-without-docker) but satisfying the project dependencies may be more difficult.

### Prerequisites

- Ubuntu 16.04
- Nvidia GPU with driver >= 384
- Install [docker](https://docs.docker.com/engine/installation/) with gpu support
- Note: CUDA / CuDNN toolkits do not need to be installed (these are provided by the docker image)

### Building using Docker

Build the docker image:
```
docker build -t cvdn .
```

Run the docker container, mounting both the git repo and the dataset:
```
docker run -it --volume `pwd`:/root/mount/Matterport3DSimulator -w /root/mount/Matterport3DSimulator cvdn
```

### CVDN Dataset Download

Download the `train`, `val_seen`, `val_unseen`, and `test` splits of the whole CVDN dataset by executing the following script:
```
tasks/CVDN/data/download.sh
```

### Matterport3D Dataset Download

To use the simulator you must first download the [Matterport3D Dataset](https://niessner.github.io/Matterport/) which is available after requesting access [here](https://niessner.github.io/Matterport/). The download script that will be provided allows for downloading of selected data types.  

The experiments rely on the ResNet-152-imagenet features which must be pre-processed before hand.


Pre-processed features can be obtained as follows:
```
mkdir -p img_features/
cd img_features/
wget https://www.dropbox.com/s/o57kxh2mn5rkx4o/ResNet-152-imagenet.zip?dl=1 -O ResNet-152-imagenet.zip
unzip ResNet-152-imagenet.zip
cd ..
```

## Train and Evaluate

### Pretraining
Pretraining is done using the classic [speaker follower](https://github.com/ronghanghu/speaker_follower) setup.

Agent pretraining:
```
python src/train.py --train_datasets=CVDN --eval_datasets=CVDN
```
Speaker pretraining:
```
python src/train.py --entity=speaker --train_datasets=CVDN --eval_datasets=CVDN
```
Pre-trained models are already included in [results/baseline/CVDN_train_eval_CVDN/G1/v1/steps_4](results/baseline/CVDN_train_eval_CVDN/G1/v1/steps_4)

### Training and evaluating RMM
To train RMM with single branch evaluation run the following command:
```
python src/train.py --mode=gameplay --rl_mode=agent_speaker --train_datasets=CVDN --eval_datasets=CVDN
```

And to train RMM with multiple branch evaluation using the Action Probabilities, run the following command:
```
python src/train.py --mode=gameplay --eval_branching=3 --action_probs_branching --train_datasets=CVDN --eval_datasets=CVDN
```

Results are by default saved in 

[results/gameplay/CVDN_train_eval_CVDN/G1/v1/steps_4/agent_rl_speaker_rl/agent_sample_speaker_sample](results/gameplay/CVDN_train_eval_CVDN/G1/v1/steps_4/agent_rl_speaker_rl/agent_sample_speaker_sample)

`val_unseen_gps.csv` will contain the Goal Progresses for all the evaluation entries at each time step a question is asked as well as a final goal progress for that entry.

### Optional functionality
Including the flag `--target_only` indicates the agent to not ask questions and only use the target as textual guidance. Similarly, including the flag `--current_q_a_only` indicates that the agent will only use the latest question-answer pair and discard its dialogue history. 

## Acknowledgements

This repository is built upon the [Matterport3DSimulator](https://github.com/peteanderson80/Matterport3DSimulator) codebase.


The CVDN dataset was collected by Thomason et al. as outlined in the paper [**Vision-and-Dialog Navigation**](https://arxiv.org/abs/1907.04957)
