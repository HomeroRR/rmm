#!/bin/sh

wget https://cvdn.dev/dataset/CVDN/train_val/train.json -P tasks/CVDN/data/
wget https://cvdn.dev/dataset/CVDN/train_val/val_seen.json -P tasks/CVDN/data/
wget https://cvdn.dev/dataset/CVDN/train_val/val_unseen.json -P tasks/CVDN/data/