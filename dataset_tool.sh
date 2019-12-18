#!/bin/bash

#CUDA_VISIBLE_DEVICES=0 python3 dataset_tool.py create_from_numpy datasets/base-feature ./datasets/npy-ffhq/base_feature_maps
#CUDA_VISIBLE_DEVICES=0 python3 dataset_tool.py create_from_numpy datasets/log-feature ./datasets/npy-ffhq/log_feature_maps
CUDA_VISIBLE_DEVICES=0 python3 dataset_tool.py create_from_numpy datasets/normalized-feature ./datasets/npy-ffhq/normalized_feature_maps
