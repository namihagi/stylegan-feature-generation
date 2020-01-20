#!/bin/bash

#CUDA_VISIBLE_DEVICES=0 python3 dataset_tool.py create_from_numpy datasets/base-feature ./datasets/npy-ffhq/base_feature_maps
#CUDA_VISIBLE_DEVICES=0 python3 dataset_tool.py create_from_numpy datasets/log-feature ./datasets/npy-ffhq/log_feature_maps
#CUDA_VISIBLE_DEVICES=0 python3 dataset_tool.py create_from_numpy datasets/normalized-feature ./datasets/npy-ffhq/normalized_feature_maps

#for ClassName in $(ls /hagio/datasets/voc2007/each_class_set/train -1);
#do
#  echo "$ClassName"
#  CUDA_VISIBLE_DEVICES=0 python3 dataset_tool.py create_from_images datasets/voc-$ClassName /hagio/datasets/voc2007/each_class_set/train/$ClassName/resized_images
#done

# wider face with annotations
#CUDA_VISIBLE_DEVICES=0 python3 dataset_tool.py [module_name] [output_dir] [npy_feature_dir] [annotation_dir]
CUDA_VISIBLE_DEVICES=0 python3 dataset_tool.py \
                            create_from_feature_with_labels \
                            datasets/wider-face-with-annotations \
                            /hagio/ObjectDetection_with_MeanTeacher/result/wider_face_2/source_feature_maps \
                            /hagio/ObjectDetection_with_MeanTeacher/datasets/face/val/annotations


