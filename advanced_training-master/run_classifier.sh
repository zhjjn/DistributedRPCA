#!/bin/sh

CUDA_VISIBLE_DEVICES=1 python split_rpca_randomforest.py
CUDA_VISIBLE_DEVICES=1 python split_rpca_classfier.py 2
CUDA_VISIBLE_DEVICES=1 python split_rpca_classfier.py 8
CUDA_VISIBLE_DEVICES=1 python split_rpca_classfier.py 12
CUDA_VISIBLE_DEVICES=1 python split_rpca_classfier.py 16
CUDA_VISIBLE_DEVICES=1 python split_rpca_classfier.py 20
CUDA_VISIBLE_DEVICES=1 python split_rpca_classfier.py 40
