#!/bin/sh

CUDA_VISIBLE_DIVECES=0 python split_distributed.py 80 0.0155
CUDA_VISIBLE_DIVECES=0 python split_distributed.py 120 0.0155
CUDA_VISIBLE_DIVECES=0 python split_distributed.py 240 0.0155
CUDA_VISIBLE_DIVECES=0 python split_distributed.py 360 0.0155
CUDA_VISIBLE_DIVECES=0 python split_distributed.py 400 0.0155
CUDA_VISIBLE_DIVECES=0 python split_distributed.py 440 0.0155
CUDA_VISIBLE_DIVECES=0 python split_distributed.py 600 0.0155
CUDA_VISIBLE_DIVECES=0 python split.py 0.0155
CUDA_VISIBLE_DIVECES=0 python split_distributed.py 2 0.0155
CUDA_VISIBLE_DIVECES=0 python split_distributed.py 8 0.0155
CUDA_VISIBLE_DIVECES=0 python split_distributed.py 12 0.0155
CUDA_VISIBLE_DIVECES=0 python split_distributed.py 16 0.0155
CUDA_VISIBLE_DIVECES=0 python split_distributed.py 20 0.0155
CUDA_VISIBLE_DIVECES=0 python split_distributed.py 40 0.0155
