#!/bin/sh

for l in 0.08 0.1 0.12 0.15 0.18 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.95
        do
                  CUDA_VISIBLE_DIVECES=5 python split_lambda_rpca.py $l
done
