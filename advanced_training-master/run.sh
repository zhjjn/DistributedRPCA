#!/bin/sh

python split.py 0.0155
python split_distributed.py 2 0.0155
python split_distributed.py 8 0.0155
python split_distributed.py 12 0.0155
python split_distributed.py 16 0.0155
python split_distributed.py 20 0.0155
python split_distributed.py 40 0.0155

python split_rpca_randomforest.py
python split_rpca_classfier.py 2
python split_rpca_classfier.py 8
python split_rpca_classfier.py 12
python split_rpca_classfier.py 16
python split_rpca_classfier.py 20
python split_rpca_classfier.py 40

for l in 0.005 0.01 0.0155 0.02 0.025 0.03 0.04 0.05 0.08 0.1 0.12 0.15 0.18 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.95
    do
      python split_lambda_rpca.py $l
done

