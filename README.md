# Common sense prediction project

*Research objective*: how far are we from common sense prediction: measuring and evaluating.

We base from http://ttic.uchicago.edu/~kgimpel/papers/li+etal.acl16.pdf.

## Resources

* Paper: https://www.overleaf.com/12326026ybwcgtxbxpkr#/46877048/

* ACL paper: http://ttic.uchicago.edu/~kgimpel/papers/li+etal.acl16.pdf

## Setup

0. Project uses primarly python2.7, but should work without issues in python3.

1. Configure PYTHONPATH to include root folder of the project, DATA_DIR to point to data directory, and 
PROJECT_ROOT to point to root of project.

2. Go to DATA_DIR and:
 
    * Run `scripts/data/fetch_LiACL_data.sh`.

    * Run `scripts/data/split_intrinsic_LiACL.py ` (takes like 15 minutes)

    * Optionally run `scripts/data/fetch_and_split_extrinsic_LiACL.py ` for extrinsic evaluation.
    
    * Optionally run `scripts/data/fetch_glove.sh`, but probably you don't need to.

3. (Optional) We provide conda environment. To install it fire `conda env create -f environment-py2.yaml` and add
`source activate common-sense-prediction-py2` to you env file.
For example you can have `env.sh` file that you source before running scripts. In my case it is

```
#!/usr/bin/env bash
export PROJECT_ROOT=$PYTHONPATH:$HOME/l2lwe
export PYTHONPATH=$PYTHONPATH:$HOME/l2lwe
export DATA_DIR=$HOME/l2lwe/data
```

## Datasets

We have following datasets used in project:

* LiACL/conceptnet (Original dataset)
* LiACL/conceptnet_my (Dataset used in the paper based on the original dataset)
* LiACL/conceptnet_my_random (Dataset created by randomly shuffling train/dev/test split)
* LiACL/tuples.wiki  
    * LiACL/tuples.wiki/tuples5k.cn.txt.dev
    * LiACL/tuples.wiki/tuples5k.cn.txt.test
    * LiACL/tuples.wiki/scored_tuples5k.cn.txt.dev

## Training

### Factorized

``python scripts/train_factorized.py root test1``

to train the default configuration of the Factorized model (saves outputs to `test`).

### DNN

``python scripts/train_dnn_ce.py root test2``

to train the default configuration of the DNN+CE model from Li et al. (saves outputs to `test`).

## Evaluation

Use `scripts/evaluate/score_trplets.py` to score using model some triplets (e.g. wiki). Use `scripts/human_evaluate_triplets.py` to
prepare AB tests of two models (for human evaluators) and to process results.

## Data folder structure

* `LiACL/conceptnet_{"my","my_random",""}` - KBC dataset from Li et al models and variants

* `embeddings` - embeddings (kept separately as tend to be big)

## Notes

We use vegab, it is similar to argh, but adds convention that each run necessarily has its own folder, that
after execution will have serialized stdout, stderr, python file and config used.
