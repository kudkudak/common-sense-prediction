# Common sense prediction project

This is the codebase for the paper ["Commonsense mining as knowledge base completion? A study on the impact of novelty"](https://arxiv.org/pdf/1804.09259.pdf)

*Research objective*: how far are we from real common sense prediction measuring and evaluation.

Our paper extends the work ["Commonsense Knowledge Base Completion"](http://ttic.uchicago.edu/~kgimpel/papers/li+etal.acl16.pdf) by Li et al (ACL 2016)


## Setup

1. Clone the repo `git clone git@github.com:kudkudak/common-sense-prediction.git`

2. Setup the dependencies
    * *recommended*: using `conda`, install the environment for python 2 or python 3 `conda env create -f environment-pyX.yaml`
    * otherwise install dependencies with `pip`, `pip install -r requirements.txt`

3. Configure environment setup `env.sh` and make executable `chmod +x env.sh` and run `./env.sh`
    * `PYTHONPATH` should point to the root of the project
    * `DATA_DIR` should point to data directory
    * `PROJECT_ROOT` should point to the root of the project
    * *if using conda or similar* `source activate` the correct environment

e.g.
```
#!/usr/bin/env bash
export PROJECT_ROOT=$PYTHONPATH:$HOME/common-sense-prediction
export PYTHONPATH=$PYTHONPATH:$HOME/common-sense-prediction
export DATA_DIR=$PROJECT_ROOT/data
```

4. In `DATA_DIR`
    * Run `PATH/TO/scripts/data/fetch_LiACL_data.sh`.
    * Run `PATH/TO/scripts/data/split_intrinsic_LiACL.py ` (takes ~15 minutes)
    * (Optional) to do extrinsic eval you need wikipedia and conceptnet tuples: run `PATH/TO/scripts/data/fetch_and_split_extrinsic_LiACL.py `
    * (Optional) not currently needed `PATH/TO/scripts/data/fetch_glove.sh`

## Datasets

We have following datasets used in project:

* `LiACL/conceptnet` (Original dataset)
* `LiACL/conceptnet_my` (Dataset used in the paper based on the original dataset)
* `LiACL/conceptnet_my_random` (Dataset created by randomly shuffling train/dev/test split)
* `LiACL/tuples.wiki`
    * `tuples5k.cn.txt.dev`
    * `tuples5k.cn.txt.test`
    * `scored_tuples5k.cn.txt.dev`

## Training

intialize the environment with `./env.sh`

### Factorized

``python scripts/train_factorized.py root test1``

to train the default configuration of the Factorized model (saves outputs to `test`).

### DNN

``python scripts/train_dnn_ce.py root test2``

to train the default configuration of the DNN+CE model from Li et al. (saves outputs to `test`).

## Evaluation

Use `scripts/evaluate/score_triplets.py` to score using model some triplets (e.g. wiki). Use `scripts/human_evaluate_triplets.py` to
prepare AB tests of two models (for human evaluators) and to process results.

## Data folder structure

* `LiACL/conceptnet_{"my","my_random",""}` - KBC dataset from Li et al models and variants

* `embeddings` - embeddings (kept separately as tend to be big)

## Notes

We use vegab, it is similar to argh, but adds convention that each run necessarily has its own folder, that
after execution will have serialized stdout, stderr, python file and config used.
