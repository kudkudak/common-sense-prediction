# Common sense prediction project

This is the codebase for ["Commonsense mining as knowledge base completion? A study on the impact of novelty"](https://arxiv.org/pdf/1804.09259.pdf).

If you wish to use our evaluation, see Evaluation part of this README.

## Setup

1. Clone the repo `git clone git@github.com:kudkudak/common-sense-prediction.git`

2. Setup the dependencies
    * *recommended*: using `conda`, install the environment for python 2 or python 3 `conda env create -f environment-pyX.yaml`
    * otherwise install dependencies with `pip`, `pip install -r requirements.txt`

3. Configure environment setup `env.sh` and source the environment `source env.sh`
    * `PYTHONPATH` should point to the root of the project
    * `DATA_DIR` should point to data directory
    * `PROJECT_ROOT` should point to the root of the project

4. In `DATA_DIR`
    * Run `bash $PROJECT_ROOT/scripts/data/fetch_LiACL_data.sh`.
    * Run `python $PROJECT_ROOT/scripts/data/split_intrinsic_LiACL.py ` (takes ~15 minutes)
    * (Optional) to do extrinsic eval you need wikipedia and conceptnet tuples: run `PATH/TO/scripts/data/fetch_and_split_extrinsic_LiACL.py `
    * (Optional) `PATH/TO/scripts/data/fetch_glove.sh`

## Training

Intialize the environment with `./env.sh`

### Factorized
To train the Factorized model with `root` configuration and save outputs to folder `factorized` run:

``python scripts/train_factorized.py root factorized``

### DNN
To train DNN+CE model (from Li et al. 2016) with 'root' configuration and save outputs to folder `dnn` run: 

``python scripts/train_dnn_ce.py root dnn``


## Evaluation

To generate the table of F1 scores bucketed by the distance of the tests to the training set (using the novelty heuristic) run:

``python scripts/report.py /path/to/dnn /path/to/factorized``


## Data

We have following datasets used in project:

* `embeddings/LiACL/`
    * `embeddings_OMCS.txt` Open Mind Common Sense embeddings used in (Li et al, 2016)
* `LiACL/conceptnet` Original dataset
    * `train100k.txt`
    * `train300k.txt`
    * `train600k.txt`
    * `dev1.txt`
    * `dev2.txt`
    * `test.txt`
    * `rel.txt` list of relations
    * `rel_lowercase.txt`
* `LiACL/conceptnet_my` Dataset used in the paper (Original dataset resplit randomly)
    * `{test/dev}.dists` distance of each tuple from training data using novelty heuristic
* `LiACL/conceptnet_my_random` Extra dataset created by randomly shuffling train/dev/test split
* (Optional) `LiACL/tuples.wiki` Wikipedia tuples
    * `allrel.txt`
    * `allrel.txt.dev`
    * `allrel.txt.test`
    * `top100.txt`
    * `top100.txt.dev`
    * `top100.txt.test`
    * `top10k.txt`
    * `top10k.txt.dev`
    * `top10k.txt.test`
    ...
* (Optional) `LiACL/tuples.cn` Conceptnet tuples
    * `tuples.cn.txt`
    * `tuples.cn.txt.dev`
    * `tuples.cn.txt.test`
    ...


## Notes

* We use vegab. This is similar to argh, but adds convention that each run necessarily has its own folder, that
after execution will have serialized stdout, stderr, python file and config used.

* use `scripts/evaluate/score_triplets.py` to score using model some triplets (e.g. wiki). Use `scripts/human_evaluate_triplets.py` to
prepare AB tests of two models (for human evaluators) and to process results.
