# Common sense prediction project

*Research question*: how far are we from predicting common sense knowledge (from raw text, but not limited to)?

*Extra research question*: how can we further SOTA in word representation learning in terms of representing common
sense knowledge?

We base on http://ttic.uchicago.edu/~kgimpel/papers/li+etal.acl16.pdf.

## Resources

* Write-up: https://www.overleaf.com/10600980csjksczhwpgb

* ACL paper: http://ttic.uchicago.edu/~kgimpel/papers/li+etal.acl16.pdf

* Note: https://drive.google.com/drive/folders/0B8-M39RV4diKVlBvTEZidnJNYm8 (or this link https://drive.google.com/open?id=0B8-M39RV4diKVlBvTEZidnJNYm8)

## Setup

0. Project uses primarly python2.7, but should work without issues in python3.

1. Configure PYTHONPATH to include root folder of the project. Configure DATA_DIR to point to data directory

2. Go to DATA_DIR and run `scripts/data/fetch_LiACL_data.sh`.

    * Optionally run `scripts/data/fetch_and_split_wiki.py ` for extrinsic evaluation.
    
    * Optionally run `scripts/data/fetch_glove.sh`, but probably you don't need to.

3. (Optional) We provide conda environment. To install it fire `conda env create -f environment-py2.yaml` and add
`source activate common-sense-prediction-py2` to you env file.
For example you can have `env.sh` file that you source before running scripts. In my case it is

```
#!/usr/bin/env bash

export PYTHONPATH=$PYTHONPATH:$HOME/l2lwe
export DATA_DIR=$HOME/l2lwe/data
export ACL_ROOT_DIR=$HOME/l2lwe/ACL_CKBC
export PYTHONPATH=$PYTHONPATH:$ACL_ROOT_DIR
```

## Data

* `LiACL/conceptnet` - all files used in ACL models

* `embeddings` - embeddings (kept separately as tend to be big)

* `glove embeddings` - GloVe embeddings

## Notes

We use vegab, it is similar to argh, but adds convention that each run necessarily has its own folder, that
after execution will have serialized stdout, stderr, python file and config used.
