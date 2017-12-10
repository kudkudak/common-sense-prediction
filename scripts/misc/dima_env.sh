#!/usr/bin/env bash

set -x

export KERAS_BACKEND=theano
export PYTHONPATH=$HOME/Dist/theano:$HOME/Dist/common-sense-prediction
export DATA_DIR=~jastrzes/l2lwe/data
export ACL_ROOT_DIR=$HOME/Dist/ACL_CKBC
export PYTHONPATH=$PYTHONPATH:$ACL_ROOT_DIR
export THEANO_FLAGS=device=cuda,optimizer=fast_run

eval $@
