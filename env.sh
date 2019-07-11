#!/usr/bin/bash

source activate csp
export KERAS_BACKEND=tensorflow
export PROJECT_ROOT=$HOME/common-sense-prediction
export PYTHONPATH=$PYTHONPATH:$PROJECT_ROOT
export DATA_DIR=$PROJECT_ROOT/data

