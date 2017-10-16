#!/bin/bash
source ~/.bashrc

ssh -N -R 4422:localhost:4422 elisa1&
tensorboard --logdir=results/ --port=4422

