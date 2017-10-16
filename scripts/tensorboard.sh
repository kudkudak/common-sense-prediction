#!/bin/bash
ssh -N -R 4422:localhost:4422 elisa&
tensorboard --logdir=results/ --port=4422

