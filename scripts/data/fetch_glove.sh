#!/usr/bin/env bash

# Fetches glove pre-trained embeddings (Common Crawl 42B and 840B tokens)
set -x
set -e

mkdir -p  embeddings/glove

wget -nc http://nlp.stanford.edu/data/wordvecs/glove.42B.300d.zip

wget -nc http://nlp.stanford.edu/data/wordvecs/glove.840B.300d.zip

unzip glove.42B.300d.zip -d embeddings/glove/
unzip glove.840B.300d.zip -d embeddings/glove/

