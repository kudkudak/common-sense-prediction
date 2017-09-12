#!/usr/bin/env bash

# Fetches data used in http://ttic.uchicago.edu/~kgimpel/papers/li+etal.acl16.pdf

# Execute this script in data folder

set -x
set -e

mkdir -p ACL/conceptnet

wget http://ttic.uchicago.edu/~kgimpel/comsense_resources/train100k.txt.gz
wget http://ttic.uchicago.edu/~kgimpel/comsense_resources/train300k.txt.gz
wget http://ttic.uchicago.edu/~kgimpel/comsense_resources/dev1.txt.gz
wget http://ttic.uchicago.edu/~kgimpel/comsense_resources/dev2.txt.gz
wget http://ttic.uchicago.edu/~kgimpel/comsense_resources/test.txt.gz
wget http://ttic.uchicago.edu/~kgimpel/comsense_resources/embeddings.txt.gz

gunzip train100k.txt.gz
gunzip train300k.txt.gz
gunzip dev1.txt.gz
gunzip dev2.txt.gz
gunzip test.txt.gz
gunzip embeddings.txt.gz

mv train100k.txt train300k.txt dev1.txt dev2.txt test.txt ACL/conceptnet
mv embeddings.txt ACL/conceptnet/embeddings_OMCS.txt

