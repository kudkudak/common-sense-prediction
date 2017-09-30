#!/usr/bin/env bash

# Fetches data used in http://ttic.uchicago.edu/~kgimpel/papers/li+etal.acl16.pdf

# Execute this script in data folder

set -x
set -e

mkdir -p LiACL/conceptnet
mkdir -p embeddings/LiACL

wget http://ttic.uchicago.edu/~kgimpel/comsense_resources/train100k.txt.gz
wget http://ttic.uchicago.edu/~kgimpel/comsense_resources/train300k.txt.gz
wget http://ttic.uchicago.edu/~kgimpel/comsense_resources/dev1.txt.gz
wget http://ttic.uchicago.edu/~kgimpel/comsense_resources/dev2.txt.gz
wget http://ttic.uchicago.edu/~kgimpel/comsense_resources/test.txt.gz
wget http://ttic.uchicago.edu/~kgimpel/comsense_resources/embeddings.txt.gz
wget https://gist.github.com/mnuke/86d8a02c20b24f7f58a50db415babcdf/raw/b5c2d2d70e2bd091ea03ed799b28e041e78c94ed/rel.txt

gunzip train100k.txt.gz
gunzip train300k.txt.gz
gunzip dev1.txt.gz
gunzip dev2.txt.gz
gunzip test.txt.gz
gunzip embeddings.txt.gz

mv train100k.txt train300k.txt dev1.txt dev2.txt test.txt rel.txt LiACL/conceptnet
mv embeddings.txt embeddings/LiACL/embeddings_OMCS.txt

