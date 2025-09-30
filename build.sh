#!/bin/sh
# This script builds the C++ static library for word2vec.
set -e
mkdir -p _build
cd _build
cmake -DCMAKE_BUILD_TYPE=Release ../libw2v
make