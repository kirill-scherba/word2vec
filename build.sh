#!/bin/sh
# This script builds the C++ static library for word2vec.
set -e

mkdir -p _build
cd _build
cmake -DCMAKE_ARCHIVE_OUTPUT_DIRECTORY=$1 -DCMAKE_BUILD_TYPE=Release $2
make
cd ..
rm -rf _build
