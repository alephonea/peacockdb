#!/bin/bash

set -e

HOSTNAME="build-dev"

rsync -P -av --prune-empty-dirs --include "*/" --include "*.json" --exclude "*" $HOSTNAME:~/peacockdb/cpp/build/compile_commands.json cpp/

rsync -P -av --prune-empty-dirs --include "*/" --include "*.h" --include "*.hpp" --include "*.cuh" --include "*.inl" --include "*.inc" --include "*.modmap" --exclude "*" $HOSTNAME:~/peacockdb/cpp/build cpp/
