#!/bin/bash

# This script runs as root

while [[ $# -gt 0 ]]; do
  case "$1" in
    --user) TARGET_USER="$2"; shift 2 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

TARGET_USER="${TARGET_USER:-build}"

# Install miniforge and cudf via conda for the build user
su - "$TARGET_USER" -c '
  set -e
  curl -fsSL -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
  bash "Miniforge3-$(uname)-$(uname -m).sh" -b -p "$HOME/miniforge3"
  rm "Miniforge3-$(uname)-$(uname -m).sh"
  source "$HOME/miniforge3/etc/profile.d/conda.sh"
  conda config --set channel_priority flexible
  conda config --set auto_activate_base false
  conda create -y -n rapids \
    -c rapidsai -c conda-forge \
    cudf=26.02 \
    libcudf=26.02 \
    python=3.12 \
    "cuda-version>=12.1,<=12.9"
  echo "source \$HOME/miniforge3/etc/profile.d/conda.sh" >> "$HOME/.bashrc"
'
