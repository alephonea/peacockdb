#!/bin/bash

useradd -m -s /bin/bash build
echo "build ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
mkdir -p /home/build/.ssh
cp /root/.ssh/authorized_keys /home/build/.ssh/
chown -R build:build /home/build/.ssh
chmod 700 /home/build/.ssh
chmod 600 /home/build/.ssh/authorized_keys

apt update

# Cuda toolkit is already present.
apt install -y \
    curl \
    unzip \
    patchelf \
    libssl-dev \
    zlib1g-dev \
    libboost-dev \
    git \
    pkg-config \
    python-is-python3 \
    python3-pip \
    gcc-14 g++-14

# Install cmake 4.2.3 from the official tarball
CMAKE_VERSION=4.2.3
curl -fsSL "https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-x86_64.tar.gz" \
  | tar -xz -C /usr/local --strip-components=1

# Install ninja from the GitHub release
NINJA_VERSION=1.12.1
curl -fsSL "https://github.com/ninja-build/ninja/releases/download/v${NINJA_VERSION}/ninja-linux.zip" \
  -o /tmp/ninja-linux.zip
unzip -o /tmp/ninja-linux.zip -d /usr/local/bin
rm /tmp/ninja-linux.zip
chmod +x /usr/local/bin/ninja

# Install miniforge and cudf via conda for the build user
su - build -c '
  set -e
  curl -fsSL -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
  bash "Miniforge3-$(uname)-$(uname -m).sh" -b -p "$HOME/miniforge3"
  rm "Miniforge3-$(uname)-$(uname -m).sh"
  source "$HOME/miniforge3/etc/profile.d/conda.sh"
  conda config --set channel_priority flexible
  conda create -y -n rapids \
    -c rapidsai -c conda-forge \
    cudf=26.02 \
    libcudf=26.02 \
    python=3.12 \
    "cuda-version>=12.1,<=12.9"
  echo "source \$HOME/miniforge3/etc/profile.d/conda.sh" >> "$HOME/.bashrc"
'
