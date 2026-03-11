#!/bin/bash

useradd -m -s /bin/bash build
echo "build ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
mkdir -p /home/build/.ssh
cp /root/.ssh/authorized_keys /home/build/.ssh/
chown -R build:build /home/build/.ssh
chmod 700 /home/build/.ssh
chmod 600 /home/build/.ssh/authorized_keys

apt update
apt install -y 
    libssl-dev \
    zlib1g-dev \
    libboost-dev \
    git \
    pkg-config \
    python-is-python3 \
    python3-pip \
    gcc-14 g++-14 \
    cuda-toolkit-12-6

sudo build pip install --break-system-packages cmake
sudo build pip install --break-system-packages ninja
