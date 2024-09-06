FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime
WORKDIR /workspaces/synthetic-tgnn-benchmark
RUN apt-get update
RUN apt-get -y install git

# For signed commits: https://code.visualstudio.com/remote/advancedcontainers/sharing-git-credentials#_sharing-gpg-keys
RUN apt install gnupg2 -y

# Add utils folder to PATH
ENV PATH="/workspaces/synthetic-tgnn-benchmark/utils:${PATH}"
