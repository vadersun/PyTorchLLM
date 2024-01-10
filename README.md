# PyTorchLLM Testing
## Introduction
This is a testing script for LLM models (such as Llama-2-7b, Chatglm-2-6b) based on the PyTorch library. It calculates the latency between the generation of the First Token and the Next Token for different models. This is performed to compare and verify the performance of models optimized using the IPEX library.

## Environment Setup
1. Get the source code
```bash
git clone https://github.com/vadersun/PyTorchLLM.git
cd PyTorchLLM
```
2. It is highly recommended to build a Docker container from the provided Dockerfile ("LLM_Torch_Dockerfile").
```bash
# Build an image with the provided Dockerfile
DOCKER_BUILDKIT=1 docker build -f /home/kanli/zhenhui/LLM_Torch_Dockerfile --network=host --no-cache --build-arg COMPILE=ON -t torch-llm:2.1.100 .\
# Run the container with command below
docker run --rm -it --privileged torch-llm:2.1.100 bash
# When the command prompt shows inside the docker container, enter llm examples directory
cd llm
```
You can also configure a Python environment using conda.
```bash
# Create a conda environment
conda create -n llm python=3.10 -y
conda activate llm
# Install PyTorch and Transformers
pip install torch==2.1.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
pip install transformers==4.31.0
pip install sentencepiece
```

## Run Models Generations
