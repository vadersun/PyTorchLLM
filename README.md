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
DOCKER_BUILDKIT=1 docker build -f LLM_Torch_Dockerfile --network=host --no-cache --build-arg COMPILE=ON -t torch-llm:2.1.100 .\
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
You can run LLM with a Python script "llama.py/chatglm.py/..." for all inference cases.
```bash
python llama-2-7b-hf.py --help # for more detailed usages
python chatglm-2-6b.py --help # for more detailed usages
```

| Key Args        | Notes           |
| ------------- |-------------|
| `--dtype`      | Data type (default: bfloat16). |
| `--input-tokens`      | Number of input tokens (default: 32). |
| `--max-new-tokens` | Maximum number of new tokens to generate (default: 32). |
| `--num-iter` | Number of iterations (default: 10). |
| `--num-warmup` | Number of warmup iterations (default: 5). |
| `--batch-size` | Batch size (default: 1). |
| `--repeats` | Number of repetitions to calculate the average latency (default: 12). |
| `--prompt` | Input prompt for self-defined if needed. |

Note: You may need to log in your HuggingFace account to access the model files. Please refer to HuggingFace login.

## Example usages of Python script
### Quick start example commands for benchmarking
```bash
# The following "OMP_NUM_THREADS" and "numactl" settings are based on the assumption that
# the target server has 48 physical cores per numa socket, and we benchmark with 1 socket.
# Please adjust the settings per your hardware.

# Running FP32 Llama-2-7b model
OMP_NUM_THREADS=48 numactl -m 0 -C 0-47 python llama-2-7b-hf.py --dtype float32

# Running BF16 Llama-2-7b model
OMP_NUM_THREADS=48 numactl -m 0 -C 0-47 python llama-2-7b-hf.py --dtype bfloat16

# Running BF16 Llama-2-7b model, input/output tokens = 64, batch size = 8
OMP_NUM_THREADS=48 numactl -m 0 -C 0-47 python llama-2-7b-hf.py --input-tokens 64 --max-new-tokens 64 --batch-size 8 --dtype bfloat16

# Running BF16 Llama-2-7b model, input/output tokens = 64, batch size = 8, perform calculations 4 times and average the results for a faster iteration
OMP_NUM_THREADS=48 numactl -m 0 -C 0-47 python llama-2-7b-hf.py --input-tokens 64 --max-new-tokens 64 --batch-size 8 --dtype bfloat16 --repeats 4

# Running FP32 chatglm-2-6b model
OMP_NUM_THREADS=48 numactl -m 0 -C 0-47 python chatglm-2-6b.py --input-tokens 32 --max-new-tokens 32 --batch-size 1 --dtype float32

# Running BF16 chatglm-2-6b model
OMP_NUM_THREADS=48 numactl -m 0 -C 0-47 python chatglm-2-6b.py --input-tokens 32 --max-new-tokens 32 --batch-size 1 --dtype bfloat16
```
### Example output
Latency results for each iteration:
```bash
---- Prompt size: 512
[2024-01-10-06-25-00] Iteration 1 token latency: 97.7842948436737 sec
[2024-01-10-06-36-47] Iteration 2 token latency: 106.43137431144714 sec
[2024-01-10-06-48-45] Iteration 3 token latency: 97.99792194366455 sec
[2024-01-10-07-01-19] Iteration 4 token latency: 116.03425550460815 sec
[2024-01-10-07-13-20] Iteration 5 token latency: 93.38325834274292 sec
[2024-01-10-07-25-19] Iteration 6 token latency: 110.51585626602173 sec
[2024-01-10-07-37-04] Iteration 7 token latency: 108.74671125411987 sec
[2024-01-10-07-48-54] Iteration 8 token latency: 107.04621076583862 sec
[2024-01-10-08-01-07] Iteration 9 token latency: 108.75247311592102 sec
[2024-01-10-08-12-41] Iteration 10 token latency: 93.31934213638306 sec
```
Summary of the latency results:
```bash
# First token average latency
First token average latency: 1.807 sec.
# Token average latency
Average 2... latency: 0.332 sec.
# The 90th percentile (P90) latency for next tokens
P90 2... latency: 0.214 sec.
# The 99th percentile (P99) latency for next tokens
P99 2... latency: 1.818 sec.
```

Note: The calculation methods for the two types of latency are as follows: Set the output token count to 1, generate the model's output, and record the time taken for generating the first token as the latency of the first token. Next, set the output token count to a fixed value, generate the model's output, subtract the time taken for the first token, and calculate the average latency time for the remaining tokens. Finally, this average latency time can be used as the latency for the next token.
