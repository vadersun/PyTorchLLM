from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import torch
import json
import time
import argparse
import pathlib
from datetime import datetime

# Command line arguments
parser = argparse.ArgumentParser(description="Generate tokens and measure latency")
parser.add_argument("--dtype", type=str, default="bfloat16", help="Data type (default: bfloat16)")
parser.add_argument("--input-tokens", type=int, default=32, help="Number of input tokens (default: 32)")
parser.add_argument("--max-new-tokens", type=int, default=32, help="Maximum number of new tokens to generate (default: 32)")
parser.add_argument("--num-iter", type=int, default=10, help="Number of iterations (default: 10)")
parser.add_argument("--num-warmup", type=int, default=5, help="Number of warmup iterations (default: 5)")
parser.add_argument("--batch-size", type=int, default=1, help="Batch size (default: 1)")
parser.add_argument("--repeats", type=int, default=12, help="Number of repetitions to calculate the average latency (default: 12)")
parser.add_argument(
    "--prompt", default=None, type=str, help="input prompt for self-defined if needed"
)
args = parser.parse_args()

# Set the device and data type
device = torch.device("cpu")
dtype = torch.bfloat16  # data type = float32
dtype = torch.__dict__[args.dtype]

# set model paras
tokenizer = AutoTokenizer.from_pretrained("Llama-2-7b-hf", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("Llama-2-7b-hf", trust_remote_code=True).to(device).to(dtype)
model = model.eval()
if not hasattr(model.config, "token_latency"):
    model.config.token_latency = True

# set input/output tokens and batch size
input_tokens = args.input_tokens
output_tokens = args.max_new_tokens
batch_size = args.batch_size
repeats = 12
if args.repeats<=batch_size and args.repeats > 1:
    repeats = args.repeats

# generate output example
# input prompt
current_path = pathlib.Path(__file__).parent.resolve()
with open(str(current_path) + "/prompt.json") as f:
    prompt_pool = json.load(f)
if args.prompt is not None:
    prompt = args.prompt
elif int(args.input_tokens) > 8192:
    prompt = prompt_pool["llama"]["8192"] * int(int(args.input_tokens) / 8192)
elif str(args.input_tokens) in list(prompt_pool["llama"].keys()):
    print(prompt_pool["llama"][str(args.input_tokens)])
    prompt = prompt_pool["llama"][str(args.input_tokens)]
else:
    print(list(prompt_pool["llama"].keys()))
    print(args.input_tokens)
    raise SystemExit("[ERROR] Plese use --prompt if want to use custom input.")

input_size = tokenizer(prompt, return_tensors="pt").input_ids.size(dim=1)
print("---- Prompt size:", input_size)
# prompt = "Once upon a time, there existed a little girl who liked to have adventures. She wanted to go to places and meet new people, and have fun. She wanted to be a part of something bigger than herself. She wanted to be a part of a community.\nShe wanted to be a part of a"
# Generate tokens and measure latency
num_warmup = args.num_warmup
num_iterations = args.num_iter
vNextTokenList = []
vFristTokenList = []
for j in range(num_iterations):
    with torch.no_grad():
        outputs = []
        first_token_latency = None  # Record the latency of the first token
        nStepSize = int(output_tokens / repeats)
        for i in range(0, output_tokens, nStepSize):
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            input_ids = input_ids.repeat(batch_size, 1)
            start_time = time.time()
            output = model.generate(input_ids, max_new_tokens=i+1, eos_token_id=2, pad_token_id=2)
            end_time = time.time()
            token_latency = (end_time - start_time)
            if i == 0:
                first_token_latency = token_latency
            else:
                token_latency -= first_token_latency
                if token_latency < 0:
                    continue
            outputs.append(output[:, -1])
            if i + nStepSize > output_tokens-1:
                current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
                output = f"[{current_time}] Iteration {j+1} token latency: {token_latency} sec"
                print(output)
            if j >= num_warmup:
                if i != 0:
                    vNextTokenList.append(token_latency/(i+1))
                # print(token_latency/(i+1))

        if j >= num_warmup:
            vFristTokenList.append(first_token_latency)


# calculate the latency
import numpy as np
from itertools import chain
first_latency = np.mean(vFristTokenList)
average_2n_latency = np.mean(vNextTokenList)
average_2n = vNextTokenList
average_2n.sort()
p90_index = int(len(average_2n) * 0.9)
p99_index = int(len(average_2n) * 0.99)
# print(p90_index)
# print(p99_index)
p90_latency = average_2n[p90_index]
p99_latency = average_2n[p99_index]
print("First token average latency: %.3f sec." % first_latency)
print("Average 2... latency: %.3f sec." % average_2n_latency)
print("P90 2... latency: %.3f sec." % p90_latency)
print("P99 2... latency: %.3f sec." % p99_latency)
print("*********************")
