import torch
from safetensors.torch import save_file
import os
import json
from tqdm import tqdm

model_dir = "/dataset/common/huggingface/model/Qwen3-8B-Tool_DPO"
index_path = os.path.join(model_dir, "pytorch_model.bin.index.json")

# Load index JSON
with open(index_path, "r") as f:
    index = json.load(f)

shard_files = set(index["weight_map"].values())

# Convert all .bin shards to .safetensors
for shard_file in tqdm(shard_files, total=len(shard_files)):
    shard_path = os.path.join(model_dir, shard_file)
    print(f"Converting {shard_path} ...")
    
    # Load .bin
    state_dict = torch.load(shard_path, map_location="cpu")
    
    # Save as .safetensors
    safetensors_path = shard_path.replace(".bin", ".safetensors")
    save_file(state_dict, safetensors_path)
    print(f"Saved {safetensors_path}")
    
    # Delete .bin shard
    os.remove(shard_path)

# Update JSON index to point to .safetensors files instead of .bin
index["weight_map"] = {
    k: v.replace(".bin", ".safetensors") for k, v in index["weight_map"].items()
}

# Save updated index JSON under safetensors name
with open(index_path, "w") as f:
    json.dump(index, f, indent=2)