import os, glob
from safetensors.torch import safe_open, save_file

src_dir = "/dataset/common/huggingface/model/Qwen3-8B-Tool_ASIDE_SFT"
for f in glob.glob(os.path.join(src_dir, "*.safetensors")):
    tensors = {}
    removed = 0
    with safe_open(f, framework="pt") as sf:
        for k in sf.keys():
            if "rotation_matrix" in k:
                removed += 1
                continue
            tensors[k] = sf.get_tensor(k)
    out = os.path.join(src_dir, os.path.basename(f))
    save_file(tensors, out)
    print(f"Saved {out}, removed {removed} keys")

