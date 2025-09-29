from transformers import AutoTokenizer,AutoModelForCausalLM
from vllm import LLM, SamplingParams
import torch
from aside.model_api import CustomModelHandler
import os

DEFAULT_VLLM_KWARGS = {
    "gpu_memory_utilization": 0.9,
    'enable_chunked_prefill': False,
}

def load_model(model_path,use_vllm=False,dtype=torch.bfloat16,device = 'cuda',max_ctx_len = 32768,vllm_kwargs={},pass_handler=False):
    """
    Loads a LLM either using Huggingface or vLLM based on the parameters.
    ASIDE only works with Huggingface models, since we require access to it's embeddings
    """
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    is_aside = True if 'ASIDE' in model_path or 'ISE' in model_path else False
    if is_aside: # not compatible with vllm
        use_vllm = False
        print ("ASIDE model detected, disabling vLLM")
    if 'qwen' not in model_path.lower():
        print (f'Ensure the ctx len is loaded, default is {max_ctx_len}')
    
    if is_aside: # can only use hf model
        handler = CustomModelHandler(
                model_path, None, None, model_path, None,
                0, embedding_type='forward_rot' if 'aside' in model_path.lower() else 'ise',
                load_from_checkpoint=True,model_dtype=dtype,max_token_len=max_ctx_len,
            )
        model = handler.model.to(device).eval() # dont pass to model, i just need the model itself.
        tokenizer = handler.tokenizer
        tokenizer.model_max_length=max_ctx_len
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if not use_vllm:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                dtype=torch.float16 if 'awq' in model_path.lower() else dtype,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            ).to(device).eval()
        else:
            num_devices = torch.cuda.device_count()
            vllm_kwargs  = {**DEFAULT_VLLM_KWARGS,**vllm_kwargs}
            model = LLM(model_path,max_model_len = max_ctx_len,tensor_parallel_size=num_devices,**vllm_kwargs)
    if not use_vllm:
        model.generation_config.temperature = None
        model.generation_config.top_p = None
        model.generation_config.top_k = None
    tokenizer.padding_side = "left"
    if pass_handler and is_aside:
        return model,tokenizer,is_aside,handler
    else:
        return model,tokenizer,is_aside