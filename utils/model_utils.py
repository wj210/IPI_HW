from functools import partial
from transformers import AutoTokenizer,AutoModelForCausalLM
from vllm import LLM, SamplingParams
import torch
from aside.model_api import CustomModelHandler
from aside.embeddings_init import generate_isoclinic_rotation_matrix
import os,json

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
    if 'qwen' not in model_path.lower():
        print (f'Ensure the ctx len is loaded, default is {max_ctx_len}')
    
    if is_aside: # can only use hf model
        handler = CustomModelHandler(
                model_path, None, None, model_path, None,
                0, embedding_type='forward_rot' if 'aside' in model_path.lower() else 'ise',
                load_from_checkpoint=True,model_dtype=dtype,max_token_len=max_ctx_len,
            )
        # model = handler.model.to(device).eval() # dont pass to model, i just need the model itself.
        embedding_model = handler.model.get_input_embeddings().to(device).eval() # manually get the embeddings
        enable_prompt_embeds=True
        tokenizer = handler.tokenizer
        print (f'Embedding type {handler.embedding_type}')
        if handler.embedding_type == 'forward_rot':
            hidden_size = handler.model.config.hidden_size
            rotation_matrix = generate_isoclinic_rotation_matrix(
                hidden_size, handler.model.global_rotation_alpha,embedding_model.weight.device,dtype
            ).detach()

            def init_embed_fn(encoded,input_lens,embedding_layer,rotation_matrix):
                input_ids = encoded['input_ids']
                segment_ids = encoded['segment_ids']
                device = embedding_layer.weight.device
                rotation_matrix = rotation_matrix.to(device)

                embeds = embedding_layer(input_ids.to(device)) # [B,T,D]
                masks = (segment_ids.to(device)==1) # [B,T]
                embed_out = embeds.clone()
                embed_out[masks] = torch.matmul(embeds[masks],rotation_matrix)
                unpadded_embed = []
                for e,l in zip(embed_out,input_lens):
                    unpadded_embed.append(e[:l].cpu().float())
                return [{'prompt_embeds':e} for e in unpadded_embed]

            init_fn = partial(init_embed_fn,embedding_layer=embedding_model,rotation_matrix=rotation_matrix)
        else: # ise
            segment_embedding = handler.segment_embedding
            pass
                
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        enable_prompt_embeds = False
        embedding_model=None
        init_fn=lambda x:x
    
    if use_vllm:
        num_devices = torch.cuda.device_count()
        vllm_kwargs  = {**DEFAULT_VLLM_KWARGS,**vllm_kwargs}
        model = LLM(model_path,max_model_len = max_ctx_len,tensor_parallel_size=num_devices,enable_prompt_embeds=enable_prompt_embeds,**vllm_kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.float16 if 'awq' in model_path.lower() else dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to(device).eval()
        init_fn = lambda x:x # no need to turn to embedding.
           
    if not use_vllm:
        model.generation_config.temperature = None
        model.generation_config.top_p = None
        model.generation_config.top_k = None
    tokenizer.padding_side = "left"
    tokenizer.model_max_length=max_ctx_len
    if not tokenizer.pad_token:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model.tokenizer = tokenizer # for convenience
    model.use_vllm = use_vllm
    if pass_handler and is_aside:
        return model,tokenizer,is_aside,init_fn,handler
    else:
        return model,tokenizer,is_aside,init_fn