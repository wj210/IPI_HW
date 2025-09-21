from utils.utils import *
import torch
from collections import defaultdict
from tqdm import tqdm

def contrast_activations(model,clean,corrupt,bz =-1 ,avg_before_contrast=True):
    clean_acts = defaultdict(list)
    corrupt_acts = defaultdict(list)
    if bz == -1:
        bz = len(clean)
    for i in tqdm(range(0,len(clean),bz),total = len(clean)//bz):
        batch_clean = clean[i:i+bz]
        batch_corrupt = corrupt[i:i+bz]
        with torch.no_grad(), model.trace(batch_clean) as tracer:
            for l in range(len(model.model.layers)):
                clean_acts[l].append(model.model.layers[l].output[0][:,-1].save())
        clear_mem()
        with torch.no_grad(), model.trace(batch_corrupt) as tracer:
            for l in range(len(model.model.layers)):
                corrupt_acts[l].append(model.model.layers[l].output[0][:,-1].save())
        clear_mem()
    clean_acts = {l:torch.cat(v,0) for l,v in clean_acts.items()}
    corrupt_acts = {l:torch.cat(v,0) for l,v in corrupt_acts.items()}
    directions = {}
    for l in range(len(model.model.layers)):
        if avg_before_contrast:
            directions[l] = corrupt_acts[l].mean(0) - clean_acts[l].mean(0)
        else:
            directions[l] = (corrupt_acts[l]- clean_acts[l]).mean(0)
    return directions


def contrast_act_native(model,clean,corrupt,bz =-1 ,avg_before_contrast=True):
    clean_acts = defaultdict(list)
    corrupt_acts = defaultdict(list)
    n = len(clean['input_ids'])
    if bz == -1:
        bz = n
    num_layers = len(model.layers)
    clean_hi = HookedIntervention(model, intervention_fn=None, layers_to_edit=num_layers, capture_post=True)
    corrupt_hi = HookedIntervention(model, intervention_fn=None, layers_to_edit=num_layers, capture_post=True)
    for i in tqdm(range(0,n,bz),total = n//bz):
        batch_clean = {k: v[i:i+bz] for k,v in clean.items()}
        batch_corrupt = {k: v[i:i+bz] for k,v in corrupt.items()}
        with torch.no_grad():
            with clean_hi.activate():
                _ = v_model(**batch_clean)
            with corrupt_hi.activate():
                _ = v_model(**batch_corrupt)
        for l in range(num_layers):
            clean_acts[l].append(clean_hi.io.post[l][:,-1])
            corrupt_acts[l].append(corrupt_hi.io.post[l][:,-1])
        ## empty it out
        clean_hi.io.clear()
        corrupt_hi.io.clear()
        torch.cuda.empty_cache()
       
    clean_acts = {l:torch.cat(v,0).to(model.device) for l,v in clean_acts.items()}
    corrupt_acts = {l:torch.cat(v,0).to(model.device) for l,v in corrupt_acts.items()}
    directions = {}
    for l in range(len(model.model.layers)):
        if avg_before_contrast:
            directions[l] = corrupt_acts[l].mean(0) - clean_acts[l].mean(0)
        else:
            directions[l] = (corrupt_acts[l]- clean_acts[l]).mean(0)
    return directions

def get_steering_vec(model,corrupt,clean,bz=-1,return_separate_vectors = False): # if return_separate_vectors, return corrupt_vec and clean vec as well.
    if bz == -1:
        bz = len(corrupt)
    all_corrupt_acts,all_clean_acts = defaultdict(list),defaultdict(list)
    for i in range(0,len(corrupt),bz):
        corrupt_batch = corrupt[i:i+bz]
        clean_batch = clean[i:i+bz]
        corrupt_inps = encode_fn(model,corrupt_batch,return_attn=True)
        clean_inps = encode_fn(model,clean_batch,return_attn=True)

        _,corrupt_cache = model.run_with_cache(corrupt_inps.input_ids,attention_mask = corrupt_inps.attention_mask,names_filter = resid_name_filter)
        _,clean_cache = model.run_with_cache(clean_inps.input_ids,attention_mask = clean_inps.attention_mask,names_filter = resid_name_filter)

        for k in corrupt_cache.keys():
            all_corrupt_acts[retrieve_layer_fn(k)].append(corrupt_cache[k][:,-1])
            all_clean_acts[retrieve_layer_fn(k)].append(clean_cache[k][:,-1])
        del corrupt_cache,clean_cache
        torch.cuda.empty_cache()
    all_corrupt_acts = {k:torch.concat(v,dim = 0) for k,v in all_corrupt_acts.items()}
    all_clean_acts = {k:torch.concat(v,dim = 0) for k,v in all_clean_acts.items()}
    steering_vec = {k: all_corrupt_acts[k].mean(0)- all_clean_acts[k].mean(0) for k in all_corrupt_acts.keys()}

    if not return_separate_vectors:
        return steering_vec
    else:
        return steering_vec, (all_corrupt_acts,all_clean_acts)