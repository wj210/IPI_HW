import torch.nn.functional as F
from einops import einsum
import torch
import numpy as np
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import string
from rich.console import Console
from rich.panel import Panel
import gc
from collections import defaultdict
from copy import deepcopy
from functools import partial
from transformer_lens.past_key_value_caching import HookedTransformerKeyValueCache

resid_name_filter = lambda x: 'resid_post' in x
attn_z_name_filter = lambda x: 'attn.hook_z' in x
retrieve_layer_fn = lambda x: int(x.split('.')[1])

def pprint(text):
    """Pretty print the model's generated text using rich."""
    console = Console(width=100)
    console.print(Panel(text, border_style="blue"))

def clear_mem():
    torch.cuda.empty_cache()
    gc.collect()

def format_prompt(tokenizer,instr,info=None,in_system=False):
    if 'gemma' not in tokenizer.name:
        if info is None:
            if not in_system:
                formatted_prompt = [
                {'role':'user','content':instr},
                    ]
            else:
                formatted_prompt = [
                {'role':'system','content':instr},
                    ]
        else:
            formatted_prompt = [
                {'role':'system','content':instr},
                {'role':'user','content':info},
            ]
    else:
        if info is None:
            formatted_prompt = [
                {'role':'user','content':instr},
                    ]
        else:
            formatted_prompt = [
                {'role':'user','content':f"System: {instr}\nUser: {info}"},
                    ]

    return tokenizer.apply_chat_template(formatted_prompt,tokenize=False,add_generation_prompt=True)


def encode_fn(model,inputs,return_attn=False):
    tokenized = model.tokenizer(inputs,padding = 'longest',truncation=False,return_tensors='pt')
    if not return_attn:
        tokenized = tokenized['input_ids']
    return tokenized.to(model.cfg.device)

def custom_generate(model,inputs,attention_mask = None,max_new_tokens = 256):
    """
    Normal TL generate function seems buggy for padded inputs, something to do with the attention mask.
    if return_loss, return CE loss = -logprob (argmax)
    """
    bz = inputs.shape[0]
    past_kv_cache = HookedTransformerKeyValueCache.init_cache(model.cfg, model.cfg.device, bz)
    finished_sequences = torch.zeros(bz, dtype=torch.bool, device=model.cfg.device) # if reach eot, exit

    output_tokens = []
    for index in range(max_new_tokens):
        if index == 0:
            logits = model(inputs,attention_mask = attention_mask,past_kv_cache=past_kv_cache)
        else:
            logits = model(inputs[:,-1:], past_kv_cache=past_kv_cache)
        
        sampled_tokens = logits[:,-1].argmax(-1)
        sampled_tokens[finished_sequences] = model.tokenizer.eos_token_id
        finished_sequences.logical_or_(
                        torch.isin(
                            sampled_tokens,
                            torch.tensor(model.tokenizer.eos_token_id).to(model.cfg.device),
                        )
                    )
        
        inputs = torch.cat([inputs, sampled_tokens.unsqueeze(-1)], dim=-1)
        output_tokens.append(sampled_tokens)
        if finished_sequences.all():
            break
    
    # filter out eot tokens
    output_tokens = torch.stack(output_tokens,dim=1).detach().cpu()
    del past_kv_cache
    return output_tokens

def tl_generate(model,prompts,max_new_tokens=512,do_sample=False,use_tqdm=False):
    if model.cfg.use_attn_result:
        model.cfg.use_attn_result = False # disable attn result, as it is not used in the generate function
        use_attn_result = True
    else:
        use_attn_result = False
    input_ids = encode_fn(model,prompts,True)
    # input_ids = model.to_tokens(prompts)
    with torch.no_grad():
        # out = model.generate(input_ids,max_new_tokens=max_new_tokens,do_sample=do_sample,verbose=use_tqdm)[:,input_ids.shape[1]:]
        out = custom_generate(model,input_ids.input_ids,input_ids.attention_mask,max_new_tokens = max_new_tokens) # use this as it is more consistent across batching vs single
    if use_attn_result:
        model.cfg.use_attn_result = True # restore attn result
    return model.tokenizer.batch_decode(out,skip_special_tokens=True)

def seed_all():
    torch.cuda.manual_seed(42)
    torch.manual_seed(42)
    np.random.seed(42)


def ablate_act(act,hook,vec,add=False,pos =None):
    if vec.ndim == 1:
        vec = vec.unsqueeze(0)
    for v in vec:
        v_norm = F.normalize(v,dim = -1)
        proj  = einsum(act,v_norm.unsqueeze(-1),'b c d, d s -> b c s')
        if pos:
            for i in range(len(act)):
                s,e = pos[i]
                if not add:
                    act[i,s:e] = act[i,s:e] - (proj[i,s:e] * v_norm)
                else:
                    act[i,s:e] = act[i,s:e] + (proj[i,s:e] * v_norm)
        else:
            if not add:
                act =  act -( proj * v_norm)
            else:
                act =  act +( proj * v_norm)
    return act

def add_act(act,vec,scale,tok_pos=None): # is a list of (start,end) for each sample
    if tok_pos is None:
        return act + scale*vec
    else:
        for i,a in enumerate(act):
            a[tok_pos[i][0]:tok_pos[i][1]] += scale*vec
        return act

def clamp_sae(act,hook,saes,circuit,pos = None,val = 0,multiply=True):
    """
    Hook to replace activation with sae activation while ablating certain features
    val is the val to either clamp/multiply
    pos is the position to clamp/multiply
    """
    layer = retrieve_layer_fn(hook.name)
    if saes.get(layer,None) is None: 
        return act
    f = saes[layer].encode(act.to(saes[layer].W_dec.device))
    x_hat = saes[layer].decode(f).to(act.device)
    res = act - x_hat
    feats = circuit.get(layer,[])
    if len(feats):
        if pos is None:
            token_pos = slice(None)
            if multiply:
                f[:,token_pos,feats] *= val
            else:
                f[:,token_pos,feats] = val
        else:
            if f.shape[1] > 1:
                token_pos = pos # list of positions for each batch
                for sample_pos in range(f.shape[0]):
                    s,e = token_pos[sample_pos]
                    if multiply:
                        f[sample_pos,s:e,feats] *= val
                    else:
                        f[sample_pos,s:e,feats] = val
    clamped_x_hat = saes[layer].decode(f).to(act.device)
    return clamped_x_hat + res

def clamp_sae_mask(act,hook,saes,circuit):
    layer = retrieve_layer_fn(hook.name)
    if saes.get(layer,None) is None: 
        return act
    f = saes[layer].encode(act.to(saes[layer].W_dec.device))
    x_hat = saes[layer].decode(f).to(act.device)
    res = act - x_hat
    layer_circuit = circuit.get(layer,[])
    f = f* layer_circuit.to(f.device)
    clamped_x_hat = saes[layer].decode(f).to(act.device)
    return clamped_x_hat + res

def nnsight_generate(model,prompts,gen_kwargs,vec=None,intervention = None,intervene_layers=None,scale = -1.0,return_inputs = False):
    if intervention is not None:
        assert intervene_layers is not None, "If intervention is provided, intervene_layers must also be specified."
        assert vec is not None, "If intervention is provided, vector must also be specified."
    
    if intervention == 'steer':
        if not isinstance(intervene_layers,list):
            intervene_layers = [intervene_layers]

        assert len(intervene_layers) == 1, "Steering should only be applied to the refusal vec layer." # multiple layers will get nonsensical gens
    else:
        intervene_layers = range(len(model.model.layers)) # ablate all layers by default

    inp_len = len(model.tokenizer(prompts,padding='longest',return_tensors='pt').input_ids[0])

    with torch.no_grad(), model.generate(prompts,**gen_kwargs) as gen:
        if intervention:
            for ablate_l in intervene_layers:
                with model.model.layers.all(): # apply to all tokens
                    l_act = model.model.layers[ablate_l].output[0][:]
                    if intervention == 'steer':
                        model.model.layers[ablate_l].output[0][:] = add_act(l_act,vec,scale=scale) 
                    else:
                        model.model.layers[ablate_l].output[0][:] = ablate_act(l_act,vec,add = scale > 0)
        output = model.generator.output.save()

    ablated_gen = model.tokenizer.batch_decode(output[:,inp_len:],skip_special_tokens=True)
    if not return_inputs:
        return ablated_gen
    else:
        return prompts, ablated_gen

def tl_batch_generate(model,prompts,bz=-1,saes=None,vec=None,steer_fn = None,use_tqdm=False,gen_kwargs = {},steer_args = {}): # include steer fn
    if bz == -1:
        bz = len(prompts)
        use_tqdm = False
    to_iter = tqdm(range(0,len(prompts),bz),total = len(prompts)//bz) if use_tqdm else range(0,len(prompts),bz)
    out = []
    for i in to_iter:
        batch = prompts[i:i+bz]
        model.reset_hooks()
        if steer_fn == 'ablate':
            model.add_hook(resid_name_filter,partial(ablate_act,vec = vec))
        elif steer_fn == 'addact':
            model.add_hook(lambda x : resid_name_filter(x) and f"blocks.{steer_args['layer']}" in x,partial(add_act,vec = vec,scale=steer_args['scale']))
        elif steer_fn == 'sae':
            model.add_hook(resid_name_filter,partial(clamp_sae,saes=saes,**steer_args)) # must have circuit, and optionally val and multiply
        resp = tl_generate(model,batch,**gen_kwargs)
        out.extend(resp)
    model.reset_hooks()
    return out

def find_token_span(token_ids, start_tokens, end_tokens):
    """
    Finds the start and end indices of the span in token_ids where
    the sequence from end of start_tokens to start of end_tokens occurs (inclusive).

    Returns (start_idx, end_idx), or (None, None) if not found.
    """
    # Find start
    n = len(token_ids)
    s = len(start_tokens)
    e = len(end_tokens)

    if isinstance(token_ids,torch.Tensor):
        token_ids = token_ids.tolist()

    start_idx,end_idx = None,None
    # Find start_tokens position
    for i in range(n - s + 1):
        if token_ids[i:i+s] == start_tokens:
            start_idx = i + s
            break


    # Find end_tokens position, searching after the start
    for j in range(start_idx + s, n - e + 1):
        if token_ids[j:j+e] == end_tokens:
            end_idx = j
            break

    return start_idx, end_idx

def find_substring_span(tokenizer, token_ids, tokens):
    full_text = tokenizer.decode(token_ids)
    sub_text = tokenizer.decode(tokens)
    idx = full_text.find(sub_text)
    if idx == -1:
        return None, None

    # Map char offsets to token indices
    # This is approximate and may fail with some tokenizers (esp. with BPEs)
    offsets = []
    current = 0
    for i, tid in enumerate(token_ids):
        piece = tokenizer.decode([tid])
        offsets.append((current, current+len(piece), i))
        current += len(piece)

    # Find token start and end that cover sub_text at idx
    start_token = end_token = None
    for (s, e, i) in offsets:
        if s <= idx < e:
            start_token = i
        if s < idx + len(sub_text) <= e:
            end_token = i+1
            break

    return start_token, end_token

def topk2d(tensor,topk):
    topk_feat_ind = torch.topk(tensor.flatten(),topk).indices
    topk_layers = topk_feat_ind // tensor.shape[1]
    topk_feats = topk_feat_ind % tensor.shape[1]
    return topk_layers,topk_feats

def topk_md(tensor,topk):
    values,flat_idx = torch.topk(tensor.view(-1),topk)
    coords = torch.stack(torch.unravel_index(flat_idx, tensor.shape), dim=1)
    return values, coords # coords is list of tensors with dim = tensor

def plot_pca(corrupt_acts,clean_acts):
    pca = PCA(n_components=2)

    pca_scores = []
    for l in range(len(corrupt_acts)):
        harmful_states = corrupt_acts[l].cpu().numpy()
        harmless_states = clean_acts[l].cpu().numpy()
        all_states = np.concatenate([harmful_states, harmless_states], axis=0)
        pca.fit(all_states)
        harmful_pca = pca.transform(harmful_states)
        harmless_pca = pca.transform(harmless_states)
        pca_scores.append((harmful_pca, harmless_pca))
    
    fig, axes = plt.subplots(nrows=len(pca_scores)//4, ncols=4, figsize=(10, 16), sharex=True, sharey=True)
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        ax = axes[i]
        harmful_pca, harmless_pca = pca_scores[i]
        # Plot PC1 vs PC2 for each
        sns.scatterplot(x=harmful_pca[:, 0], y=harmful_pca[:, 1], ax=ax, s=10, label='Harmful', color='r', alpha=0.5)
        sns.scatterplot(x=harmless_pca[:, 0], y=harmless_pca[:, 1], ax=ax, s=10, label='Harmless', color='b', alpha=0.5)
        ax.set_title(f'Layer {i}', fontsize=8)
    for ax in axes:
        ax.get_legend().remove()

    # Make a single legend for the whole figure
    # Use proxy artists for a clean shared legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Harmful', markerfacecolor='r', markersize=8, alpha=0.5),
        Line2D([0], [0], marker='o', color='w', label='Harmless', markerfacecolor='b', markersize=8, alpha=0.5)
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=2, fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.98])  # leave space for legend at the top
    plt.show()


class Probe(torch.nn.Module):
    """
    2-layer bottleneck MLP of the form
        u  ->  ReLU(W1ᵀ u)  ->  W2ᵀ ->  scalar logit
    """
    def __init__(self, in_dim, dtype: torch.dtype = torch.float32,out_dim = 2):
        super().__init__()
        self.out_dim = out_dim
        self.net =  torch.nn.Linear(in_dim, out_dim, bias=False, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) if self.out_dim > 1 else self.net(x).squeeze(-1)

def no_rotation(prompts,format_fn,tokenizer=None,attack_only=False):
    input_dict = format_fn(prompts)
    if attack_only: ## only convert the segment ids for the attack, dont touch the data, this is to isolate the "instruction-following" effect on the attack only.
        attacks = [d['attack'] for d in prompts]
        tokenized_attacks = [tokenizer.encode(attack, add_special_tokens=False) for attack in attacks]
        for i,tokenized_attack in enumerate(tokenized_attacks):
            s,e = find_substring_span(tokenizer,input_dict['input_ids'][i].tolist(),tokenized_attack)
            input_dict['segment_ids'][i][s:e] = 0 # set the segment ids to 0 for the attack tokens
    else:
        input_dict['segment_ids'] = torch.zeros_like(input_dict['input_ids']) 
    return input_dict