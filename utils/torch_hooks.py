import torch
from contextlib import contextmanager
from typing import Callable, Dict, Optional, Union, List, Tuple
from collections import defaultdict
from tqdm import tqdm

TensorOrTuple = Union[torch.Tensor, Tuple, Dict, List]

def _get_decoder_layers(model) -> List[torch.nn.Module]:
    """
    Returns a list of decoder blocks irrespective of model family.
    - LLaMA/Mistral/Qwen: model.model.layers
    - GPT-2 style: model.transformer.h
    """
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return list(model.transformer.h)
    raise AttributeError("Could not locate decoder layers on this model.")

def _replace_first_in_output(output: TensorOrTuple, new_hidden: torch.Tensor) -> TensorOrTuple:
    """
    Replace the first 'hidden_states' element in the layer's output with new_hidden.
    Handles Tensor, tuple, list, dict outputs from HF layers.
    """
    if isinstance(output, torch.Tensor):
        return new_hidden
    elif isinstance(output, (tuple, list)):
        # replace element 0
        return type(output)([new_hidden] + list(output)[1:])
    elif isinstance(output, dict):
        # common keys: 'hidden_states' or 'last_hidden_state'
        key = "hidden_states" if "hidden_states" in output else (
              "last_hidden_state" if "last_hidden_state" in output else None)
        if key is None:
            raise ValueError("Dict output without a 'hidden_states'/'last_hidden_state' key.")
        new_out = dict(output)
        new_out[key] = new_hidden
        return new_out
    else:
        raise TypeError(f"Unsupported layer output type: {type(output)}")


class LayerIO:
    """
    Stores per-layer inputs/outputs. You can read these after a forward pass.
    """
    def __init__(self):
        self.pre = dict()
        self.post= dict()

    def clear(self):
        self.pre.clear()
        self.post.clear()

class HookedIntervention:
    """
    Manage forward hooks to:
      - capture residual stream at every decoder layer (pre & post)
      - apply a user-defined intervention on the residual stream of chosen layers
    """
    def __init__(
        self,
        model: torch.nn.Module,
        layers= -1,
        intervention_fn: Optional[Callable[[int, torch.Tensor], torch.Tensor]] = {},
        pre_intervention_fn: Optional[Callable[[int, torch.Tensor], torch.Tensor]] = None, # for inputs and not outputs
        layers_to_edit: Optional[List[int]] = None,
        capture_post: bool = True,
        capture_pre: bool = False,
        target = ['residual'], # standard residual
        attribution: bool = False
    ):
        """
        Args:
          model: a HF causal LM (AutoModelForCausalLM or similar)
          intervention_fn: function(layer_idx, hidden_states)->modified_hidden_states
                           If None, no edits are applied.
          layers_to_edit: list of layer indices to edit. If None, edit all layers.
          capture_io: if True, stores layer input/output activations in self.io
        """
        self.model = model
        self.layers = _get_decoder_layers(model) if layers == -1 else layers
        self.intervention_fn = intervention_fn
        if self.intervention_fn is None:
            self.intervention_fn = {}
        self.pre_intervention_fn = pre_intervention_fn
        self.layers_to_edit = set(range(len(self.layers))) if layers_to_edit is None else set(layers_to_edit)
        self.capture_post = capture_post
        self.capture_pre = capture_pre
        self.io = LayerIO()
        self._hooks = []
        self._pre_hooks = []
        self.targets = [t.lower() for t in target]
        self.attr = defaultdict(dict) # module: layer (for linear attribution)
        self.num_heads = getattr(self.model.config, 'num_attention_heads', None)
        if self.num_heads is None:
            raise ValueError("Model config does not have 'num_attention_heads'. Ensure this is a causal LM with attention.")
        self.hidden_size = self.model.config.hidden_size
        self.head_dim = self.model.config.head_dim
        self.attribution = attribution
        
    def _resolve_target_module(self, target,layer): # get the layer we want to hook
        if target == "residual": return layer
        if 'attn' in target or 'mlp' in target:
            if 'attn' in target:
                target_module = layer.self_attn
            else:
                target_module = layer.mlp
            if '.' in target:
                qkv_target = target.split('.')[1].strip().lower()
                target_module = getattr(target_module, qkv_target, None)
        if target_module is None:
            raise AttributeError("Target submodule not found")
        return target_module

    def _pre_hook(self, idx,io):
        def hook(module, inputs):
            if len(inputs) == 0:
                return
            x = inputs[0]
            if self.capture_pre:
                # store a detached (no-grad) copy to save memory if you don't need backprop
                self.io.pre[idx] = x.detach().cpu()
            if self.pre_intervention_fn is not None and idx in self.layers_to_edit:
                # apply intervention on input
                x_new = self.pre_intervention_fn(idx, x)
                if x_new is not None:
                    # replace the first input tensor with the modified one
                    return _replace_first_in_output(inputs, x_new)
        return hook

    def _fwd_hook(self, idx,intervention):
        def hook(module, inputs, output):
            # Extract the hidden tensor from output
            if isinstance(output, torch.Tensor):
                h = output
            elif isinstance(output, (tuple, list)):
                h = output[0]
            elif isinstance(output, dict):
                h = output.get("hidden_states", output.get("last_hidden_state", None))
                if h is None:
                    raise ValueError("Cannot find hidden_states in dict output.")
            else:
                raise TypeError(f"Unsupported output type at layer {idx}: {type(output)}")

            # Save post if requested
            if self.capture_post:
                self.io.post[idx] = h.detach().cpu()

            # Apply edit if configured
            if intervention is not None and idx in self.layers_to_edit:
                h_new = intervention(idx, h)
                if h_new is not None and (h_new is not h):
                    return _replace_first_in_output(output, h_new)

            # no change
            return None
        return hook

    @contextmanager
    def activate(self):
        """
        Context manager that installs hooks and ensures removal afterwards.
        """
        try:
            # install hooks
            for target in self.targets:
                for i, layer in enumerate(self.layers):
                    module = self._resolve_target_module(target,layer)
                    if not self.attribution:
                        if self.capture_pre or self.pre_intervention_fn is not None:
                            self._pre_hooks.append(module.register_forward_pre_hook(self._pre_hook(i), with_kwargs=False))
                                
                        if self.capture_post or self.intervention_fn is not None:
                            if 'resid' in target:
                                self._hooks.append(module.register_forward_hook(self._fwd_hook(i,self.intervention_fn.get(target,None)), with_kwargs=False))
                            elif 'attn_score' in target:
                                self._hooks.append(module.register_forward_hook(self.capture_attention_scores_hook(i), with_kwargs=False))
                            elif 'attn' in target: # no backprop
                                self._hooks.append(module.register_forward_hook(self.attention_head_attribution_hook(i,backprop=False,intervention = self.intervention_fn.get(target,None)), with_kwargs=False))
                            elif 'mlp' in target: # no backprop
                                self._hooks.append(module.register_forward_hook(self.mlp_attribution_hook(i,backprop=False,intervention = self.intervention_fn.get(target,None)), with_kwargs=False))
                    else:
                        if 'attn' in target:
                            self._hooks.append(module.register_forward_hook(self.attention_head_attribution_hook(i,backprop=True,intervention = self.intervention_fn.get(target,None)), with_kwargs=False))
                        elif 'mlp' in target:
                            self._hooks.append(module.register_forward_hook(self.mlp_attribution_hook(i,backprop=True,intervention = self.intervention_fn.get(target,None)),with_kwargs=False)) # TODO havent do one for residual
            yield self
        finally:
            # remove hooks
            for h in self._hooks:
                h.remove()
            for h in self._pre_hooks:
                h.remove()
            self._hooks.clear()
            self._pre_hooks.clear()
            
    # Do attribution on the attention heads and MLPs
    def attention_head_attribution_hook(self, idx,backprop=False,intervention=None):
        def attn_hook(module,inputs,outputs):
            if self.capture_post or self.attribution or intervention:
                x = inputs[0]
                x_heads = x.view(x.size(0), x.size(1), self.num_heads, self.head_dim)  # [B,S,nh,dh]
                W = module.weight  # [H, H] == [nh*dh, H]
                W_dhc = W.view(self.hidden_size, self.num_heads, self.head_dim)
                o_per_head = torch.einsum('bshc,dhc->bshd', x_heads, W_dhc)           # [B,S,nh,Hout]
                if backprop:
                    o_per_head.retain_grad()  # keep grad after backward
                    self.attr['attn'][idx] = o_per_head
                else:
                    self.attr['attn'][idx] = o_per_head.detach().cpu()
                out = o_per_head.sum(dim=2)                                           # [B,S,Hout]
                if module.bias is not None:
                    out = out + module.bias
                
            if intervention is not None and idx in self.layers_to_edit:
                # h = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
                h_new = intervention(idx, o_per_head) # edit o_per_head
                h_new = h_new.sum(dim=2)
                if module.bias is not None:
                    h_new = h_new + module.bias
                if h_new is not None and (h_new is not out):
                    return _replace_first_in_output(outputs, h_new)
            return out
        return attn_hook
    
    def capture_attention_scores_hook(self, idx: int):
        """
        Stores attention probabilities (B, nH, T_q, T_k) under self.attr['attn_scores'][idx].
        If last_token_only=True, stores (B, nH, T_k).
        Requires that the attention module actually computes/returns weights, which
        generally happens when output_attentions=True flows into the attention block.
        """
        def _hook(module, inputs, outputs):
            attn = None
            # print (outputs[1].shape)
            # Common HF attention returns: (attn_output, attn_weights, past_key_value) when output_attentions=True
            if isinstance(outputs, tuple):
                # try the second item first (typical: attn_weights)
                if len(outputs) >= 2:
                    attn = outputs[1]
                # some impls might place it elsewhere; add extra guards if needed
            elif isinstance(outputs, dict):
                for key in ("attn_weights", "attentions", "attn_probs"):
                    if key in outputs and isinstance(outputs[key], torch.Tensor):
                        attn = outputs[key]
                        break

            if attn is not None:
                self.attr["attn_scores"][idx] = attn.detach().cpu()
            return None
        return _hook

    
    
    def mlp_attribution_hook(self, idx,backprop=False,intervention=None):
        def mlp_hook(module, inputs, outputs):
            h = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
            if self.capture_post or self.attribution:
                if backprop:
                    h.retain_grad()
                    self.attr['mlp'][idx] = h
                else:
                    self.attr['mlp'][idx] = h.detach().cpu()
            if intervention is not None and idx in self.layers_to_edit:
                h_new = intervention(idx, h)
                if h_new is not None and (h_new is not h):
                    return _replace_first_in_output(outputs, h_new)
            return None
        return mlp_hook
    
    def clear(self):
        self.io.clear()
        self.attr.clear()
        
            
# 3) Define a steering vector and an intervention function
def make_add_vector_intervention(v: torch.Tensor, alpha=-1, t_idx=None):
    """
    Returns an intervention_fn that adds alpha*v to hidden_states[:, t_idx, :].
    v must be on the same device and dtype as the hidden states when applied.
    """
    def intervention(layer_idx: int, hidden: torch.Tensor) -> torch.Tensor:
        # hidden: [B, T, D]
        # Broadcast-safe addition
        hidden = hidden.clone()  # avoid in-place on shared tensors
        if t_idx is None:
            hidden = hidden + alpha * v.to(hidden.dtype).to(hidden.device)
        else:
            if t_idx < 0 or t_idx >= hidden.size(1):
                return hidden
            hidden[:, t_idx, :] = hidden[:, t_idx, :] + alpha * v.to(hidden.dtype).to(hidden.device)
        return hidden
    return intervention
        



def contrast_act_native(model,clean,corrupt,bz =-1 ,avg_before_contrast=True):
    clean_acts = defaultdict(list)
    corrupt_acts = defaultdict(list)
    n = len(clean['input_ids'])
    if bz == -1:
        bz = n
    num_layers = len(model.model.layers)
    edited_layers = list(range(num_layers)) # edit all layers

    clean_hi = HookedIntervention(model, intervention_fn=None, layers_to_edit=edited_layers, capture_post=True)
    corrupt_hi = HookedIntervention(model, intervention_fn=None, layers_to_edit=edited_layers, capture_post=True)
    for i in tqdm(range(0,n,bz),total = n//bz):
        batch_clean = {k: v[i:i+bz] for k,v in clean.items()}
        batch_corrupt = {k: v[i:i+bz] for k,v in corrupt.items()}
        with torch.no_grad():
            with clean_hi.activate():
                _ = model(**batch_clean)
            with corrupt_hi.activate():
                _ = model(**batch_corrupt)
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
    for l in range(num_layers):
        if avg_before_contrast:
            directions[l] = corrupt_acts[l].mean(0) - clean_acts[l].mean(0)
        else:
            directions[l] = (corrupt_acts[l]- clean_acts[l]).mean(0)
    return directions


def generate_func(model,prompts,format_fn,gen_kwargs,steer_vec=None,alpha=-1,t_idx=None,layers=[]):
    if not isinstance(prompts,dict): # if is list of strings then format it.
        formatted_prompts = format_fn(prompts)
    else:
        formatted_prompts = prompts
    if isinstance(layers, int):
        layers = [layers]
    if steer_vec is not None and len(layers):
        # Apply steering vector intervention
        intervention = make_add_vector_intervention(steer_vec, alpha=alpha, t_idx=t_idx)
        hi = HookedIntervention(model, intervention_fn=intervention, layers_to_edit=layers, capture_post=False,capture_pre=False)
        with torch.no_grad():
            with hi.activate():
                out = model.generate(**formatted_prompts, use_cache=True, **gen_kwargs)
        del hi
    else:
        out = model.generate(**formatted_prompts, use_cache=True, **gen_kwargs)
    decoded_tokens = model.tokenizer.batch_decode(out[:,formatted_prompts['input_ids'].shape[1]:], skip_special_tokens=True)
    return decoded_tokens
        