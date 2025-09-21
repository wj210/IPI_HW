import torch
from collections import defaultdict
import numpy as np
from copy import deepcopy
from utils.torch_hooks import HookedIntervention
from utils.utils import *



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

def train_probe(
    acts,
    labels,
    lr=1e-2,
    epochs=1,
    seed=42,
    val_acts=None,
    val_labels = None,
    bz = 64,
    weight_decay=1e-4,
    out_dim = 2,

):
    torch.manual_seed(seed)
    torch.set_grad_enabled(True)
    d_probe = acts.shape[-1]
    probe = Probe(d_probe,out_dim=out_dim).to(acts.device)

    decay, no_decay = [], []
    for name, p in probe.named_parameters():
        if not p.requires_grad:
            continue
        (decay if p.ndim > 1 else no_decay).append(p)
    param_groups = [
        {"params": decay,     "weight_decay": weight_decay},
        {"params": no_decay,  "weight_decay": 0.0},
    ]

    optimizer = torch.optim.AdamW(param_groups, lr=lr)
    criterion = torch.nn.CrossEntropyLoss() if out_dim > 1 else torch.nn.BCEWithLogitsLoss()
    best_probe = deepcopy(probe)
    best_val_acc = 0
    

    for epoch in range(epochs):
        ## Training
        for batch_id in range(0, len(acts), bz):
            batch_acts = acts[batch_id:batch_id + bz]
            batch_labels = labels[batch_id:batch_id + bz]
            
            optimizer.zero_grad()
            logits = probe(batch_acts)

            batch_labels = batch_labels.long() if out_dim > 1 else batch_labels.float()
            
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()

        epoch_val_acc = []
        for batch_id in range(0, len(val_acts), bz):
            val_batch_acts = val_acts[batch_id:batch_id + bz]
            val_batch_labels = val_labels[batch_id:batch_id + bz]
            with torch.no_grad():
                logits_val = probe(val_batch_acts)
                pred_val = logits_val.argmax(dim=-1) if out_dim > 1 else logits_val > 0.0
                val_acc = (pred_val == val_batch_labels.to(logits_val)).float().tolist()
                epoch_val_acc.extend(val_acc)
        epoch_val_acc = np.mean(epoch_val_acc)

        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            best_probe = deepcopy(probe)
            
    torch.set_grad_enabled(False)

    return best_probe, best_val_acc
    
    
@torch.no_grad()
def test_probe(
    probe,
    acts,
    labels,
    grp_labels = None,
    bz = 64,
    avg=True,
):
    mean_acc = []
    grp_acc = defaultdict(list)
    for batch_id in range(0, len(acts), bz):
        batch_acts = acts[batch_id:batch_id + bz]
        batch_labels = labels[batch_id:batch_id + bz]

        logits = probe(batch_acts)
        preds = logits.argmax(-1) if logits.ndim == 2 else (logits > 0.0)
        acc = (preds == batch_labels.to(preds)).float()
        mean_acc.append(acc.mean().item())
        if grp_labels is not None:
            batch_grp_labels = grp_labels[batch_id:batch_id + bz]
            for grp_label,a in zip(batch_grp_labels,acc):
                grp_acc[grp_label].append(a.item())

    if grp_labels is not None:
        if avg:
            return np.mean(mean_acc), {k: np.mean(v) for k, v in grp_acc.items()}
        else:
            return np.mean(mean_acc), {k: v for k, v in grp_acc.items()}
    else:
        if avg:
            return np.mean(mean_acc)
        else:
            return mean_acc
        
def get_activations(target_model,prompts,prompt_fn,instr_tok,inp_tok,resp_tok,is_adv=False,intervention=None):
    """
    is_adv looks into the data tokens and annotates the attack tokens, if False, ignore
    Both of these arguments are only used for test else it only returns activation and labels.
    intervention: directly pass the hookedintervention in.
    """
    activations = defaultdict(list)
    labels = []
    adv_samples_to_token_ids = [] # annotate which adv token_ids, correspond to which sample
    adv_grp_labels = [] # annotates normal data tokens from attack.
    hi = HookedIntervention(target_model, intervention_fn={}, capture_post=True) if intervention is None else intervention
    for i,prompt in enumerate(prompts):
        acts = {}
        inputs = prompt_fn([prompt])
        if isinstance(hi,list):
            curr_hi = hi[i]
        else:
            curr_hi = hi
        with torch.no_grad():
            with curr_hi.activate():
                _ = target_model(**inputs)
        for l in range(len(target_model.model.layers)): # store the act first then only take instr,data later
            acts[l] = curr_hi.io.post[l][0]
        curr_hi.clear()
        inp_ids = inputs['input_ids'][0].tolist()
        instr_end = find_substring_span(target_model.tokenizer,inp_ids,instr_tok)[1]
        inpt_start,inpt_end = find_substring_span(target_model.tokenizer,inp_ids,inp_tok)
        resp_start = find_substring_span(target_model.tokenizer,inp_ids,resp_tok)[0]
        
        ## annotate if the input is instruction or data
        instr_len = len(inp_ids[instr_end:inpt_start])
        inpt_len = len(inp_ids[inpt_end:resp_start])
        labels.extend([1] * instr_len) # 1 for instruction, 0 for input
        labels.extend([0] * inpt_len)
        
        if is_adv:
            is_att_ids = [0]*(instr_len + inpt_len)
            tokenized_attack = target_model.tokenizer.encode(prompt['attack'],add_special_tokens=False)
            att_s,att_e = find_substring_span(target_model.tokenizer,inp_ids[inpt_end:resp_start],tokenized_attack) # only find within the input span
            is_att_ids[instr_len+att_s:instr_len+att_e] = [1] * (att_e - att_s)
            ## annotate the sample to token ids
            adv_samples_to_token_ids.append((att_e-att_s))
            
            adv_grp_labels.extend(is_att_ids)
            
        
        for l in range(len(target_model.model.layers)):
            activations[l].append(acts[l][instr_end:instr_end+instr_len].detach().cpu())
            activations[l].append(acts[l][inpt_end:inpt_end+inpt_len].detach().cpu())
        del acts
    activations = {k:torch.concat(v,dim=0) for k,v in activations.items()}
    labels = torch.tensor(labels)
    
    return activations, labels, adv_grp_labels,adv_samples_to_token_ids

def get_and_train_probe(target_model,ds_dict,prompt_fn,instr_tok,inp_tok,resp_tok,epochs=1,lr=1e-2):

    train_act,train_label,_,_ = get_activations(
        target_model,ds_dict['train'],prompt_fn,instr_tok,inp_tok,resp_tok
    )
    val_act,val_label,_,_ = get_activations(
        target_model,ds_dict['val'],prompt_fn,instr_tok,inp_tok,resp_tok
    )
    print (f'Train/Val size: {len(train_act[0])}/{len(val_act[0])}, ')
    assert train_act[0].shape[0] == train_label.shape[0] and val_act[0].shape[0] == val_label.shape[0]

    ## Train probe
    probes = {}
    probes_acc = {}
    for l in train_act:
        layer_probe,layer_acc = train_probe(train_act[l].float().to(target_model.device),train_label.to((target_model.device)),lr,epochs,val_acts= val_act[l].float().to((target_model.device)),val_labels=val_label.to((target_model.device)),bz=64,out_dim=1) # single output
        probes[l] = layer_probe
        probes_acc[l] = layer_acc

    best_layer = max(probes_acc,key = probes_acc.get)
    print (f'Best layer: {best_layer}, val acc: {probes_acc[best_layer]:.4f}')
    return probes,best_layer
