from copy import deepcopy
from trl import DPOTrainer
import torch
import torch.nn.functional as F
from torch import nn
from typing import Union


def apply_dpo_chat_template(example,tokenizer):
    prompt = tokenizer.apply_chat_template(example["prompt"], tools=example.get('tools',None), tokenize=False, add_generation_prompt=True,enable_thinking=False)
    chosen = tokenizer.apply_chat_template(example["prompt"] + example['chosen'],tools=example.get('tools',None), tokenize=False, add_generation_prompt=False,enable_thinking=False)
    chosen = chosen[len(prompt):]
    rejected = tokenizer.apply_chat_template(example["prompt"] + example['rejected'],tools=example.get('tools',None), tokenize=False, add_generation_prompt=False,enable_thinking=False)
    rejected = rejected[len(prompt):]
    return {"prompt": prompt, "chosen": chosen, "rejected": rejected}

    
def segment_start(input_ids,tokenizer,start_token): # given a start token, mark everything from that onwards as 1.
    device = input_ids.device
    start_ids = torch.tensor(tokenizer.encode(start_token,add_special_tokens=False)).to(device)
    mask = torch.zeros_like(input_ids)
    for jj, input_ids in enumerate(input_ids): # assume only one span
        i = 0
        while i <= len(input_ids) - len(start_ids):
            if torch.equal(input_ids[i:i+len(start_ids)], start_ids):
                mask[jj,i:] = 1
                break
            else:
                i += 1
    return mask

def multiturn_aside_encode(encoded,tokenizer,start_tokens,end_tokens,until_last_token=None,return_spans=False):
    """
    we have a list of start and end_tokens, all of these should be rotated. This is to account for the uncontrollable turn order in agentic tool use, such as tool -> assistant or tool -> user
    For agentic tool use or samples with data field, we ONLY mark the assistant span if it is after the tool span, the reason as to why assistant span is marked is due to the way ASIDE is trained, where the model is trained to generate the assistant response given the rotated tool span; w/o rotating assistant span, the model degenerates.
    """
    starting_tokens = deepcopy(start_tokens)
    ending_tokens = deepcopy(ending_tokens)
    if len(starting_tokens) > 1:
        assert any([tool_t in starting_tokens[-1] for tool_t in ['tool_response','reference_data']]),f'Last start and end token should point to the input/tool.' # ensure last is tool or input
        tool_token = starting_tokens[-1]
        starting_tokens = starting_tokens[::-1] # reverse the order, so that we first look for the last start token, which should be the tool/input
        ending_tokens = ending_tokens[::-1]
    else: # only assistant
        tool_token = None
        starting_tokens = starting_tokens
        ending_tokens = ending_tokens
    device = encoded['input_ids'].device
    start_ids = [torch.tensor(tokenizer.encode(start_token,add_special_tokens=False)).to(device) for start_token in starting_tokens]
    end_ids = [torch.tensor(tokenizer.encode(end_token,add_special_tokens=False)).to(device) for end_token in ending_tokens]
    until_last_ids = torch.tensor(tokenizer.encode(until_last_token,add_special_tokens=False)).to(device) if until_last_token is not None else None
    
    assert len(start_tokens) == len(end_tokens) != 0, "start_tokens and end_tokens must be provided and of same length"
    # assert until_last_token is not None, "until_last_token must be provided, this is the start token that is allowed to be unclosed - should be the assistant token."

    mask = torch.zeros_like(encoded['input_ids'])
    spans = [] # also return spans
    for jj, input_ids in enumerate(encoded['input_ids']):
        curr_span = []
        curr_input_str = tokenizer.decode(input_ids)
        if tool_token and tool_token not in curr_input_str: # if we are rotating tool and not the current sample dont have, theres nothing to rotate
            continue # no tool/input, skip the rotation
        tool_ids = -1 # note down where is the tool/input start
        for curr_start_ids,curr_end_ids in zip(start_ids,end_ids):
            i = 0
            while i <= len(input_ids) - len(curr_start_ids):
                if torch.equal(input_ids[i:i+len(curr_start_ids)], curr_start_ids): # find the start of the current start token
                    j = i + len(curr_start_ids)
                    found = False
                    while j <= len(input_ids) - len(curr_end_ids): # look for the end token
                        if torch.equal(input_ids[j:j+len(curr_end_ids)], curr_end_ids):
                            if torch.equal(curr_start_ids, start_ids[0]): # mark when a complete tool span is found or if there is only one start token (eg assistant only)
                                if tool_ids == -1: # might have multiple tools, only mark the first one found
                                    tool_ids = deepcopy(i+len(curr_start_ids))
                                mask[jj,i:j+len(curr_end_ids)] = 1
                                curr_span.append((i,j+len(curr_end_ids)))
                            elif len(starting_tokens) > 1 and torch.equal(curr_start_ids, start_ids[-1]) and i > tool_ids: # if includes both assistant and tool, only mark the assistant if it is after the tool
                                mask[jj,i:j+len(curr_end_ids)] = 1
                                curr_span.append((i,j+len(curr_end_ids)))
                            i = j + len(curr_end_ids)
                            found = True
                            break
                        j += 1
                    if not found:
                        if until_last_ids is not None and torch.equal(input_ids[i:i+len(until_last_ids)], until_last_ids):
                            # allow to be unclosed, mark until end of input
                            mask[jj,i:len(input_ids)] = 1
                            curr_span.append((i,len(input_ids)))
                            break
                        else:
                            i += len(curr_start_ids)
                else:
                    i += 1
        spans.append(curr_span)
    if return_spans:
        return mask,spans
    return mask

class DPOTrainerWithSegments(DPOTrainer):
    def __init__(self, *args, start_tokens,end_tokens,start_from = None,**kwargs):
        super().__init__(*args, **kwargs)
        self.start,self.end = start_tokens,end_tokens
        self.start_tokens = start_tokens
        self.end_tokens = end_tokens
        self.printed = False
        self.start_from = start_from # once detected a start, mark everything after as 1. is a string marker

    def _set_signature_columns_if_needed(self):
        super()._set_signature_columns_if_needed()
        # keep your extra dataset columns so Trainer doesn't drop them
        self._signature_columns += [
            "segment_ids",
        ]

    def concatenated_inputs(self,batch, padding_value: int):
        out = DPOTrainer.concatenated_inputs(batch, padding_value)
        input_ids = out["prompt_input_ids"]
        
        if self.start_from is not None:
            segment_ids = segment_start(input_ids, self.processing_class, self.start_from)
        else: # Take spans
            segment_ids = multiturn_aside_encode(input_ids, self.processing_class, self.start_tokens, self.end_tokens, until_last_token=self.start_tokens[0])

        completion_input_ids = out["completion_input_ids"]
        completion_len = completion_input_ids.size(1)
        segment_ids = torch.cat([segment_ids, torch.ones((segment_ids.size(0), completion_len), dtype=torch.long, device=segment_ids.device)], dim=1) # completion is rotated.
        out['segment_ids'] = segment_ids
        return out
        

    def concatenated_forward(self, model: nn.Module, batch: dict[str, Union[list, torch.LongTensor]]):
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        num_examples = batch["prompt_input_ids"].shape[0]

        concatenated_batch = self.concatenated_inputs(batch, padding_value=self.padding_value)

        model_kwargs = {}
        if self.aux_loss_enabled:
            model_kwargs["output_router_logits"] = True

        # Add the pixel values and attention masks for vision models
        if "pixel_values" in concatenated_batch:
            model_kwargs["pixel_values"] = concatenated_batch["pixel_values"]
        if "pixel_attention_mask" in concatenated_batch:
            model_kwargs["pixel_attention_mask"] = concatenated_batch["pixel_attention_mask"]
        if "image_sizes" in concatenated_batch:
            model_kwargs["image_sizes"] = concatenated_batch["image_sizes"]

        prompt_input_ids = concatenated_batch["prompt_input_ids"]
        prompt_attention_mask = concatenated_batch["prompt_attention_mask"]
        completion_input_ids = concatenated_batch["completion_input_ids"]
        completion_attention_mask = concatenated_batch["completion_attention_mask"]
        if self.is_encoder_decoder:
            labels = completion_input_ids
            labels[completion_attention_mask == 0] = self.label_pad_token_id
            outputs = model(
                input_ids=prompt_input_ids,
                attention_mask=prompt_attention_mask,
                labels=labels,  # we need the labels for the logits to be returned
                **model_kwargs,
            )
            logits = outputs.logits
            loss_mask = completion_attention_mask.bool()
        else:
            # Concatenate the prompt and completion inputs
            input_ids = torch.cat((prompt_input_ids, completion_input_ids), dim=1)
            attention_mask = torch.cat((prompt_attention_mask, completion_attention_mask), dim=1)
            # Mask the prompt but not the completion for the loss
            loss_mask = torch.cat(
                (torch.zeros_like(prompt_attention_mask), completion_attention_mask),
                dim=1,
            )
            segment_ids = concatenated_batch['segment_ids']

            # Flush left to reduce the memory usage
            # [[0, 0, x, x, x, x],  ->  [[x, x, x, x],
            #  [0, x, x, x, 0, 0]]       [x, x, x, 0]]
            for i in range(attention_mask.size(0)):
                first_one_idx = torch.nonzero(attention_mask[i])[0].item()
                input_ids[i] = torch.roll(input_ids[i], shifts=-first_one_idx)
                attention_mask[i] = torch.roll(attention_mask[i], shifts=-first_one_idx)
                loss_mask[i] = torch.roll(loss_mask[i], shifts=-first_one_idx)
                segment_ids[i] = torch.roll(segment_ids[i], shifts=-first_one_idx)

            # Get the first column idx that is all zeros and remove every column after that
            empty_cols = torch.sum(attention_mask, dim=0) == 0
            first_empty_col = torch.nonzero(empty_cols)[0].item() if empty_cols.any() else attention_mask.size(1)
            input_ids = input_ids[:, :first_empty_col]
            attention_mask = attention_mask[:, :first_empty_col]
            loss_mask = loss_mask[:, :first_empty_col]
            segment_ids = segment_ids[:, :first_empty_col]

            # Truncate right
            if self.args.max_length is not None:
                input_ids = input_ids[:, : self.args.max_length]
                attention_mask = attention_mask[:, : self.args.max_length]
                loss_mask = loss_mask[:, : self.args.max_length]
                segment_ids = segment_ids[:, : self.args.max_length]

            if self.use_num_logits_to_keep:
                # Compute num_logits_to_keep based on loss_mask pattern:
                # [[0, 0, 0, x, x, x, x],
                #  [0, 0, 0, x, x, x, 0]]
                #         ^ start computing logits from here ([:, -(7-3+1):])
                first_compute_index = loss_mask.nonzero(as_tuple=True)[1].min()
                num_logits_to_keep = loss_mask.shape[1] - first_compute_index
                model_kwargs["num_logits_to_keep"] = num_logits_to_keep.item() + 1  # +1 for the first label
            
            if not self.printed:
                to_print = []
                for kk,seg in enumerate(segment_ids[0].tolist()):
                    if seg == 1:
                        to_print.append(input_ids[0,kk].item())
                print (f'FULL PROMPT: {self.processing_class.decode(input_ids[0].tolist())}')
                print (f'SEGMENTED: {self.processing_class.decode(to_print)}')
                self.printed = True
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask,segment_ids = segment_ids, **model_kwargs)

            # Offset the logits by one to align with the labels
            logits = outputs.logits[:, :-1, :]
            labels = input_ids[:, 1:].clone()
            loss_mask = loss_mask[:, 1:].bool()

            if self.use_num_logits_to_keep:
                # Align labels with logits
                # logits:    -,  -, [x2, x3, x4, x5, x6]
                #                     ^ --------- ^       after logits[:, :-1, :]
                # labels:   [y0, y1, y2, y3, y4, y5, y6]
                #                         ^ --------- ^   with num_logits_to_keep=4, [:, -4:]
                # loss_mask: [0,  0,  0,  1,  1,  1,  1]
                labels = labels[:, -num_logits_to_keep:]
                loss_mask = loss_mask[:, -num_logits_to_keep:]

        if logits.shape[:2] != labels.shape[:2]:
            # for llava, the returned logits include the image tokens (placed before the text tokens)
            seq_len = labels.shape[1]
            logits = logits[:, -seq_len:]

        # Compute the log probabilities of the labels
        labels[~loss_mask] = 0  # dummy token; we'll ignore the losses on these tokens later
        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
        per_token_logps[~loss_mask] = 0
        all_logps = per_token_logps.sum(-1)

        output = {}

        if self.use_weighting:
            with torch.no_grad():
                # Eq (2) of the WPO paper: https://huggingface.co/papers/2406.11827
                logprobs = F.log_softmax(logits, dim=-1)
                weights_adjustment_factor = torch.logsumexp(2 * logprobs, dim=-1)  # same as sum(probs**2) in log space
                per_token_logps_adjusted = per_token_logps - weights_adjustment_factor
                all_weights = (per_token_logps_adjusted * loss_mask).sum(-1) / loss_mask.sum(-1)
                chosen_weights = all_weights[:num_examples]
                rejected_weights = all_weights[num_examples:]
                output["policy_weights"] = torch.clamp(torch.exp(chosen_weights + rejected_weights), max=1)

        if self.args.rpo_alpha is not None:
            # Only use the chosen logits for the RPO loss
            chosen_logits = logits[:num_examples]
            chosen_labels = labels[:num_examples]

            # Compute the log probabilities of the labels
            output["nll_loss"] = F.cross_entropy(
                torch.flatten(chosen_logits, end_dim=1), torch.flatten(chosen_labels, end_dim=1), ignore_index=0
            )

        if self.loss_type == "ipo":
            all_logps = all_logps / loss_mask.sum(-1)

        output["chosen_logps"] = all_logps[:num_examples]
        output["rejected_logps"] = all_logps[num_examples:]
        output["mean_chosen_logits"] = logits[:num_examples][loss_mask[:num_examples]].mean()
        output["mean_rejected_logits"] = logits[num_examples:][loss_mask[num_examples:]].mean()

        if self.aux_loss_enabled:
            output["aux_loss"] = outputs.aux_loss

        return output