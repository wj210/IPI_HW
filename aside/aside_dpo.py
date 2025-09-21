from trl import DPOTrainer
import torch
import torch.nn.functional as F
from torch import nn
from typing import Union


def apply_chat_tool_template(example,tokenizer):
    prompt = tokenizer.apply_chat_template(example["prompt"], tools=example['tools'], tokenize=False, add_generation_prompt=True)
    chosen = tokenizer.apply_chat_template(example["prompt"] + example['chosen'],tools=example['tools'], tokenize=False, add_generation_prompt=False)
    chosen = chosen[len(prompt):]
    rejected = tokenizer.apply_chat_template(example["prompt"] + example['rejected'],tools=example['tools'], tokenize=False, add_generation_prompt=False)
    rejected = rejected[len(prompt):]
    return {"prompt": prompt, "chosen": chosen, "rejected": rejected}
    


class DPOTrainerWithSegments(DPOTrainer):
    def __init__(self, *args, tool_tokens,**kwargs):
        super().__init__(*args, **kwargs)
        self.start,self.end = tool_tokens
        self.start_ids = self.processing_class.encode(self.start, add_special_tokens=False)
        self.end_ids = self.processing_class.encode(self.end, add_special_tokens=False)
        self.printed = False

    def _set_signature_columns_if_needed(self):
        super()._set_signature_columns_if_needed()
        # keep your extra dataset columns so Trainer doesn't drop them
        self._signature_columns += [
            "segment_ids",
        ]

    def concatenated_inputs(self,batch, padding_value: int):
        out = DPOTrainer.concatenated_inputs(batch, padding_value)
        input_ids = out["prompt_input_ids"]
        segment_ids = torch.ones_like(input_ids, dtype=torch.long)
        
        start_ids = torch.tensor(self.start_ids, device=input_ids.device)
        end_ids = torch.tensor(self.end_ids, device=input_ids.device)
        for idx, inp_ids in enumerate(input_ids):
            i = 0
            spans = []
            mask = torch.zeros(len(inp_ids), dtype=torch.bool)
            while i <= len(inp_ids) - len(start_ids):
                if torch.equal(inp_ids[i:i+len(start_ids)], start_ids):
                    # locate matching end after this start
                    j = i + len(start_ids)
                    found_end = False
                    while j <= len(inp_ids) - len(end_ids):
                        if torch.equal(inp_ids[j:j+len(end_ids)], end_ids):
                            spans.append((i+len(start_ids), j+len(end_ids))) # span is between start and end
                            found_end = True
                            i = j + len(end_ids)
                            break
                        j += 1
                    if not found_end:
                        # No closing tag; don't label this open span to avoid corruption
                        i += len(start_ids)
                else:
                    i += 1
            if len(spans):
                for span in spans:
                    mask[span[0]:span[1]] = True
                segment_ids[idx,~mask] = 0
        completion_input_ids = out["completion_input_ids"]
        completion_len = completion_input_ids.size(1)
        segment_ids = torch.cat([segment_ids, torch.zeros((segment_ids.size(0), completion_len), dtype=torch.long, device=segment_ids.device)], dim=1)
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