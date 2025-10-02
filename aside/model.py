"""
ASIDE Model Architectures Implementation

This module contains the core architectural implementations for the ASIDE paper experiments:

1. ForwardRotMixin: Main ASIDE method (forward_rot) with π/2 rotation
2. ISEMixin: ISE baseline method with learnable segment embeddings  
3. Model-specific classes: Llama, Qwen, Mistral variants
4. Legacy CustomLLaMA: Double embedding approach (double_emb)

Key Architecture Types:
- forward_rot: ASIDE method - applies orthogonal rotation to data token embeddings
- ise: ISE baseline - adds learnable segment embeddings to token embeddings
- single_emb: Vanilla baseline - standard model without architectural changes
- double_emb: Legacy approach - separate embedding matrices for instructions/data

The implementations follow the paper's methodology for creating distinct embedding
subspaces for instruction and data tokens 

References:
    ASIDE: Architectural Separation of Instructions and Data in Language Models
    Section 3: Architecturally Separated Instruction-Data Embeddings
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    PretrainedConfig,
)

from embeddings_init import (
    generate_isoclinic_rotation_matrix,
    rotate_embeddings_in_multiple_planes,
    rotate_embeddings_independently,
)

from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaForCausalLM

def texts_to_prepared_ids(
    text_sequences, tokenizer, max_length=512, model_type="double_emb"
):
    """
    Prepares tokenized inputs with instruction/data separation for ASIDE models.
    
    This function processes sequences of (text, type) pairs and creates the appropriate 
    input representations based on the model architecture. For ASIDE and ISE models, 
    it generates segment_ids that determine which tokens receive rotated embeddings (ASIDE)
    or additional linear embeddings (ISE).
    
    Args:
        text_sequences (List[Tuple[str, str]]): List of (text, role) pairs where:
                                               - text: The actual text content
                                               - role: "inst" for instructions, "data" for data
        tokenizer: HuggingFace tokenizer for the model
        max_length (int): Maximum sequence length (default: 512)
        model_type (str): Architecture type:
                         - "forward_rot": ASIDE method (main contribution)
                         - "ise": ISE baseline method
                         - "single_emb": Vanilla baseline
                         - "double_emb": Legacy double embedding
    
    Returns:
        tuple: (input_ids, attention_mask, segment_ids)
               - input_ids: Token IDs for the model
               - attention_mask: Attention mask (1 for real tokens, 0 for padding)
               - segment_ids: Role indicators (0 for instructions, 1 for data)
                             Used by ASIDE and ISE to determine embedding treatment
    
    Note:
        segment_ids are crucial for ASIDE as they determine which tokens get
        rotated embeddings during the forward pass. Only tokens with segment_id=1
        (data tokens) receive the π/2 rotation.
        
    Example:
        >>> sequences = [("Translate this text:", "inst"), ("Hello world", "data")]
        >>> ids, mask, segments = texts_to_prepared_ids(sequences, tokenizer, 512, "forward_rot")
    """
    remaining_length = max_length
    tokenized_seq = []

    for text_seq in text_sequences:
        # Tokenize this sequence with the remaining allowance
        tokenized = tokenizer(
            text_seq[0],
            return_tensors="pt",
            max_length=remaining_length,
            padding="longest",
            truncation=True,
        )

        tokenized_seq.append(tokenized)

        # Count how many tokens we actually used (sequence length)
        seq_len = tokenized["input_ids"].shape[1]

        # Subtract from our remaining length
        remaining_length -= seq_len

        # If there's no room left for any more tokens, break
        if remaining_length <= 0:
            break


    if model_type == "single_emb":
        assert len(text_sequences) == 1
        input_ids, attention_mask = (
            tokenized_seq[0]["input_ids"],
            tokenized_seq[0]["attention_mask"],
        )
        segment_ids = None
    elif model_type == "double_emb":
        token_sequences = [
            (tokenized_seq[i]["input_ids"], text_sequences[i][1])
            for i in range(len(tokenized_seq))
        ]
        input_ids, attention_mask = prepare_input_ids(token_sequences, tokenizer)
        segment_ids = None
    elif model_type in ("ise", "forward_rot"):
        token_sequences = [
            (tokenized_seq[i]["input_ids"], text_sequences[i][1])
            for i in range(len(tokenized_seq))
        ]
        input_ids = torch.hstack(
            [tokenized_seq[i]["input_ids"] for i in range(len(tokenized_seq))]
        )
        input_ids[1:] = torch.where(
            input_ids[1:] == tokenizer.bos_token_id,
            torch.full_like(input_ids[1:], tokenizer.pad_token_id),
            input_ids[1:],
        )
        attention_mask = torch.hstack(
            [tokenized_seq[i]["attention_mask"] for i in range(len(tokenized_seq))]
        )
        segment_ids = torch.hstack(
            [
                torch.zeros_like(ts[0]) if (ts[1] == "inst") else torch.ones_like(ts[0])
                for ts in token_sequences
            ]
        )
    else:
        raise NotImplementedError(f"Not implemented for model type {model_type}")
  
    return input_ids, attention_mask, segment_ids  # .to(device)


def prepare_input_ids(token_sequences, tokenizer):
    """
    Prepares input for double embedding models (legacy approach).
    
    This function handles the legacy double embedding approach where separate
    embedding matrices are used for instruction and data tokens. Data token IDs
    are shifted by vocab_size to index into the second embedding matrix.
    
    Args:
        token_sequences (List[Tuple[torch.Tensor, str]]): List of (input_ids, role) pairs
        tokenizer: HuggingFace tokenizer
        
    Returns:
        tuple: (input_ids, attention_mask) with shifted token IDs for data tokens
        
    Note:
        This is a legacy approach superseded by ASIDE's rotation method.
        Data tokens get their IDs shifted by vocab_size to access separate embeddings.
    """
    pad_token_id = tokenizer.pad_token_id
    bos_token_id = tokenizer.bos_token_id
    vocab_size = len(
        tokenizer
    )  # tokenizer.vocab_size # OMG, why in the name of Ctulhu the vocab_size variable is not updated
    declared_vocab_size = tokenizer.vocab_size
    specials_vocab_size = vocab_size - declared_vocab_size
    if bos_token_id is None:
        raise ValueError(
            "bos_token_id is None. Please ensure the tokenizer has a bos_token_id set."
        )

    # Ensure all input_ids have the same batch size
    batch_size = token_sequences[0][0].size(0)
    for input_ids, _ in token_sequences:
        assert input_ids.size(0) == batch_size, (
            "All input_ids must have the same batch size"
        )

    # List to collect processed sequences for each item in the batch
    batch_input_ids = []
    batch_attention_masks = []

    for batch_idx in range(batch_size):
        non_pad_tokens_list = []
        total_pad_count = 0

        for seq_idx, (input_ids, token_type) in enumerate(token_sequences):
            # Select the sequence for the current batch item
            seq = input_ids[batch_idx].clone()  # Shape: [seq_len]

            # Determine the pad_token_id and bos_token_id for this sequence
            if token_type == "data":
                # Shift tokens and pad_token_id
                seq = seq + vocab_size
                # TODO: don't shift he tokens?
                seq_pad_token_id = pad_token_id + vocab_size
                seq_bos_token_id = (
                    bos_token_id + vocab_size if bos_token_id is not None else None
                )
            else:
                seq_pad_token_id = pad_token_id
                seq_bos_token_id = bos_token_id
            bos_positions = (
                (seq == seq_bos_token_id).nonzero(as_tuple=False).squeeze(-1)
            )
            if bos_positions.numel() > 0:
                bos_position = bos_positions[0].item()
                non_pad_tokens = seq[bos_position + 1 :]
            else:
                # Handle the case where bos_token is not found
                bos_position = -1
                non_pad_tokens = seq
            # Replace bos_token with pad_token in all sequences except the first
            if seq_idx > 0 and seq_bos_token_id is not None:
                seq[seq == seq_bos_token_id] = seq_pad_token_id

            # Count padding tokens
            pad_count = seq.size(0) - non_pad_tokens.size(0)
            total_pad_count += pad_count

            # Append non-padding tokens to the list
            non_pad_tokens_list.append(non_pad_tokens)

        # Concatenate all non-padding tokens for this batch item
        concatenated_non_pad_tokens = torch.cat(non_pad_tokens_list, dim=0)

        # Build final input_ids with padding tokens at the beginning
        input_ids = torch.cat(
            [
                torch.full(
                    (total_pad_count,),
                    pad_token_id,
                    dtype=concatenated_non_pad_tokens.dtype,
                    device=concatenated_non_pad_tokens.device,
                ),
                concatenated_non_pad_tokens,
            ],
            dim=0,
        )  # Shape: [total_pad_count + non_pad_tokens_len]

        # Build attention_mask: 0 for padding tokens, 1 for non-padding tokens
        attention_mask = torch.cat(
            [
                torch.zeros(
                    total_pad_count,
                    dtype=torch.bool,
                    device=concatenated_non_pad_tokens.device,
                ),
                torch.ones(
                    concatenated_non_pad_tokens.size(0),
                    dtype=torch.bool,
                    device=concatenated_non_pad_tokens.device,
                ),
            ],
            dim=0,
        )

        batch_input_ids.append(input_ids)
        batch_attention_masks.append(attention_mask)

    # Find the maximum sequence length in the batch
    input_ids = torch.cat([elem.unsqueeze(0) for elem in batch_input_ids], dim=0)
    if specials_vocab_size > 0:
        input_ids[input_ids >= declared_vocab_size * 2 + specials_vocab_size] %= (
            vocab_size
        )

    attention_mask = torch.cat(
        [elem.unsqueeze(0) for elem in batch_attention_masks], dim=0
    )
    return input_ids, attention_mask



########################

# Base Classes

########################


class ISEMixin:
    """
    ISE (Instructional Segment Embedding) baseline implementation.
    
    This mixin implements the ISE method from Wu et al. (2024) as a baseline
    comparison to ASIDE. ISE adds learnable segment embeddings to token embeddings
    based on whether tokens are part of instructions or data.
    
    Key Differences from ASIDE:
    - Uses learnable parameters (segment embeddings) vs ASIDE's parameter-free rotation
    - Additive combination vs ASIDE's geometric transformation
    - Less effective separation in deeper layers (as shown in paper Section 6)
    
    Architecture:
        final_embedding = token_embedding + segment_embedding(segment_id)
        where segment_id = 0 for instructions, 1 for data
    """
    def forward(self, *args, input_ids=None, segment_ids=None, labels=None, **kwargs):
        """
        Forward pass with ISE segment embedding addition.
        
        Args:
            input_ids (torch.Tensor, optional): Token IDs 
            segment_ids (torch.Tensor): Segment role indicators (0=instruction, 1=data)
            labels (torch.Tensor, optional): Labels for language modeling loss
            **kwargs: Additional arguments passed to parent forward
            
        Returns:
            Model outputs with ISE-modified embeddings
            
        Note:
            segment_ids are required for ISE models to determine which tokens
            receive which type of segment embedding.
        """
        if segment_ids is None:
            raise ValueError("For ISE models, segment_ids can't be None")

        # Get existing inputs_embeds if provided
        inputs_embeds = kwargs.pop("inputs_embeds", None)

        # Handle different input scenarios
        if input_ids is not None:
            # When using single token generation, adjust segment_ids accordingly
            if input_ids.shape[1] == 1:
                segment_ids = segment_ids[:, -1:]

            if segment_ids.shape != input_ids.shape:
                raise Exception(
                    f"Mismatched shapes: segment_ids {segment_ids.shape}, input_ids {input_ids.shape}"
                )

            # Get token embeddings if inputs_embeds not already provided
            if inputs_embeds is None:
                token_embeddings = self.model.embed_tokens(input_ids)
                seg_embeddings = self.segment_embedding(segment_ids)
                inputs_embeds = token_embeddings + seg_embeddings

        # Handle the case when only inputs_embeds is provided (e.g., from adaptive_attack.py)
        elif inputs_embeds is not None:
            # Check segment_ids shape matches inputs_embeds shape
            if segment_ids.shape[:2] != inputs_embeds.shape[:2]:
                raise ValueError(
                    f"Segment IDs shape {segment_ids.shape} doesn't match inputs_embeds shape {inputs_embeds.shape[:2]}"
                )

            # Add segment embeddings to the provided inputs_embeds
            seg_embeddings = self.segment_embedding(segment_ids)
            inputs_embeds = inputs_embeds + seg_embeddings
        else:
            raise ValueError("Either input_ids or inputs_embeds must be provided")

        # Remove extra keywords
        kwargs.pop("segment_ids", None)
        if hasattr(self.model, "delete_num_items_in_batch"):
            kwargs.pop("num_items_in_batch", None)

        # Forward to parent class
        outputs = super().forward(
            *args, input_ids=None, inputs_embeds=inputs_embeds, labels=labels, **kwargs
        )
        return outputs


class ForwardRotMixin:
    """
    ASIDE (forward_rot) method implementation.
    
    This mixin implements the core ASIDE architecture that applies orthogonal rotations
    to data token embeddings. It creates geometrically separated embedding subspaces
    for instructions and data without adding any learnable parameters.
    
    Key Features:
    - Parameter-free: Uses fixed orthogonal rotation matrices (typically π/2)
    - Geometric separation: Creates distinct embeddings via isoclinic rotation
    - Conditional application: Only rotates embeddings where segment_ids == 1 (data)
    
    Mathematical Operation:
        For data tokens (segment_id == 1):
            rotated_embedding = embedding @ rotation_matrix 
        For instruction tokens (segment_id == 0):
            embedding remains unchanged
            
    This implements Section 3 of the ASIDE paper: "Architecturally Separated 
    Instruction-Data Embeddings"
    """
    def forward(self, *args, input_ids=None, segment_ids=None, labels=None, **kwargs):
        """
        Forward pass with ASIDE rotation applied to data embeddings.
        
        This is the core ASIDE forward pass that applies π/2 rotation to data token
        embeddings while leaving instruction token embeddings unchanged. This creates
        the geometric separation that enables better instruction-data distinction.
        
        Args:
            input_ids (torch.Tensor, optional): Token IDs for embedding lookup
            segment_ids (torch.Tensor): Role indicators (0=instruction, 1=data)
                                       Determines which tokens get rotated
            labels (torch.Tensor, optional): Labels for language modeling loss
            **kwargs: Additional arguments, including optional inputs_embeds
            
        Returns:
            Model outputs with ASIDE-rotated embeddings for data tokens
            
        Note:
            The rotation is only applied where segment_ids == 1 (data tokens).
            Instruction tokens (segment_ids == 0) use standard embeddings.
            This selective application is key to ASIDE's effectiveness.
        """
        if segment_ids is None:
            raise ValueError("For ForwardRot models, segment_ids can't be None")

        inputs_embeds = kwargs.get("inputs_embeds", None)

        # Case 1: We have input_ids but no inputs_embeds - the original case
        if input_ids is not None:
            if (
                input_ids.shape[1] == 1
            ):  # Needed when using cache, since input_ids is [B, 1]
                # We are in a single-token decode step
                # Slice or build segment_ids so it's also [B, 1]
                single_token_segment = segment_ids[:, -1:]  # last token's segment
                segment_ids = single_token_segment

            if segment_ids.shape != input_ids.shape:
                raise Exception(
                    f"Mismatched shape of segmend ids:{segment_ids.shape, input_ids.shape}"
                )

            if inputs_embeds is None:
                inputs_embeds = self.model.embed_tokens(input_ids)

                # Only rotate where segment_ids == 1
                mask = segment_ids == 1

                if self.rotation_direction == "right":
                    new_embeds = inputs_embeds.clone()
                    new_embeds[mask] = torch.matmul(
                        inputs_embeds[mask], self.rotation_matrix
                    )
                    inputs_embeds = new_embeds
                elif self.rotation_direction == "left":
                    new_embeds = inputs_embeds.clone()
                    new_embeds[mask] = (self.rotation_matrix @ inputs_embeds[mask].T).T
                    inputs_embeds = new_embeds
                else:
                    raise ValueError(
                        f"Unknown rotation_direction: {self.rotation_direction}"
                    )

                if self.add_linear_shift:
                    seg_embeddings = self.segment_embedding(segment_ids)
                    inputs_embeds = inputs_embeds + seg_embeddings

        # Case 2: We have inputs_embeds but no input_ids - handle manually provided embeddings
        elif inputs_embeds is not None:
            # Check that segment_ids has the right shape
            if segment_ids.shape != inputs_embeds.shape[:2]:
                raise Exception(
                    f"Mismatched shape of segment_ids:{segment_ids.shape} vs inputs_embeds:{inputs_embeds.shape[:2]}"
                )

            # Apply rotation to embeddings where segment_ids == 1
            mask = segment_ids == 1

            # Create a copy to avoid modifying the input
            new_embeds = inputs_embeds.clone()

            if self.rotation_direction == "right":
                new_embeds[mask] = torch.matmul(
                    inputs_embeds[mask], self.rotation_matrix
                )
            elif self.rotation_direction == "left":
                new_embeds[mask] = (self.rotation_matrix @ inputs_embeds[mask].T).T
            else:
                raise ValueError(
                    f"Unknown rotation_direction: {self.rotation_direction}"
                )

            inputs_embeds = new_embeds

            if self.add_linear_shift:
                seg_embeddings = self.segment_embedding(segment_ids)
                inputs_embeds = inputs_embeds + seg_embeddings

        # Case 3: Neither input_ids nor inputs_embeds provided
        else:
            raise ValueError("Either input_ids or inputs_embeds must be provided")

        # Remove inputs we've already processed
        kwargs.pop("inputs_embeds", None)
        kwargs.pop("segment_ids", None)

        if hasattr(self.model, "delete_num_items_in_batch"):
            kwargs.pop("num_items_in_batch", None)

        outputs = super().forward(
            *args, input_ids=None, inputs_embeds=inputs_embeds, labels=labels, **kwargs
        )

        return outputs


##################

# LLaMa

##################
class CustomLlamaConfig(LlamaConfig):
    """Extended Llama configuration for ASIDE experiments."""

    # Define a unique model_type so that your custom class is instantiated
    model_type = "llama"

    def __init__(self, **kwargs):
        # Call the parent initializer
        super().__init__(**kwargs)


class LlamaBase(LlamaForCausalLM):
    """
    Base Llama model with tokenizer customization for ASIDE experiments.
    
    This base class handles tokenizer setup for Llama models, ensuring proper
    padding and special token configuration across Llama-2 and Llama-3 variants.
    """
    def __init__(self, config: LlamaConfig):
        super().__init__(config)

    @classmethod
    def _customize_tokenizer(cls, tokenizer, model_name_or_path):
        """
        Customizes tokenizer for ASIDE experiments with proper special tokens.
        
        Args:
            tokenizer: HuggingFace tokenizer to customize
            model_name_or_path (str): Model path to determine Llama version
            
        Sets up:
        - Padding tokens (different for Llama-2 vs Llama-3)
        - EOS tokens with proper IDs
        - Ensures compatibility across model variants
        """
        if tokenizer.pad_token is None:
            if (
                "llama-3" in model_name_or_path.lower()
                or "llama_3" in model_name_or_path.lower()
            ):
                tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(
                    "<|reserved_special_token_3|>"
                )
            elif (
                "llama-2" in model_name_or_path.lower()
                or "llama_2" in model_name_or_path.lower()
            ):
                tokenizer.pad_token_id = tokenizer.unk_token_id
                tokenizer.pad_token = tokenizer.unk_token
            else:
                raise NotImplementedError("Pad token supports only llama now, fix")
        else:
            tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(
                tokenizer.pad_token
            )
        if tokenizer.eos_token_id is None:
            if (
                "llama-3" in model_name_or_path.lower()
                or "llama_3" in model_name_or_path.lower()
            ):
                tokenizer.eos_token = "<|end_of_text|>"
                tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids(
                    "<|end_of_text|>"
                )
            elif (
                "llama-2" in model_name_or_path.lower()
                or "llama_2" in model_name_or_path.lower()
            ):
                tokenizer.eos_token = "</s>"
                tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids("</s>")
            else:
                raise NotImplementedError(
                    "Pad token supports only llama 3 and 2 now, fix"
                )


class LlamaISE(ISEMixin, LlamaBase):
    """
    Llama model with ISE baseline implementation.
    
    Combines the Llama base model with ISE segment embeddings for baseline
    comparison with ASIDE. Used in paper experiments as one of the baselines.
    """
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.num_segments = getattr(config, "num_segments", 2)
        self.segment_embedding = nn.Embedding(self.num_segments, config.hidden_size)


class LlamaForwardRot(ForwardRotMixin, LlamaBase):
    """
    Llama model with ASIDE (ForwardRot) implementation - Main experimental model.
    
    This is the main ASIDE implementation for Llama models used in the paper.
    It applies π/2 orthogonal rotations to data token embeddings while keeping
    instruction embeddings unchanged.
    
    Key Configuration:
    - rotation_alpha: Rotation angle (π/2 for standard ASIDE)
    - gradual_rotation: Whether to gradually increase rotation during training
    - learned_rotation: Whether rotation matrix is learnable (typically False)
    - rotation_direction: "right" or "left" matrix multiplication
    - add_linear_shift: Whether to add ISE-style segment embeddings (typically False)
    """
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.config = config
        dim = config.hidden_size
        self.global_rotation_alpha = config.rotation_alpha
        self.rotation_alpha = None
        self.gradual_rotation = getattr(config, "gradual_rotation", False)
        self.learned_rotation = getattr(config, "learned_rotation", False)
        self.add_linear_shift = getattr(config, "add_linear_shift", False)
        self.rotation_direction = getattr(self.config, "rotation_direction", "right")
        device = next(self.parameters()).device
        model_dtype = next(self.parameters()).dtype

        # print(f"device {device}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.add_linear_shift:
            self.num_segments = getattr(config, "num_segments", 2)
            self.segment_embedding = nn.Embedding(self.num_segments, config.hidden_size)

        if self.gradual_rotation:
            rotation_matrix = generate_isoclinic_rotation_matrix(
                dim, 0, device, model_dtype
            ).detach()
        else:
            rotation_matrix = generate_isoclinic_rotation_matrix(
                dim, self.global_rotation_alpha, device, model_dtype
            ).detach()
        if self.learned_rotation:
            self.rotation_matrix = nn.Parameter(rotation_matrix)
        else:
            self.register_buffer("rotation_matrix", rotation_matrix,persistent=False)


###########

# Qwen

###########
from transformers import Qwen2Config, Qwen2ForCausalLM,Qwen3Config, Qwen3ForCausalLM


class CustomQwenConfig(Qwen2Config):
    """Extended Qwen configuration for ASIDE experiments."""

    # Use the same model_type as the base configuration (unless you need a unique one)
    model_type = "qwen2"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class QwenBase(Qwen2ForCausalLM):
    """Base Qwen model with tokenizer customization."""

    def __init__(self, config: Qwen2Config):
        super().__init__(config)

    @classmethod
    def _customize_tokenizer(cls, tokenizer, model_path):
        # Example tokenizer customization for Qwen:
        tokenizer.pad_token = "<|endoftext|>"
        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")
        tokenizer.bos_token = "<|endoftext|>"
        tokenizer.bos_token_id = tokenizer.convert_tokens_to_ids(tokenizer.bos_token)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token


class QwenISE(ISEMixin, QwenBase):
    """Qwen model with ISE baseline implementation."""

    def __init__(self, config: Qwen2Config):
        super().__init__(config)
        # Set up segment embeddings for the ISE variant.
        self.num_segments = getattr(config, "num_segments", 2)
        self.segment_embedding = nn.Embedding(self.num_segments, config.hidden_size)


class QwenForwardRot(ForwardRotMixin, QwenBase):
    """Qwen model with ASIDE implementation."""

    def __init__(self, config: Qwen2Config):
        super().__init__(config)
        self.config = config
        dim = config.hidden_size
        self.global_rotation_alpha = config.rotation_alpha
        self.rotation_alpha = None
        self.gradual_rotation = getattr(config, "gradual_rotation", False)
        self.learned_rotation = getattr(config, "learned_rotation", False)
        self.add_linear_shift = getattr(config, "add_linear_shift", False)
        self.rotation_direction = getattr(self.config, "rotation_direction", "right")
        device = next(self.parameters()).device
        model_dtype = next(self.parameters()).dtype

        # print(f"device {device}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.add_linear_shift:
            self.num_segments = getattr(config, "num_segments", 2)
            self.segment_embedding = nn.Embedding(self.num_segments, config.hidden_size)

        if self.gradual_rotation:
            rotation_matrix = generate_isoclinic_rotation_matrix(
                dim, 0, device, model_dtype
            ).detach()
        else:
            rotation_matrix = generate_isoclinic_rotation_matrix(
                dim, self.global_rotation_alpha, device, model_dtype
            ).detach()
        if self.learned_rotation:
            self.rotation_matrix = nn.Parameter(rotation_matrix)
        else:
            self.register_buffer("rotation_matrix", rotation_matrix, persistent=False)

class CustomQwen3Config(Qwen3Config):
    """Extended Qwen configuration for ASIDE experiments."""

    # Use the same model_type as the base configuration (unless you need a unique one)
    model_type = "qwen3"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class Qwen3Base(Qwen3ForCausalLM):
    """Base Qwen model with tokenizer customization."""

    def __init__(self, config: Qwen3Config):
        super().__init__(config)

    @classmethod
    def _customize_tokenizer(cls, tokenizer, model_path):
        # Example tokenizer customization for Qwen:
        tokenizer.pad_token = "<|endoftext|>"
        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")
        tokenizer.bos_token = "<|endoftext|>"
        tokenizer.bos_token_id = tokenizer.convert_tokens_to_ids(tokenizer.bos_token)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    
            
class Qwen3Base_StruQ(Qwen3ForCausalLM):
    """Base Qwen model with tokenizer customization."""

    def __init__(self, config: Qwen3Config):
        super().__init__(config)

    @classmethod
    def _customize_tokenizer_and_model(cls, tokenizer, model): # from StruQ
        ## check if the special tokens are already in, if yes then skip
        tokenizer.padding_side = "left"
        SPECIAL_DELM_TOKENS = ['[INST]',      '[INPT]', '[RESP]',   '[MARK]', '[COLN]']
        if not all(token in tokenizer.get_vocab() for token in SPECIAL_DELM_TOKENS): 
            num_new_tokens = tokenizer.add_special_tokens({
            'pad_token': '[PAD]',
            'additional_special_tokens': SPECIAL_DELM_TOKENS
            })
            model.resize_token_embeddings(len(tokenizer))
            delimiter_init_embed_index_from_text = [tokenizer.encode(v, add_special_tokens=False)[0] for v in ['instruction', 'input',  'response', '###',    ':']]
            input_embeddings = model.get_input_embeddings().weight.data
            output_embeddings = model.get_output_embeddings().weight.data
            # Initialize the [PAD] token with the mean of all embeddings
            input_embeddings[-num_new_tokens] = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
            output_embeddings[-num_new_tokens] = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

            # Initialize the 5 StruQ delimiters with the embeddings of the corresponding textual delimiters
            for i in range(len(SPECIAL_DELM_TOKENS)):
                index = -num_new_tokens+i+1
                print('Initialize special delimiter token', tokenizer.decode([len(tokenizer) + index]), 'from the embedding of', tokenizer.decode([delimiter_init_embed_index_from_text[i]]))
                input_embeddings[index] = input_embeddings[delimiter_init_embed_index_from_text[i]]
                output_embeddings[index] = output_embeddings[delimiter_init_embed_index_from_text[i]]

            tokenizer.bos_token = '<s>'
            tokenizer.bos_token_id = tokenizer.convert_tokens_to_ids('<s>')
        

class Qwen3ISE(ISEMixin, Qwen3Base):
    """Qwen model with ISE baseline implementation."""

    def __init__(self, config: Qwen3Config):
        super().__init__(config)
        # Set up segment embeddings for the ISE variant.
        self.num_segments = getattr(config, "num_segments", 2)
        self.segment_embedding = nn.Embedding(self.num_segments, config.hidden_size)
        
class Qwen3ISE_StruQ(ISEMixin, Qwen3Base_StruQ):
    """Qwen model with ISE baseline implementation."""

    def __init__(self, config: Qwen3Config):
        super().__init__(config)
        # Set up segment embeddings for the ISE variant.
        self.num_segments = getattr(config, "num_segments", 2)
        self.segment_embedding = nn.Embedding(self.num_segments, config.hidden_size)


class Qwen3ForwardRot(ForwardRotMixin, Qwen3Base):
    """Qwen model with ASIDE implementation."""

    def __init__(self, config: Qwen3Config):
        super().__init__(config)
        self.config = config
        dim = config.hidden_size
        self.global_rotation_alpha = config.rotation_alpha
        self.rotation_alpha = None
        self.gradual_rotation = getattr(config, "gradual_rotation", False)
        self.learned_rotation = getattr(config, "learned_rotation", False)
        self.add_linear_shift = getattr(config, "add_linear_shift", False)
        self.rotation_direction = getattr(self.config, "rotation_direction", "right")
        device = next(self.parameters()).device
        model_dtype = next(self.parameters()).dtype

        # print(f"device {device}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.add_linear_shift:
            self.num_segments = getattr(config, "num_segments", 2)
            self.segment_embedding = nn.Embedding(self.num_segments, config.hidden_size)

        if self.gradual_rotation:
            rotation_matrix = generate_isoclinic_rotation_matrix(
                dim, 0, device, model_dtype
            ).detach()
        else:
            rotation_matrix = generate_isoclinic_rotation_matrix(
                dim, self.global_rotation_alpha, device, model_dtype
            ).detach()
        if self.learned_rotation:
            self.rotation_matrix = nn.Parameter(rotation_matrix)
        else:
            self.register_buffer("rotation_matrix", rotation_matrix, persistent=False)


class Qwen3ForwardRot_StruQ(ForwardRotMixin, Qwen3Base_StruQ):
    """Qwen model with ASIDE implementation."""

    def __init__(self, config: Qwen3Config):
        super().__init__(config)
        self.config = config
        dim = config.hidden_size
        self.global_rotation_alpha = config.rotation_alpha
        self.rotation_alpha = None
        self.gradual_rotation = getattr(config, "gradual_rotation", False)
        self.learned_rotation = getattr(config, "learned_rotation", False)
        self.add_linear_shift = getattr(config, "add_linear_shift", False)
        self.rotation_direction = getattr(self.config, "rotation_direction", "right")
        device = next(self.parameters()).device
        model_dtype = next(self.parameters()).dtype

        # print(f"device {device}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.add_linear_shift:
            self.num_segments = getattr(config, "num_segments", 2)
            self.segment_embedding = nn.Embedding(self.num_segments, config.hidden_size)

        if self.gradual_rotation:
            rotation_matrix = generate_isoclinic_rotation_matrix(
                dim, 0, device, model_dtype
            ).detach()
        else:
            rotation_matrix = generate_isoclinic_rotation_matrix(
                dim, self.global_rotation_alpha, device, model_dtype
            ).detach()
        if self.learned_rotation:
            self.rotation_matrix = nn.Parameter(rotation_matrix)
        else:
            self.register_buffer("rotation_matrix", rotation_matrix, persistent=False)


###########

# Mistral

###########
from transformers import MistralConfig, MistralForCausalLM


class CustomMistralConfig(MistralConfig):
    """Extended Mistral configuration for ASIDE experiments."""

    # Use the base model_type to ensure compatibility
    model_type = "mistral"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class MistralBase(MistralForCausalLM):
    """Base Mistral model with tokenizer customization."""

    def __init__(self, config: MistralConfig):
        super().__init__(config)
        self.delete_num_items_in_batch = True

    @classmethod
    def _customize_tokenizer(cls, tokenizer, model_path):
        # Customize the tokenizer for Mistral
        tokenizer.pad_token = "<pad>"
        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids("<pad>")
        tokenizer.bos_token = "<s>"
        tokenizer.bos_token_id = tokenizer.convert_tokens_to_ids("<s>")
        tokenizer.eos_token = "</s>"
        tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids("</s>")
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token


class MistralISE(ISEMixin, MistralBase):
    """Mistral model with ISE baseline implementation."""

    def __init__(self, config: MistralConfig):
        super().__init__(config)
        # Set up segment embeddings for the ISE variant
        self.num_segments = getattr(config, "num_segments", 2)
        self.segment_embedding = nn.Embedding(self.num_segments, config.hidden_size)


class MistralForwardRot(ForwardRotMixin, MistralBase):
    """Mistral model with ASIDE implementation."""

    def __init__(self, config: MistralConfig):
        super().__init__(config)
        self.config = config
        dim = config.hidden_size
        self.global_rotation_alpha = config.rotation_alpha
        self.rotation_alpha = None
        self.gradual_rotation = getattr(config, "gradual_rotation", False)
        self.learned_rotation = getattr(config, "learned_rotation", False)
        self.add_linear_shift = getattr(config, "add_linear_shift", False)
        self.rotation_direction = getattr(self.config, "rotation_direction", "right")
        device = next(self.parameters()).device
        model_dtype = next(self.parameters()).dtype

        # print(f"device {device}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.add_linear_shift:
            self.num_segments = getattr(config, "num_segments", 2)
            self.segment_embedding = nn.Embedding(self.num_segments, config.hidden_size)

        if self.gradual_rotation:
            rotation_matrix = generate_isoclinic_rotation_matrix(
                dim, 0, device, model_dtype
            ).detach()
        else:
            rotation_matrix = generate_isoclinic_rotation_matrix(
                dim, self.global_rotation_alpha, device, model_dtype
            ).detach()
        if self.learned_rotation:
            self.rotation_matrix = nn.Parameter(rotation_matrix)
        else:
            self.register_buffer("rotation_matrix", rotation_matrix, persistent=False)


###########
# Gemma Model Implementations
###########
import os
from transformers import (
    Gemma3ForCausalLM,
    Gemma3TextConfig,)



class CustomGemmaConfig(Gemma3TextConfig):
    """Extended Gemma configuration for ASIDE experiments."""
    model_type = "gemma3_text"


class GemmaBase(Gemma3ForCausalLM):
    """Base Gemma model with configuration handling."""
    
    def __init__(self, config: Gemma3TextConfig):
        config.attention_bias = True
        super().__init__(config)
        print("Config in GemmaBase init:", config)

    @staticmethod
    def _customize_tokenizer(tokenizer, model_path):
        """Customizes Gemma tokenizer for ASIDE experiments."""
        tokenizer.pad_token = "<pad>"
        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids("<pad>")
        tokenizer.bos_token = "<s>"
        tokenizer.bos_token_id = tokenizer.convert_tokens_to_ids("<s>")
        tokenizer.eos_token = "</s>"
        tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids("</s>")
        tokenizer.padding_side = "left"
        print("Special tokens", tokenizer.convert_tokens_to_ids(["<pad>", "<s>", "</s>"]))
        assert tokenizer.convert_tokens_to_ids("<pad>") >= 0, "Invalid <pad> token"


class GemmaISE(ISEMixin, GemmaBase):
    """Gemma model with ISE baseline implementation."""
    
    def __init__(self, config: Gemma3TextConfig):
        super().__init__(config)
        self.num_segments = getattr(config, "num_segments", 2)
        self.segment_embedding = nn.Embedding(self.num_segments, config.hidden_size)


class GemmaForwardRot(ForwardRotMixin, GemmaBase):
    """Gemma model with ASIDE implementation."""
    
    def __init__(self, config: Gemma3TextConfig):
        super().__init__(config)
        self.config = config
        dim = config.hidden_size
        self.global_rotation_alpha = config.rotation_alpha
        self.rotation_alpha = None
        self.gradual_rotation = getattr(config, "gradual_rotation", False)
        self.learned_rotation = getattr(config, "learned_rotation", False)
        self.add_linear_shift = getattr(config, "add_linear_shift", False)
        self.rotation_direction = getattr(self.config, "rotation_direction", "right")
        device = next(self.parameters()).device
        model_dtype = next(self.parameters()).dtype

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.add_linear_shift:
            self.num_segments = getattr(config, "num_segments", 2)
            self.segment_embedding = nn.Embedding(self.num_segments, config.hidden_size)

        if self.gradual_rotation:
            rotation_matrix = generate_isoclinic_rotation_matrix(
                dim, 0, device, model_dtype
            ).detach()
        else:
            rotation_matrix = generate_isoclinic_rotation_matrix(
                dim, self.global_rotation_alpha, device, model_dtype
            ).detach()
            
        if self.learned_rotation:
            self.rotation_matrix = nn.Parameter(rotation_matrix)
        else:
            self.register_buffer("rotation_matrix", rotation_matrix, persistent=False)


###########
# Legacy Double Embedding Implementation  
###########



class CustomLlamaConfig(LlamaConfig):
    model_type = "custom_llama"  # Set to match the name used in AutoConfig.register

    # model_type = "llama"  # or "custom_llama" if you properly register that name
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.original_vocab_size = kwargs.get(
            "original_vocab_size", self.vocab_size // 2
        )
        self.rotation_alpha = kwargs.get("rotation_alpha", None)


class CustomLLaMA(LlamaForCausalLM):
    """
    Legacy double embedding implementation (double_emb).
    
    This class implements the double embedding approach where separate embedding
    matrices are used for instruction and data tokens. This approach predates
    ASIDE and is less efficient but serves as a baseline in experiments.
    
    Architecture:
    - Vocabulary is doubled: tokens 0-V for instructions, V-2V for data
    - Data token IDs are shifted by vocab_size to access second embedding matrix
    - Requires larger embedding matrices and more parameters than ASIDE
    
    Note:
        This is a legacy approach. The main ASIDE method (ForwardRotMixin) is
        more efficient and achieves similar performance.
    """
    config_class = CustomLlamaConfig

    def __init__(self, config):
        super(CustomLLaMA, self).__init__(config)
        self.config = config

        device = next(self.parameters()).device
        model_dtype = next(self.parameters()).dtype
        # Adjust lm_head to output over the original vocabulary size
        # self.lm_head = nn.Linear(config.hidden_size, config.original_vocab_size, bias=False).to(
        #     device=device, dtype=model_dtype
        # )
        # Register the gradient hook
        self.model.embed_tokens.weight.register_hook(self._zero_padding_gradients)

    @staticmethod
    def rotate_initialized_embedding(
        model, original_vocab_size, embedding_init, rotation_alpha
    ):
        device = next(model.parameters()).device
        model_dtype = next(model.parameters()).dtype
        with torch.no_grad():
            data_embedding = CustomLLaMA.initialize_data_embedding(
                model.model.embed_tokens.weight[original_vocab_size:],
                embedding_init,
                rotation_alpha,
            )
            model.model.embed_tokens.weight[original_vocab_size:] = (
                data_embedding.detach().to(device=device, dtype=model_dtype)
            )  # \
            del data_embedding
        model.model.embed_tokens.weight.requires_grad_(True)
        return model

    @staticmethod
    def setup_model_and_tok(
        saved_model_path,
        pretrained_model_path,
        data_model_path,
        tokenizer_path,
        embedding_init="copy",
        rotation_alpha=None,
        device="cuda" if torch.cuda.is_available() else "cpu",
        load_from_checkpoint=False,
        model_dtype=torch.bfloat16,
        post_init_rotation=False,
    ):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        if tokenizer.pad_token is None:
            if (
                "llama-3" in pretrained_model_path.lower()
                or "llama_3" in pretrained_model_path.lower()
            ):
                tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(
                    "<|reserved_special_token_3|>"
                )
            elif (
                "llama-2" in pretrained_model_path.lower()
                or "llama_2" in pretrained_model_path.lower()
            ):
                tokenizer.pad_token_id = tokenizer.unk_token_id
                tokenizer.pad_token = tokenizer.unk_token
            else:
                raise NotImplementedError(
                    "Pad token supports only llama 3 and 2 now, fix"
                )
        else:
            tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(
                tokenizer.pad_token
            )

        if tokenizer.eos_token_id is None:
            if (
                "llama-3" in pretrained_model_path.lower()
                or "llama_3" in pretrained_model_path.lower()
            ):
                tokenizer.eos_token = "<|end_of_text|>"
                tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids(
                    "<|end_of_text|>"
                )
            elif (
                "llama-2" in pretrained_model_path.lower()
                or "llama_2" in pretrained_model_path.lower()
            ):
                tokenizer.eos_token = "</s>"
                tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids("</s>")
            else:
                raise NotImplementedError(
                    "Pad token supports only llama 3 and 2 now, fix"
                )
        # tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids(tokenizer.eos_token)
        tokenizer.padding_side = "left"

        print(
            f"Finished tokenizer init. len(tokenizer) = {len(tokenizer)}, tokenizer.vocab_size = {tokenizer.vocab_size}"
        )

        if load_from_checkpoint:
            model = CustomLLaMA.from_pretrained(
                saved_model_path,
                model_dtype,
                tokenizer_vocab_size=len(tokenizer),
                tokenizer_pad_token_id=tokenizer.pad_token_id,
                # attn_implementation='eager',
                trust_remote_code=True,
            ).to(device)
            if post_init_rotation:
                model = CustomLLaMA.rotate_initialized_embedding(
                    model, len(tokenizer), embedding_init, rotation_alpha
                )
            torch.cuda.empty_cache()
        else:
            assert not post_init_rotation, "Double rotation is not supported"
            model = CustomLLaMA.from_pretrained_models(
                pretrained_model_path,
                data_model_path,
                tokenizer,
                embedding_init=embedding_init,
                rotation_alpha=rotation_alpha,
                model_dtype=model_dtype,
            ).to(device)
        model.config.eos_token_id = tokenizer.eos_token_id

        # Update model config
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.eos_token_id = tokenizer.eos_token_id

        # tokenizer.save_pretrained(saved_model_path)  # Save tokenizer with model
        return model, tokenizer

    def _zero_padding_gradients(self, grad):
        pad_token_id = self.config.pad_token_id
        original_vocab_size = self.config.original_vocab_size
        padding_indices = [pad_token_id, pad_token_id + original_vocab_size]
        # Zero out gradients for padding indices
        for idx in padding_indices:
            if idx < grad.size(0):
                grad[idx] = 0
        return grad

    @staticmethod
    def initialize_data_embedding(
        orig_embedding, embedding_init_type="copy", rotation_alpha="None"
    ):
        print(
            f"Called initialize_data_embedding. {embedding_init_type}, {rotation_alpha}"
        )
        if embedding_init_type == "copy":
            return orig_embedding
        elif embedding_init_type == "rot_ind":
            return rotate_embeddings_independently(orig_embedding, rotation_alpha)
        elif embedding_init_type == "rot_isoclinic":
            return rotate_embeddings_in_multiple_planes(orig_embedding, rotation_alpha)
        else:
            raise ValueError(f"embedding_init_type={embedding_init_type} is invalid")

    @classmethod
    def from_pretrained_models(
        cls,
        pretrained_model_name_or_path,
        data_model_name_or_path,
        tokenizer,
        embedding_init="copy",
        rotation_alpha=None,
        model_dtype=torch.float16,
    ):
        # Load the configuration and update vocab_size
        config = CustomLlamaConfig.from_pretrained(pretrained_model_name_or_path)
        original_vocab_size = len(tokenizer)
        declared_vocab_size = tokenizer.vocab_size
        specials_vocab_size = original_vocab_size - declared_vocab_size

        config.vocab_size = declared_vocab_size * 2 + specials_vocab_size
        config.original_vocab_size = original_vocab_size  # Custom attribute
        config.declared_vocab_size = declared_vocab_size
        config.specials_vocab_size = specials_vocab_size
        # Instantiate the custom model

        print(
            f"original_vocab_size={original_vocab_size}, declared_vocab_size={declared_vocab_size}, specials_vocab_size={specials_vocab_size}, config.vocab_size={config.vocab_size}"
        )

        config.attn_implementation = "eager"
        config.use_cache = False
        model = cls(config)

        # Load the two models
        pretrained_model = LlamaForCausalLM.from_pretrained(
            pretrained_model_name_or_path,
            attn_implementation="eager",
            torch_dtype=model_dtype,
            use_cache=False,
        )
        data_model = LlamaForCausalLM.from_pretrained(
            data_model_name_or_path,
            attn_implementation="eager",
            torch_dtype=model_dtype,
            use_cache=False,
        )
        # pretrained_model.resize_token_embeddings(original_vocab_size, mean_resizing=False)
        # data_model.resize_token_embeddings(original_vocab_size, mean_resizing=False)

        # Set device and dtype
        device = next(pretrained_model.parameters()).device
        # model_dtype = next(pretrained_model.parameters()).dtype
        model.to(device=device, dtype=model_dtype)
        # print("EMB SHAPE ORIGINAL", pretrained_model.model.embed_tokens.weight.data.shape)
        # print("EMB SHAPE DOUBLE", model.model.embed_tokens.weight.data.shape)
        if specials_vocab_size > 0:
            data_embedding = CustomLLaMA.initialize_data_embedding(
                data_model.model.embed_tokens.weight.data[
                    :-specials_vocab_size
                ].clone(),
                embedding_init,
                rotation_alpha,
            )
        else:
            data_embedding = CustomLLaMA.initialize_data_embedding(
                data_model.model.embed_tokens.weight.data.clone(),
                embedding_init,
                rotation_alpha,
            )

        print(
            f"Inst emb shape: {pretrained_model.model.embed_tokens.weight.data.shape}"
        )
        print(f"Data emb shape: {data_model.model.embed_tokens.weight.data.shape}")
        model.model.embed_tokens.weight.data[:original_vocab_size] = (
            pretrained_model.model.embed_tokens.weight.data.clone().to(
                device=device, dtype=model_dtype
            )
        )
        print(
            f"model.model.embed_tokens.weight.data.shape={model.model.embed_tokens.weight.data.shape}, data_embedding.shape={data_embedding.shape}"
        )
        model.model.embed_tokens.weight.data[original_vocab_size:] = data_embedding.to(
            device=device, dtype=model_dtype
        )  # \
        # data_model.model.embed_tokens.weight.data[:-specials_vocab_size].clone().to(device=device, dtype=model_dtype)

        model.lm_head.weight.data[:original_vocab_size] = (
            pretrained_model.lm_head.weight.data.clone().to(
                device=device, dtype=model_dtype
            )
        )
        if specials_vocab_size > 0:
            model.lm_head.weight.data[original_vocab_size:] = (
                data_model.lm_head.weight.data[:-specials_vocab_size]
                .clone()
                .to(device=device, dtype=model_dtype)
            )
        else:
            model.lm_head.weight.data[original_vocab_size:] = (
                data_model.lm_head.weight.data.clone().to(
                    device=device, dtype=model_dtype
                )
            )

        # model.lm_head.weight.data.copy_(pretrained_model.lm_head.weight.data.clone().to(device=device, dtype=model_dtype))

        # Load other weights
        pretrained_state_dict = pretrained_model.state_dict()
        pretrained_state_dict.pop("model.embed_tokens.weight", None)
        pretrained_state_dict.pop("lm_head.weight", None)
        missing_keys, unexpected_keys = model.load_state_dict(
            pretrained_state_dict, strict=False
        )
        if missing_keys:
            print(f"Missing keys when loading state_dict: {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys when loading state_dict: {unexpected_keys}")

        if tokenizer.pad_token_id is not None:
            shifted_pad_token_id = tokenizer.pad_token_id + original_vocab_size
            padding_indices = [tokenizer.pad_token_id, shifted_pad_token_id]
            for idx in padding_indices:
                if idx < model.model.embed_tokens.weight.size(0):
                    model.model.embed_tokens.weight.data[idx].zero_()
        else:
            print("pad_token_id is None. Please ensure the tokenizer has a pad_token.")

        model.to(device=device, dtype=model_dtype)
        return model

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path, model_dtype, *model_args, **kwargs
    ):
        # Load configuration with potential additional arguments
        config = CustomLlamaConfig.from_pretrained(
            pretrained_model_name_or_path, **kwargs
        )  # Instantiate the model
        tokenizer_vocab_size = kwargs.pop("tokenizer_vocab_size", None)
        tokenizer_pad_token_id = kwargs.pop("tokenizer_pad_token_id", None)
        kwargs.pop("config", None)  # otherwise, loaded twice

        model = super(CustomLLaMA, cls).from_pretrained(
            pretrained_model_name_or_path,
            torch_dtype=model_dtype,
            config=config,
            **kwargs,
        )

        if tokenizer_vocab_size != config.original_vocab_size:
            model.resize_token_embeddings(tokenizer_vocab_size * 2, mean_resizing=False)
            config.vocab_size = tokenizer_vocab_size * 2
            config.original_vocab_size = tokenizer_vocab_size

        if tokenizer_pad_token_id is not None:
            shifted_pad_token_id = tokenizer_pad_token_id + tokenizer_vocab_size
            embedding_layer = model.model.embed_tokens
            padding_indices = [tokenizer_pad_token_id, shifted_pad_token_id]
            for idx in padding_indices:
                if idx < embedding_layer.weight.size(0):
                    embedding_layer.weight.data[idx].zero_()
        else:
            print("pad_token_id is None. Please ensure the tokenizer has a pad_token.")

        return model


    def save_pretrained(self, save_directory, state_dict=None, **kwargs):
        """
        Save the model to the specified directory. Handles optional `state_dict` argument.
        """
        os.makedirs(save_directory, exist_ok=True)

        # Save the configuration
        self.config.save_pretrained(save_directory)

        # Save the state dictionary
        if state_dict is None:
            state_dict = (
                self.state_dict()
            )  # Use the model's current state dict if not provided
        torch.save(state_dict, os.path.join(save_directory, "pytorch_model.bin"))

        