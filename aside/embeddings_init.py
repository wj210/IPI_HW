import time
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

def rotate_embeddings_independently(embeds: torch.Tensor, alpha: float) -> torch.Tensor:
    """
    Rotates each embedding vector independently by angle alpha in a random orthogonal direction.
    
    This function applies individual rotations to each embedding, where each embedding is
    rotated by the same angle but in a different random direction orthogonal to itself.
    This is a legacy approach, not the main ASIDE method.
    
    Args:
        embeds (torch.Tensor): Embedding matrix of shape [n_embeds, d] where n_embeds is
                           old   the number of embeddings and d is the embedding dimension.
        alpha (float): Rotation angle in radians.
        
    Returns:
        torch.Tensor: Rotated embeddings of shape [n_embeds, d]. Each row is the corresponding
                     input embedding rotated by alpha in a random orthogonal direction.
                     
    Note:
        - Zero or near-zero embeddings are left unchanged
        - This method is computationally expensive as it processes each embedding individually
        - The main ASIDE method uses isoclinic rotations instead for efficiency
        
    Implementation Details:
        For each embedding x:
        1. Find unit direction: hat_x = x / ||x||
        2. Generate random orthogonal direction: u_perp ⊥ hat_x  
        3. Rotate in plane spanned by hat_x and u_perp: y = ||x|| * (cos(α) * hat_x + sin(α) * u_perp)
    """
    print("Called rotate_embeddings_independently")
    n_embeds, d = embeds.shape
    rotated = torch.empty_like(embeds)

    print("Processing embeddings")
    for i in tqdm(range(n_embeds)):
        x = embeds[i]
        norm_x = x.norm(p=2)

        if norm_x < 1e-12:
            # If x is very small or zero, we skip rotating (or handle differently)
            rotated[i] = x
            continue
        
        # 1) Unit direction of x
        hat_x = x / norm_x

        # 2) Pick random Gaussian vector
        u = torch.randn(d, dtype=x.dtype, device=x.device)

        # 3) Make u orthogonal to x
        proj = (u @ hat_x) * hat_x
        u_perp = u - proj
        
        perp_norm = u_perp.norm(p=2)
        if perp_norm < 1e-12:
            # The random vector happened to be nearly collinear with x
            # You could try again or just skip
            rotated[i] = x
            continue

        # 4) Normalize the orthogonal direction
        n_ = u_perp / perp_norm

        # 5) Rotate in the plane spanned by hat_x and n_
        cos_a = torch.cos(torch.tensor(alpha, dtype=x.dtype, device=x.device))
        sin_a = torch.sin(torch.tensor(alpha, dtype=x.dtype, device=x.device))

        y = norm_x * (cos_a * hat_x + sin_a * n_)
        rotated[i] = y

    return rotated


def generate_isoclinic_rotation_matrix(dim, alpha, device=None, dtype=None):
    """
    Generates an isoclinic rotation matrix for the ASIDE method.
    
    An isoclinic rotation applies the same rotation angle to all pairs of dimensions.
    The embedding space is split into pairs (d_0,d_1), (d_2,d_3), ..., and each pair
    is rotated by angle alpha. This is the core operation in ASIDE for creating
    distinct embedding subspaces for instructions vs data.
    
    Args:
        dim (int): Embedding dimension. Should be even for complete pairing.
        alpha (float): Rotation angle in radians. ASIDE typically uses π/2 (90°).
        device (torch.device, optional): Device for the rotation matrix.
        dtype (torch.dtype, optional): Data type for the rotation matrix.
        
    Returns:
        torch.Tensor: Isoclinic rotation matrix of shape [dim, dim]. This is an
                     orthogonal matrix where each 2x2 block along the diagonal
                     performs a rotation by angle alpha.
                     
    Note:
        - If dim is odd, the last dimension remains unchanged
        - The rotation matrix has the block-diagonal structure:
          [[cos(α) -sin(α)  0       0      ...]
           [sin(α)  cos(α)  0       0      ...]
           [0       0       cos(α) -sin(α) ...]
           [0       0       sin(α)  cos(α) ...]
           [...     ...     ...     ...    ...]]
           
    Example:
        >>> R = generate_isoclinic_rotation_matrix(4, np.pi/2)  # 90° rotation
        >>> print(R)
        # tensor([[ 0., -1.,  0.,  0.],
        #         [ 1.,  0.,  0.,  0.],
        #         [ 0.,  0.,  0., -1.],
        #         [ 0.,  0.,  1.,  0.]])
    """
    alpha_t = torch.tensor(alpha, device=device, dtype=dtype)
    cos_alpha = torch.cos(alpha_t)
    sin_alpha = torch.sin(alpha_t)

    alpha_t_clone = alpha_t.clone().to(device)  # or .to(device) if needed to ensure it's not meta
    print(f"alpha_t: {alpha_t_clone.cpu().item()} (dtype: {alpha_t_clone.dtype})")


    M = torch.eye(dim, device=device, dtype=dtype)
    for i in range(0, dim, 2):
        M[i, i]     = cos_alpha
        M[i, i+1]   = -sin_alpha
        M[i+1, i]   = sin_alpha
        M[i+1, i+1] = cos_alpha

    return M

def rotate_embeddings_in_multiple_planes(embeds: torch.Tensor, alpha: float, silent=False) -> torch.Tensor:
    """
    Applies isoclinic rotation to all embeddings simultaneously (ASIDE core operation).
    
    This is the main rotation function used in ASIDE. It applies the same isoclinic rotation
    to all embedding vectors, creating distinct subspaces for instruction and data tokens.
    The rotation is applied efficiently by operating on dimension pairs directly.
    
    Args:
        embeds (torch.Tensor): Embedding matrix of shape [n_embeds, d] where n_embeds is
                              the vocabulary size and d is the embedding dimension.
        alpha (float): Rotation angle in radians. ASIDE uses π/2 (90°) for optimal separation.
        silent (bool, optional): If True, suppresses debug output. Defaults to False.
        
    Returns:
        torch.Tensor: Rotated embeddings of shape [n_embeds, d]. All embeddings are rotated
                     by the same isoclinic transformation.
                     
    Implementation Details:
        - Splits embedding dimensions into even/odd pairs: (0,1), (2,3), (4,5), ...
        - Applies 2D rotation to each pair: [x_even, x_odd] → [x_even*cos(α) - x_odd*sin(α), 
                                                               x_even*sin(α) + x_odd*cos(α)]
        - More efficient than matrix multiplication for large embedding matrices
        
    Note:
        This function implements the core ASIDE transformation that creates geometrically
        separated embeddings for instructions and data without adding parameters.
        
    Example:
        >>> vocab_embeds = torch.randn(50000, 768)  # Typical LLM embedding matrix
        >>> data_embeds = rotate_embeddings_in_multiple_planes(vocab_embeds, np.pi/2)
        >>> # data_embeds now represents the rotated embeddings for data tokens
    """
    # Get shape
    n_embeds, d = embeds.shape
    device = embeds.device
    dtype = embeds.dtype

    # Compute cosine and sine once
    alpha_tensor = torch.tensor(alpha, device=device, dtype=dtype)
    cos_a = torch.cos(alpha_tensor)
    sin_a = torch.sin(alpha_tensor)
    
    # Instead of constructing the full rotation matrix,
    # directly compute the rotated even and odd pairs.
    even = embeds[:, 0::2]
    odd = embeds[:, 1::2]
    
    # Apply 2D rotation to each pair
    rotated_even = even * cos_a - odd * sin_a
    rotated_odd  = even * sin_a + odd * cos_a
    
    # Create a new tensor for rotated embeddings
    rotated_embeds = embeds.clone()
    rotated_embeds[:, 0::2] = rotated_even
    rotated_embeds[:, 1::2] = rotated_odd

    # if not silent:
    #     print(f"Rotated embed shape: {rotated_embeds.shape}")
    return rotated_embeds



