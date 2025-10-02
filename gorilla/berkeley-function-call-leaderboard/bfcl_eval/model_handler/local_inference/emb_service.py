import threading, queue, time
import torch

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

class RotatingEmbeddingService:
    """
    Owns ONE HF model on ONE GPU. Call .get(input_ids, segment_ids) from any thread.
    Internally micro-batches jobs to a single CUDA forward for throughput.
    Returns CPU float32 [T,D] prompt embeddings (rotated where segment_ids==1).
    """
    def __init__(self, embed_layer, rotation_matrix: torch.Tensor,
                 pad_id: int = 0, max_batch: int = 32, wait_ms: int = 3):
        self.embed_layer = embed_layer
        self.rotation = rotation_matrix                             # [D,D] on CUDA
        self.pad_id = pad_id
        self.max_batch = max_batch
        self.wait_ms = wait_ms

        self._q: "queue.Queue[dict]" = queue.Queue(maxsize=2048)
        self._stop = threading.Event()

        self._thr = threading.Thread(target=self._worker, daemon=True)
        self._thr.start()

    def _collect(self):
        # Block for at least one job, then try to scoop up a few more (micro-batch)
        first = self._q.get()
        if first is None:
            return None
        batch = [first]
        t0 = time.time()
        while len(batch) < self.max_batch and (time.time() - t0) < (self.wait_ms / 1000.0):
            try:
                batch.append(self._q.get_nowait())
            except queue.Empty:
                time.sleep(self.wait_ms / 4000.0)  # tiny nap to avoid busy-wait
                break
        return batch

    @torch.inference_mode()
    def _worker(self):
        while not self._stop.is_set():
            batch = self._collect()
            if batch is None:
                break

            ids_list = [b["ids"] for b in batch]    # list[Tensor[T]]
            seg_list = [b["seg"] for b in batch]    # list[Tensor[T]]
            replies  = [b["rq"]  for b in batch]
            lens     = [t.size(0) for t in ids_list]

            # Right-pad both
            ids_pad = torch.nn.utils.rnn.pad_sequence(
                ids_list, batch_first=True, padding_value=self.pad_id
            ).cuda(non_blocking=True)

            seg_pad = torch.nn.utils.rnn.pad_sequence(
                seg_list, batch_first=True, padding_value=0
            ).cuda(non_blocking=True).bool()

            # Compute embeddings and rotate where seg==1
            emb = self.embed_layer(ids_pad.to(self.embed_layer.weight.device))  # [B,T,D] on CUDA          
            mask = seg_pad == 1
            
            out = emb.clone()
            out[mask] = torch.matmul(out[mask], self.rotation)  # right-rotation example

            # Return CPU float32 per request (trim padding)
            out = out.to(torch.float32).cpu()
            for i, (L, rq) in enumerate(zip(lens, replies)):
                rq.put(out[i, :L])  # [T_i, D]
        
    def get(self, input_ids: torch.Tensor, segment_ids: torch.Tensor) -> torch.Tensor:
        """input_ids, segment_ids: CPU Long tensors shape [T]. Returns CPU float32 [T,D]."""
        rq: "queue.Queue[torch.Tensor]" = queue.Queue(maxsize=1)
        self._q.put({"ids": input_ids, "seg": segment_ids, "rq": rq})
        res = rq.get()
        if isinstance(res, Exception):
            raise res
        return res

    def close(self):
        self._stop.set()
        self._q.put(None)
        self._thr.join()