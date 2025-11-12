from typing import List
from needle.autograd import Tensor
import needle.backend_ndarray.ndarray as ndarray
from needle import ops
import needle.init as init
import numpy as np
from .nn_sequence import Embedding
from .nn_basic import (
    Parameter, 
    Module, 
    ReLU,
    Dropout,
    LayerNorm1d,
    Linear,
    Sequential
)


class MultiHeadAttention(Module):
    """
    The multi-head self attention module.
    """
    def __init__(
        self,
        *,
        dropout = 0.,
        causal = False,
        device = None,
        dtype = "float32",
    ):

        super().__init__()

        self.device = device
        self.dtype = dtype

        self.causal = causal
        self.dropout = Dropout(dropout)

    def create_causal_mask(self, i, j, device):
        """
        return a triangular causal mask.
        Input: i, j: the shape of the mask to be created
        """
        mask = -np.finfo(np.float32).max * np.triu(
            np.ones((1, 1, i, j), dtype=np.float32), j - i + 1)

        return ndarray.array(
            mask, device=device)

    def matmul(self, a, b_transpose):
        """
        batched matrix multiplication;
        """
        a_shape = (*a.shape[:-1], 1, *a.shape[-1:])
        a = a.reshape(a_shape)

        b_transpose_shape = (*b_transpose.shape[:-2], 1, *b_transpose.shape[-2:])
        b_transpose = b_transpose.reshape(b_transpose_shape)

        broadcast_shape = list(a_shape)
        broadcast_shape[-2] = b_transpose_shape[-2]
        a = a.broadcast_to(broadcast_shape)

        broadcast_shape = list(b_transpose_shape)
        broadcast_shape[-3] = a_shape[-3]
        b_transpose = b_transpose.broadcast_to(broadcast_shape)

        return (a * b_transpose).sum(len(a.shape) - 1)

    def softmax(self, logit):
        """
        The softmax function; 
        """
        max_val = Tensor(
            logit.realize_cached_data().max(axis=3),
            device=logit.device,
            dtype=logit.dtype,
            requires_grad=False
        )

        max_val = max_val.reshape((*logit.shape[:-1], 1))
        max_val = max_val.broadcast_to(logit.shape)

        probs = ops.exp(logit - max_val)

        denom = probs.sum(axes=3)
        denom = denom.reshape((*logit.shape[:-1], 1))
        denom = denom.broadcast_to(logit.shape)

        return probs / denom

    def forward(
        self,
        q, k, v,
    ):
        """
        The forward function of the MultiHeadAttention activation function.
        Input: three states q, k, v, with shape (batch_size, num_head, seq_len, dim_head)
        Output: the activation output `result` and attention softmax probability `probs` (with dropout applied)
        """
        batch_size, num_head, queries_len, q_dim = q.shape
        _, _, keys_values_len, k_dim = k.shape
        _, _, _, v_dim = v.shape

        assert q_dim == k_dim == v_dim

        result = None
        probs = None

        ### BEGIN YOUR SOLUTION
        scores = self.matmul(q, k)
        
        # Step 2: Scale by 1/sqrt(D)
        scores = scores / (q_dim ** 0.5)
        
        # Step 3: Apply causal mask if needed
        if self.causal:
            mask_device = q.device
            mask = self.create_causal_mask(queries_len, keys_values_len, mask_device)
            mask_tensor = Tensor(mask, device=mask_device, requires_grad=False)
            mask_tensor = ops.broadcast_to(mask_tensor, scores.shape)
            scores = scores + mask_tensor
        
        # Step 4: Apply softmax
        probs = self.softmax(scores)
        
        # Step 5: Apply dropout to attention probabilities
        probs = self.dropout(probs)
        
        # Step 6: Compute result = probs @ V
        # Transpose values so that the sequence axis becomes the last axis (to be summed over)
        v_transpose = ops.transpose(v, axes=(2, 3))  # (B, H, T_key, D) -> (B, H, D, T_key)
        result = self.matmul(probs, v_transpose)
        ### END YOUR SOLUTION

        return result, probs


class AttentionLayer(Module):

    def __init__(
        self,
        q_features: int,
        num_head: int,
        dim_head: int,
        *,
        k_features: int = None,
        v_features: int = None,
        out_features: int = None,
        dropout = 0.,
        causal = True,
        device = None,
        dtype = "float32",
    ):

        super().__init__()

        self.device = device
        self.dtype = dtype

        if k_features is None:
            k_features = q_features
        if v_features is None:
            v_features = q_features
        if out_features is None:
            out_features = q_features

        self.q_features = q_features
        self.k_features = k_features
        self.v_features = v_features
        self.out_features = out_features

        self.num_head = num_head
        self.dim_head = dim_head

        self.prenorm_q = LayerNorm1d(
            q_features, device=device, dtype=dtype)
        self.prenorm_k = LayerNorm1d(
            k_features, device=device, dtype=dtype)
        self.prenorm_v = LayerNorm1d(
            v_features, device=device, dtype=dtype)

        inner_dim = num_head * dim_head
        
        self.q_projection = Linear(
            q_features, inner_dim, bias=False,
            device=device, dtype=dtype)
        self.k_projection = Linear(
            k_features, inner_dim, bias=False,
            device=device, dtype=dtype)
        self.v_projection = Linear(
            v_features, inner_dim, bias=False,
            device=device, dtype=dtype)

        self.attn = MultiHeadAttention(
            dropout=dropout, causal=causal,
            device=device, dtype=dtype)

        self.out_projection = Linear(
            inner_dim, out_features, bias=False,
            device=device, dtype=dtype)

    def forward(
        self,
        q, k=None, v=None,
    ):
        """
        The forward function of the self-attention layer.
        Input: `q` with shape (batch_size, q_len, q_dim)
               `k` (if not None) with shape (batch_size, kv_len, k_dim)
               `v` (if not None) with shape (batch_size, kv_len, v_dim)
        Output: the output `result` with shape (batch_size, kv_len, out_features)
        """

        if k is None:
            k = q
        if v is None:
            v = q

        batch_size, queries_len, q_dim = q.shape
        _, keys_values_len, k_dim = k.shape
        _, _, v_dim = v.shape

        result = None

        ### BEGIN YOUR SOLUTION
        inner_dim = self.num_head * self.dim_head

        # Pre-normalize queries, keys, and values
        q_norm = self.prenorm_q(q)
        k_norm = self.prenorm_k(k)
        v_norm = self.prenorm_v(v)

        # Linear projections to multi-head representations
        q_proj = self.q_projection(q_norm)
        k_proj = self.k_projection(k_norm)
        v_proj = self.v_projection(v_norm)

        # Reshape to expose the head dimension and move it before sequence length
        q_proj = ops.reshape(q_proj, (batch_size, queries_len, self.num_head, self.dim_head))
        k_proj = ops.reshape(k_proj, (batch_size, keys_values_len, self.num_head, self.dim_head))
        v_proj = ops.reshape(v_proj, (batch_size, keys_values_len, self.num_head, self.dim_head))

        q_proj = ops.transpose(q_proj, axes=(1, 2))
        k_proj = ops.transpose(k_proj, axes=(1, 2))
        v_proj = ops.transpose(v_proj, axes=(1, 2))

        # Compute attention and store probabilities for debugging
        attn_output, probs = self.attn(q_proj, k_proj, v_proj)
        self.probs = probs

        # Merge heads back into the channel dimension
        attn_output = ops.transpose(attn_output, axes=(1, 2))
        attn_output = ops.reshape(attn_output, (batch_size, queries_len, inner_dim))

        result = self.out_projection(attn_output)
        ### END YOUR SOLUTION

        return result


class TransformerLayer(Module):

    def __init__(
        self,
        q_features: int,
        num_head: int,
        dim_head: int,
        hidden_size: int,
        *,
        dropout = 0.,
        causal = True,
        device = None,
        dtype = "float32",
    ):

        super().__init__()

        self.device = device
        self.dtype = dtype

        ### BEGIN YOUR SOLUTION
        self.attention = AttentionLayer(
            q_features,
            num_head,
            dim_head,
            dropout=dropout,
            causal=causal,
            device=device,
            dtype=dtype,
        )

        self.attn_dropout = Dropout(dropout)

        self.ffn_norm = LayerNorm1d(q_features, device=device, dtype=dtype)
        self.ffn_linear1 = Linear(
            q_features,
            hidden_size,
            device=device,
            dtype=dtype,
        )
        self.ffn_activation = ReLU()
        self.ffn_dropout1 = Dropout(dropout)
        self.ffn_linear2 = Linear(
            hidden_size,
            q_features,
            device=device,
            dtype=dtype,
        )
        self.ffn_dropout2 = Dropout(dropout)
        ### END YOUR SOLUTION

    def forward(
        self,
        x
    ):
        """
        The forward function of a Transformer Layer.
        Input: the hidden states from previous layers `x` with shape (batch_size, seq_len, x_dim)
        Ouput: the hidden states after the Transformer Layer `x` with shape (batch_size, seq_len, x_dim)
        """

        batch_size, seq_len, x_dim = x.shape

        ### BEGIN YOUR SOLUTION
        attn_out = self.attention(x)
        attn_out = self.attn_dropout(attn_out)
        x = x + attn_out

        ff_input = self.ffn_norm(x)
        ff_hidden = self.ffn_linear1(ff_input)
        ff_hidden = self.ffn_activation(ff_hidden)
        ff_hidden = self.ffn_dropout1(ff_hidden)
        ff_output = self.ffn_linear2(ff_hidden)
        ff_output = self.ffn_dropout2(ff_output)

        x = x + ff_output
        ### END YOUR SOLUTION

        return x


class Transformer(Module):

    def __init__(
        self,
        embedding_size: int,
        hidden_size: int,
        num_layers: int, 
        *,
        num_head: int = 8,
        dim_head: int = 32,
        dropout = 0.,
        causal = True,
        device = None,
        dtype = "float32",
        batch_first = False,
        sequence_len = 2048
    ):

        super().__init__()

        self.device = device
        self.dtype = dtype
        self.batch_first = batch_first

        ### BEGIN YOUR SOLUTION
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.sequence_len = sequence_len

        self.pos_embedding = Embedding(
            sequence_len,
            embedding_size,
            device=device,
            dtype=dtype,
        )

        self.input_dropout = Dropout(dropout)

        self.layers: List[TransformerLayer] = [
            TransformerLayer(
                embedding_size,
                num_head,
                dim_head,
                hidden_size,
                dropout=dropout,
                causal=causal,
                device=device,
                dtype=dtype,
            )
            for _ in range(num_layers)
        ]
        ### END YOUR SOLUTION

    def forward(
        self,
        x, h=None
    ):

        if not self.batch_first:
            x = ops.transpose(x, axes=(0, 1))

        ### BEGIN YOUR SOLUTION
        batch_size, seq_len, _ = x.shape

        if seq_len > self.sequence_len:
            raise ValueError("Sequence length exceeds configured positional embedding size")

        pos_indices = np.broadcast_to(
            np.arange(seq_len, dtype=np.int32).reshape(seq_len, 1),
            (seq_len, batch_size),
        )
        pos_indices_tensor = Tensor(
            pos_indices.astype("float32"),
            device=x.device,
            dtype="float32",
        )

        pos_encoding = self.pos_embedding(pos_indices_tensor)
        pos_encoding = ops.transpose(pos_encoding, axes=(1, 0, 2))

        x = x + pos_encoding

        for layer in self.layers:
            x = layer(x)
        ### END YOUR SOLUTION

        if not self.batch_first:
            x = ops.transpose(x, axes=(0, 1))

        return x, init.zeros_like(x)