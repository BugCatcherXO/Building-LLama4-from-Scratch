# lesson_5_llama4_attention_code.py

# %% [markdown]
# # Understanding the Llama 4 Attention Mechanism
#
# This tutorial explores the attention mechanism used in the Llama 4 architecture. Attention allows the model to weigh the importance of different tokens in the input sequence when processing a specific token. Llama 4 employs several modern techniques within its attention block, including Multi-Head Attention (MHA), Grouped-Query Attention (GQA), and Rotary Positional Embeddings (RoPE).
#
# We will break down the `Llama4TextAttention` module step by step.

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional

# %% [markdown]
# ## Step 1: Setup and Configuration
#
# First, let's define some configuration parameters similar to what a Llama 4 model might use and create some sample input data.

# %%
# Configuration (Simplified for clarity)
hidden_size = 128  # Dimensionality of the model's hidden states
num_attention_heads = 16 # Total number of query heads
num_key_value_heads = 4  # Number of key/value heads (for GQA)
head_dim = hidden_size // num_attention_heads # Dimension of each attention head
max_position_embeddings = 256 # Maximum sequence length the model expects
rope_theta = 10000.0 # Base for RoPE frequency calculation
rms_norm_eps = 1e-5 # Epsilon for RMSNorm
attention_bias = False # Whether to use bias in Q 
attention_dropout = 0.0 # Dropout probability for attention weights
use_qk_norm = True # Whether to apply L2 norm to Q and K before attention

# Sample Input
batch_size = 2
sequence_length = 10
hidden_states = torch.randn(batch_size, sequence_length, hidden_size)
# Create position IDs for each token in the sequence, repeated for each batch
# torch.arange(0, sequence_length) generates a 1D tensor with values from 0 to sequence_length-1
# The unsqueeze(0) adds an extra dimension at the 0th position, making it (1, sequence_length)
# This allows repeat(batch_size, 1) to create a tensor of shape (batch_size, sequence_length)
position_ids = torch.arange(0, sequence_length).unsqueeze(0).repeat(batch_size, 1) # Shape: (batch_size, sequence_length)
# Simple causal mask (upper triangular) for demonstration
# In reality, Llama4 uses a more complex mask creation including padding handling
attention_mask = torch.triu(torch.ones(sequence_length, sequence_length) * -torch.inf, diagonal=1)
attention_mask = attention_mask.unsqueeze(0).unsqueeze(0) # Shape: (1, 1, sequence_length, sequence_length)
attention_mask = attention_mask.expand(batch_size, 1, -1, -1) # Shape: (batch_size, 1, sequence_length, sequence_length)


print("Configuration:")
print(f"  hidden_size: {hidden_size}")
print(f"  num_attention_heads: {num_attention_heads}")
print(f"  num_key_value_heads: {num_key_value_heads}")
print(f"  head_dim: {head_dim}")

print("\nSample Input Shapes:")
print(f"  hidden_states: {hidden_states.shape}")
print(f"  position_ids: {position_ids.shape}")
print(f"  attention_mask: {attention_mask.shape}")
