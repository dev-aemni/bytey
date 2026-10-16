import struct
import numpy as np
import gguf

output_filename = "bytey-1.0_15M_slm_idseqversion.gguf"
print(f"Building Bytey's Original Validated GGUF: {output_filename}...")

# 1. Load Vocab from tokenizer.bin
vocab = []
with open("../real_llm/tokenizer.bin", "rb") as f:
    max_token_length = struct.unpack("I", f.read(4))[0]
    for _ in range(32000):
        score = struct.unpack("f", f.read(4))[0]
        len_ = struct.unpack("I", f.read(4))[0]
        word = f.read(len_)
        vocab.append((word, score))

# 2. Load Bytey's Pretrained Weights
with open("../real_llm/stories15M.bin", "rb") as f:
    dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len = struct.unpack("7i", f.read(28))
    raw_weights = np.fromfile(f, dtype=np.float32)

print(f"Loaded Bytey weights: Dim={dim}, Hidden={hidden_dim}, Layers={n_layers}, Heads={n_heads}")

writer = gguf.GGUFWriter(output_filename, "llama")

# Standard LLaMA Metadata
writer.add_name("Bytey-15M-Original")
writer.add_context_length(1024)
writer.add_embedding_length(dim)
writer.add_block_count(n_layers)
writer.add_feed_forward_length(hidden_dim)
writer.add_head_count(n_heads)
writer.add_head_count_kv(n_kv_heads)
writer.add_layer_norm_rms_eps(1e-5)

# BPE Vocab
tokens = [item[0] for item in vocab]
scores = [item[1] for item in vocab]
toktypes = [gguf.TokenType.NORMAL] * vocab_size
toktypes[1] = gguf.TokenType.CONTROL # <s>
toktypes[2] = gguf.TokenType.CONTROL # </s>

writer.add_tokenizer_model("llama")
writer.add_token_list(tokens)
writer.add_token_scores(scores)
writer.add_token_types(toktypes)
writer.add_bos_token_id(1)
writer.add_eos_token_id(2)
writer.add_unk_token_id(0)

# Unpack according to llama2.c memory layout
ptr = 0
def read_tensor(shape):
    global ptr
    count = int(np.prod(shape))
    arr = raw_weights[ptr:ptr+count].reshape(shape)
    ptr += count
    return arr

# 1. Token Embeddings
token_embd = read_tensor((vocab_size, dim))
writer.add_tensor("token_embd.weight", token_embd)

# 2. Attention Norms for all layers
attn_norms = [read_tensor((dim,)) for _ in range(n_layers)]

# 3. Attention Q, K, V, Out
wq = [read_tensor((dim, dim)) for _ in range(n_layers)]
wk = [read_tensor((dim, dim)) for _ in range(n_layers)]
wv = [read_tensor((dim, dim)) for _ in range(n_layers)]
wo = [read_tensor((dim, dim)) for _ in range(n_layers)]

# 4. FFN Norms
ffn_norms = [read_tensor((dim,)) for _ in range(n_layers)]

# 5. FFN Gate, Down, Up (W1, W2, W3)
w1 = [read_tensor((hidden_dim, dim)) for _ in range(n_layers)] # Gate
w2 = [read_tensor((dim, hidden_dim)) for _ in range(n_layers)] # Down
w3 = [read_tensor((hidden_dim, dim)) for _ in range(n_layers)] # Up

# 6. Final RMS Norm
output_norm = read_tensor((dim,))

# Write to GGUF
for l in range(n_layers):
    writer.add_tensor(f"blk.{l}.attn_norm.weight", attn_norms[l])
    writer.add_tensor(f"blk.{l}.attn_q.weight", wq[l])
    writer.add_tensor(f"blk.{l}.attn_k.weight", wk[l])
    writer.add_tensor(f"blk.{l}.attn_v.weight", wv[l])
    writer.add_tensor(f"blk.{l}.attn_output.weight", wo[l])
    
    writer.add_tensor(f"blk.{l}.ffn_norm.weight", ffn_norms[l])
    writer.add_tensor(f"blk.{l}.ffn_gate.weight", w1[l])
    writer.add_tensor(f"blk.{l}.ffn_down.weight", w2[l])
    writer.add_tensor(f"blk.{l}.ffn_up.weight", w3[l])

writer.add_tensor("output_norm.weight", output_norm)

# Shared Weight Classifier Head (Standard LLaMA weight tying)
writer.add_tensor("output.weight", token_embd)

writer.write_header_to_file()
writer.write_kv_data_to_file()
writer.write_tensors_to_file()
writer.close()

print(f"✔ SUCCESS: Bytey original GGUF compiled correctly: {output_filename}")
