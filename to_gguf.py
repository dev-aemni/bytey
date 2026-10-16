import struct
import numpy as np

print("Converting bytey.dat -> bytey-1.0_15M_slm_idseqversion.gguf...")

with open("bytey.dat", "rb") as f:
    V, N = struct.unpack("ii", f.read(8))
    print(f"Reading Vocabulary ({V} words) and Knowledge ({N} answers)...")
    
    words = []
    for _ in range(V):
        w_len = struct.unpack("i", f.read(4))[0]
        words.append(f.read(w_len).decode("utf-8", errors="ignore"))
        
    answers = []
    for _ in range(N):
        a_len = struct.unpack("i", f.read(4))[0]
        answers.append(f.read(a_len).decode("utf-8", errors="ignore"))
        
    D_MODEL = 2048
    W_emb = np.frombuffer(f.read(V * D_MODEL * 4), dtype=np.float32)
    W_l1 = np.frombuffer(f.read(D_MODEL * D_MODEL * 4), dtype=np.float32)
    W_l2 = np.frombuffer(f.read(D_MODEL * D_MODEL * 4), dtype=np.float32)
    W_l3 = np.frombuffer(f.read(D_MODEL * D_MODEL * 4), dtype=np.float32)
    W_ans = np.frombuffer(f.read(N * D_MODEL * 4), dtype=np.float32)

with open("bytey-1.0_15M_slm_idseqversion.gguf", "wb") as out:
    out.write(b"GGUF")
    out.write(struct.pack("<I", 3))   # Version 3
    out.write(struct.pack("<Q", 5))   # 5 Tensors
    out.write(struct.pack("<Q", 2))   # 2 Metadata KV

    # Metadata
    arch_key = "general.architecture"
    out.write(struct.pack("<Q", len(arch_key)))
    out.write(arch_key.encode("utf-8"))
    out.write(struct.pack("<I", 8))
    arch_val = "bytey-15m"
    out.write(struct.pack("<Q", len(arch_val)))
    out.write(arch_val.encode("utf-8"))

    dim_key = "bytey.embedding_length"
    out.write(struct.pack("<Q", len(dim_key)))
    out.write(dim_key.encode("utf-8"))
    out.write(struct.pack("<I", 4))
    out.write(struct.pack("<I", D_MODEL))

    tensors = [
        ("token_embd.weight", [D_MODEL, V], W_emb),
        ("layers.0.weight", [D_MODEL, D_MODEL], W_l1),
        ("layers.1.weight", [D_MODEL, D_MODEL], W_l2),
        ("layers.2.weight", [D_MODEL, D_MODEL], W_l3),
        ("output.weight", [D_MODEL, N], W_ans)
    ]
    
    offset = 0
    for name, dims, data in tensors:
        out.write(struct.pack("<Q", len(name)))
        out.write(name.encode("utf-8"))
        out.write(struct.pack("<I", len(dims)))
        for d in dims:
            out.write(struct.pack("<Q", d))
        out.write(struct.pack("<I", 0)) # FP32
        out.write(struct.pack("<Q", offset))
        offset += data.nbytes

    for _, _, data in tensors:
        out.write(data.tobytes())

print(f"✔ SUCCESS: Exported bytey-1.0_15M_slm_idseqversion.gguf ({V} vocab, {N} responses, 15.04M params)!")
