# Bytey (15M Small Language Model)

A native C++ 15-Million parameter Small Language Model (SLM) trained and executed directly on an Android smartphone via Termux.

## Architecture
- **Parameters:** ~15.5 Million floats (FP32)
- **Vector Width:** 2048 Dimensions (Deep MLP Memory)
- **Hardware Acceleration:** Multi-core parallel processing (OpenMP capped for mobile thermal control)
- **Engine:** Pure C++17 with GNU Readline shell navigation
- **Model Format:** Native `.dat` and `.gguf` (`bytey-1.0_15M_slm_idseqversion.gguf`)

## Features
- **Zero-Cloud:** 100% offline local training and inference.
- **Knowledge Base:** Science, Cellular Biology (ATP, Glycolysis, Endocrine systems), C++, Linux, and emotional support.
- **Independent Persona:** Dynamic conversational reasoning with low-confidence philosophical responses.

## Compilation & Run
```bash
# Compile
clang++ -O3 -ffast-math -march=armv8-a -fopenmp -std=c++17 bytey.cpp -lreadline -o bytey

# Train
./bytey train -steps 35k

# Chat
./bytey chat

