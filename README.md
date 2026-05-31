# Custom-Triton-Kernels

An **educational repository** for learning GPU kernel development using **Triton**. This project provides custom implementations of key LLM neural network operators, demonstrating core concepts in kernel optimization, memory management, and GPU programming with Triton JIT compilation.

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Notebooks Documentation](#notebooks-documentation)
- [Usage](#usage)
- [Kernels Implemented](#kernels-implemented)
- [Performance Considerations](#performance-considerations)
- [License](#license)

## Overview

This project is an **educational exploration** of custom GPU kernel development using **Triton**, specifically focusing on implementing key LLM components for the Llama-3 model family. The primary goal is to deepen understanding of:

- GPU kernel architecture and optimization principles
- Triton JIT compilation and programming model
- Memory access patterns and SRAM management
- Parallelization strategies for tensor operations
- Integration between custom kernels and LLM inference frameworks

The implementations are tested against the **Llama-3-8B-Instruct-AWQ** quantized model using vLLM. While the custom kernels serve as a learning tool for kernel development concepts, **native vLLM implementations provide superior production-level performance** through additional optimizations and careful tuning.

## Project Structure

```
Custom-Triton-Kernels/
├── vLLM_&_Triton.ipynb              # Main entry point - Basic vLLM inference setup
├── vLLM_&_Triton_RMSNorm.ipynb     # RMSNorm layer custom kernel implementation
├── vLLM_&_Triton_SwiGLU.ipynb      # SwiGLU activation custom kernel implementation
├── vLLM_&_Triton_Sequential.ipynb  # Sequential operation optimization
├── 50-prompts.txt                   # Test dataset with 50 diverse prompts
├── LICENSE                          # MIT License
└── README.md                        # This file
```

## Features

✨ **Educational Kernel Implementations**
- Custom RMSNorm (Root Mean Square Layer Normalization) kernel - demonstrates layer normalization patterns
- Custom SwiGLU activation function kernel - explores gating mechanisms and element-wise operations
- Sequential operation optimizations - studies batching and memory access strategies
- Support for residual connections - shows integration with modern architectural patterns
- Configurable block sizes - enables experimentation with performance vs. resource tradeoffs

🚀 **vLLM Integration for Context**
- Integration with vLLM v0.18.0 for comparison
- Support for AWQ (Activation-aware Weight Quantization)
- Batch processing capabilities for kernel validation
- Chat template support for realistic inference scenarios

🔧 **Model Support**
- Llama-3-8B-Instruct-AWQ from HuggingFace
- Quantized inference for memory efficiency
- Tensor parallel processing concepts

📊 **Testing & Learning Infrastructure**
- 50 diverse test prompts for kernel validation
- Batch inference testing against native implementations
- Detailed kernel implementation documentation

## Prerequisites

- **Python 3.8+**
- **CUDA 11.8+** (for GPU compatibility)
- **PyTorch 2.0+**
- **NVIDIA GPU** with compute capability 8.0+ (A100, H100, etc.)
- Jupyter Notebook or JupyterLab

## Installation

### Step 1: Set up Python Environment

```bash
# Create a virtual environment
python -m venv triton_env
source triton_env/bin/activate  # On Windows: triton_env\Scripts\activate

# Upgrade pip
pip install --upgrade pip
```

### Step 2: Install Core Dependencies

```bash
# Install Transformers and Datasets
pip install -U transformers datasets

# Install vLLM (optimized inference engine)
uv pip install vllm==0.18.0 --torch-backend=auto

# Install Triton
pip install triton

# Install GPTQModel (for quantized model support)
pip install gptqmodel

# Install Jupyter
pip install jupyter jupyterlab
```

### Step 3: Verify Installation

```bash
python -c "import vllm, triton, torch; print('All dependencies installed successfully!')"
```

### Step 4: Run Jupyter Notebook

```bash
jupyter lab
```

Navigate to the `Custom-Triton-Kernels` directory and open the notebooks.

## Project Purpose & Learning Objectives

This repository serves as an educational resource for understanding:

1. **Triton Programming Model**: How to write JIT-compiled GPU kernels without manual CUDA
2. **Kernel Optimization Principles**: Memory hierarchies, thread scheduling, resource allocation
3. **LLM Operator Implementation**: Deep dive into RMSNorm, SwiGLU, and other transformer components
4. **Performance Analysis**: Profiling, bottleneck identification, and optimization strategies
5. **Integration Patterns**: How custom kernels interface with high-level frameworks like vLLM

By studying these implementations, you'll gain insights into the performance considerations that production systems address.

## Notebooks Documentation

### 1. **vLLM_&_Triton.ipynb** - Basic Inference Setup

**Purpose**: Introduction to vLLM-based inference with the Llama-3 model.

**Key Components**:
- Environment setup and dependency installation
- Model loading: Llama-3-8B-Instruct-AWQ with AWQ quantization
- Tokenizer initialization and chat template handling
- Batch prompt formatting using HuggingFace chat templates
- Inference execution with configurable sampling parameters

**Key Operations**:
```python
# Initialize vLLM with quantization
llm = LLM(model="casperhansen/llama-3-8b-instruct-awq", 
          quantization="awq", tensor_parallel_size=1)

# Set sampling parameters
params = SamplingParams(temperature=0.8, max_tokens=100)

# Generate outputs
outputs = llm.generate(formatted_prompts, sampling_params=params)
```

**Output**: Generated text completions for each input prompt.

---

### 2. **vLLM_&_Triton_RMSNorm.ipynb** - RMSNorm Kernel

**Purpose**: Custom Triton implementation of Root Mean Square Layer Normalization (RMSNorm).

**Why RMSNorm?**
- More efficient than LayerNorm (no variance computation)
- Replaces mean-centering with RMS-based normalization
- Commonly used in modern LLMs (LLaMA, Falcon, etc.)
- Reduces computational overhead compared to traditional LayerNorm

**Kernel Features**:
- 2D block-based processing (tokens × hidden dimension)
- Support for residual connections
- Configurable epsilon for numerical stability
- Efficient memory access patterns with stride awareness
- Handles variable block sizes for performance tuning

**Kernel Parameters**:
- `x_ptr`: Input token tensor
- `residual_ptr`: Residual connection tensor (optional)
- `gamma_ptr`: Learned scale parameter
- `num_tokens`: Number of tokens in batch
- `hidden_dim`: Hidden dimension size
- `has_residual`: Boolean flag for residual addition
- `block_size_m`: Block size for token dimension
- `block_size_k`: Block size for hidden dimension

**Optimization Highlights**:
- Minimal memory accesses through coalesced loads
- Efficient reduction operations for RMS computation
- Support for masked operations for variable-length sequences

---

### 3. **vLLM_&_Triton_SwiGLU.ipynb** - SwiGLU Activation Kernel

**Purpose**: Custom Triton implementation of SwiGLU (Swish-Gated Linear Unit) activation function.

**What is SwiGLU?**
- Activation function: $\text{SwiGLU}(x, W, V, W_g, b, b_g) = (Wx + b) \otimes \sigma(W_g x + b_g)$
- Where $\otimes$ is element-wise multiplication and $\sigma$ is Swish activation
- Used in feedforward networks of modern LLMs (PaLM, LLaMA)
- More expressive than ReLU with gating mechanism

**Kernel Features**:
- 2D block-based processing (tokens × hidden dimension)
- Efficient Swish activation computation: $\text{Swish}(x) = x \cdot \sigma(x)$
- Support for gate and up projections
- Element-wise multiplication of gated outputs
- Optimized memory access for high throughput

**Kernel Parameters**:
- `x_ptr`: Input matrix with concatenated gate and up projections
- `out_ptr`: Output matrix
- `token_stride`: Stride between consecutive tokens
- `hidden_dim_stride`: Stride between hidden dimensions
- `num_tokens`: Number of tokens
- `hidden_dim`: Number of hidden dimensions
- `block_size_m`: Block size for tokens
- `block_size_k`: Block size for hidden dimensions

**Mathematical Formulation**:
$$\text{Output} = \text{gate} \otimes \sigma(\text{gate}) \otimes \text{up}$$

Where:
- gate = first half of input
- up = second half of input
- $\sigma$ = Swish activation function

---

### 4. **vLLM_&_Triton_Sequential.ipynb** - Sequential Operations

**Purpose**: Optimization for sequential processing patterns in LLM inference.

**Use Cases**:
- Efficient processing of token sequences
- Autoregressive generation optimization
- KV-cache aware operations

---

## Usage

### Running the Full Pipeline

1. **Open Jupyter Lab**:
   ```bash
   jupyter lab
   ```

2. **Execute Notebooks in Order**:
   - Start with `vLLM_&_Triton.ipynb` for basic setup
   - Proceed to specialized kernel notebooks (RMSNorm, SwiGLU, Sequential)

3. **Sample Inference Code**:
   ```python
   from vllm import LLM, SamplingParams
   
   # Load model
   llm = LLM(model="casperhansen/llama-3-8b-instruct-awq", 
             quantization="awq")
   
   # Prepare prompts
   prompts = ["What is machine learning?", "Explain quantum computing"]
   
   # Set parameters
   params = SamplingParams(temperature=0.7, max_tokens=512)
   
   # Generate
   outputs = llm.generate(prompts, sampling_params=params)
   for output in outputs:
       print(output.outputs[0].text)
   ```

### Using the 50 Test Prompts

The `50-prompts.txt` file contains 50 diverse prompts covering:
- General knowledge questions
- Science and mathematics
- Creative writing tasks
- Problem-solving scenarios
- Technical explanations

**Load and use prompts**:
```python
with open('50-prompts.txt', 'r') as f:
    prompts = [line.strip() for line in f if line.strip()]

# Use for batch inference
outputs = llm.generate(prompts, sampling_params=params)
```

## Kernels Implemented

### RMSNorm Kernel

| Aspect | Details |
|--------|---------|
| **Algorithm** | Root Mean Square Normalization with optional residual |
| **Complexity** | O(N·D) where N = tokens, D = hidden dim |
| **Memory Pattern** | Coalesced reads, sequential writes |
| **Parallelization** | 1D grid, 2D blocks (tokens × hidden dim) |
| **Key Optimizations** | Efficient reduction, mask support, residual fusion |

### SwiGLU Kernel

| Aspect | Details |
|--------|---------|
| **Algorithm** | Swish Gated Linear Unit: gate ⊗ σ(gate) ⊗ up |
| **Complexity** | O(N·D) where N = tokens, D = hidden dim |
| **Memory Pattern** | Sequential access to concatenated tensors |
| **Parallelization** | 1D grid, 2D blocks (tokens × hidden dim) |
| **Key Optimizations** | Fused gate computation, efficient SRAM usage |

## Educational Notes on Kernel Development

### Block Size Tuning - A Learning Exercise

The kernels use configurable block sizes (`block_size_m`, `block_size_k`) that demonstrate the fundamental tradeoffs in GPU kernel design:

- **Larger blocks**: Higher SRAM usage, better cache locality, fewer kernel launches
- **Smaller blocks**: Lower SRAM pressure, better for smaller GPUs
- **Experimentation**: Adjusting these values illustrates how GPU resources constrain optimization

**Suggested Values for Learning** (for A100/H100):
- RMSNorm: `block_size_m=32`, `block_size_k=128`
- SwiGLU: `block_size_m=32`, `block_size_k=64`

### Memory Optimization Concepts

- **Quantization**: Using AWQ (8-bit) reduces memory by 75% vs FP32 - a key strategy in production systems
- **Batch Processing**: Larger batches improve GPU utilization - demonstrates the importance of amortizing kernel overhead
- **Kernel Fusion**: Multiple operations fused to reduce memory bandwidth - a critical optimization technique

### Performance vs. Native vLLM

**Important Note**: These custom kernels are **not** designed to exceed native vLLM performance. Native vLLM implementations include:
- Extensive platform-specific optimizations
- Careful tuning for various GPU architectures
- Sophisticated memory management and scheduling
- Production-level code hardening

**The value of this project** lies in understanding *how* and *why* these optimizations work, not in achieving production performance. For production inference, native vLLM implementations are strongly recommended.

## Model Information

**Model**: Llama-3-8B-Instruct-AWQ
- **Architecture**: Llama-3 with Instruction Tuning
- **Quantization**: AWQ (Activation-aware Weight Quantization)
- **Parameters**: 8 Billion
- **Source**: [HuggingFace Model Card](https://huggingface.co/casperhansen/llama-3-8b-instruct-awq)

## Troubleshooting

### Common Issues

**Issue**: `RuntimeError: CUDA out of memory`
- **Solution**: Reduce batch size, use smaller block sizes, or enable gradient checkpointing

**Issue**: `vLLM import fails`
- **Solution**: Reinstall vLLM: `pip uninstall vllm && uv pip install vllm==0.18.0 --torch-backend=auto`

**Issue**: `Triton kernel compilation error`
- **Solution**: Update CUDA toolkit and verify GPU compute capability

### Environment Variables

```bash
# Enable detailed logging
export TRITON_PRINT_IR=1

# Disable multiprocessing (for Colab compatibility)
export VLLM_ENABLE_V1_MULTIPROCESSING=0

# Set CUDA device
export CUDA_VISIBLE_DEVICES=0
```

## References

### Academic Papers
- Llama-3 Architecture: https://arxiv.org/abs/2407.21783
- Triton: A Language and Compiler for Efficient GPU Kernels: https://arxiv.org/abs/1910.01093
- SwiGLU: GLU Variants Improve Transformer: https://arxiv.org/abs/2002.05202

### Related Resources
- [vLLM Documentation](https://docs.vllm.ai/)
- [Triton Official Documentation](https://triton-lang.org/)
- [HuggingFace Model Hub](https://huggingface.co/)
- [PyTorch Documentation](https://pytorch.org/docs/)

## Contributing

Contributions are welcome! This is an educational project, so contributions that enhance learning value are particularly appreciated:
- Clearer explanations of kernel concepts
- Additional kernels demonstrating different optimization techniques
- Performance profiling and analysis notebooks
- Comparative studies with native implementations
- Documentation improvements and tutorials
- Bug fixes and code quality improvements

## Citation

If you reference this project in your learning or research, please cite:

```bibtex
@software{custom_triton_kernels,
  title={Custom-Triton-Kernels: Educational GPU Kernel Development with Triton},
  author={Thakur, Prathamesh},
  year={2026},
  note={Educational repository for learning kernel optimization and LLM operator implementation}
}
```

## License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2026 Prathamesh-Thakur

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```

## Contact & Support

For issues, questions, or suggestions:
- Open an issue on the GitHub repository
- Check the HuggingFace model documentation
- Refer to the Triton and vLLM official documentation

---

**Last Updated**: May 2026
**Status**: Active Development
**Python Version**: 3.8+
**CUDA Version**: 11.8+