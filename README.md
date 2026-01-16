# LLM - Transformer from Scratch

## Brief Foreword

This project implements a simple self-attention transformer architecture from scratch for language models in particular in C++ and CUDA with a modularized design, allowing for easy experimentation with different compontent structures. It includes a command-line interface (CLI) for training and inference, as well as a simplified web-based user interface (UI) for a more interactive experience.

My main goal with this project has been to learn concretely how neural networks work at a low-level via its most prevalent architecture today. I hope to also provide a useful reference for those without deep ML expertise and who may also be a little intimidated by the complexity of existing frameworks, as I was and still consider myself to be a bit of a layman to the field. For a simplified and detailed overview of the project, please refer below in the [Concise Overview](#concise-overview) section. However, if you are unfamiliar with transformers, I want to detail how I built up my own intuition for how they work in a way that is hopefully accessible to those without a deep background in machine learning.

## An Intuition for Transformers

### Keywords

For now until I have a more comprehensive write-up, I want to share the most important concepts I believe are necessary to understand this codebase. If you are unfamiliar with transformers, I recommend first familiarizing yourself with the following concepts:

 - Tokens
 - Embeddings
 - Feed-Forward layers
 - Activation Function
 - Self-Attention / Multi-Head Attention
 - Layer Normalization
 - Residual Connections
 - Backpropagation
 - Optimizers (SGD, Adam, etc.)
 - Logits and Softmax

### Building up from Linear Regressions

I'd like to just focus on one simple part of the transformer architecture: the feed-forward layer. If you do not have much or any experience with linear algebra or multivariable calculus, as I did not before starting this project, I'd like for you to consider the most basic form of a self-learning algorithm: a linear regression, or a "Line of Best Fit". If you have a set of data points on a grid with some linear-looking relationship, you can draw a line that most accurately "cuts through" the datapoints. This line can be represented by ```y = mx + b```, where `m` is the slope of the line and `b` is the y-intercept. The goal of linear regression is to find the values of `m` and `b` that minimize the difference between the predicted values (the line) and the actual data points.

Now if this was all that you knew about neural networks, naturally you'd wonder if you could layer multiple of these linear regressions to create a more complex model. The problem however can be seen if you consider just two layers of consecutive linear regressions:

```y_1 = m_1 * x_1 + b_1```
```y_2 = m_2 * x_2 + b_2```

```y_2 = m_2 * (m_1 * x_1 + b_1) + b_2```
```y_2 = (m_2 * m_1) * x_1 + (m_2 * b_1 + b_2)```

*Therefore:* ```y_2 = mx + b``` where ```m = m_2 * m_1``` and ```b = m_2 * b_1 + b_2```

If we stack two linear regressions together, we create another linear regression! We have just recreated the simple ```y = mx + b``` formula only with more constants, meaning two consecutive linear layers is no more powerful than one. The solution to this is clever, still thinking in this one-dimension context, we introduce some "Activation Function", one that is not easily represented by a linear function. It could be as simple as:

```y = { x if x > 0; 0 otherwise }```

And if we now stack two layers together with an activation inbetween:

```y_1 = m_1 * x_1 + b_1```
```y_2 = { y_1 if y_1 > 0; 0 otherwise }```
```y_3 = m_2 * y_2 + b_2```

We no longer can simplify this into a single linear regression, and have created a more powerful model! What we have now created is the most basic form of a Feed-Forward layer. The next step you need to do is imagine this being done in the context of the y and x variables being matrices instead of single values.

### An Intuition for Matrix Operations

For simplicity, if we imagine a 'token' to cleanly represent one word, an 'embedding' is essentially a decomposition of that word into a string of numbers, each representing some characteristic of that word in its meaning. Neural networks are nearly incomprehensible in its innards, so its hard to say exactly what these characteristics are, as they are nebulously defined by the model itself. What is important for understanding the layers of a transformation is that the generic data piece we start with at the beginning of any transformation begins as a string of tokens that the model intakes, with each token being broken up into a vector of numbers (the embedding). This matrix is of size ```S x D```, where S is the input sequence length, and D is the embedding dimension / amount of embeddings the model uses per token.

In the context of a feed-forward layer, this ```y = mx + b``` design needs to be adapted to the requirements of matrix multiplication. Matrix multiplication is a function:

```A x B :: (M_1 x N_1) x (M_2 x N_2) -> (M_1 x N_2) where N_1 = M_2```

Or in words, the "inside dimensions" (N_1, M_2) of the (M_1 x N_1), (M_2 x N_2) matrices much match, and the outside ones represent the dimensions of the resulting matrix. The operation itself is not super important for the very basics, however what is important is that it mixes a large amount of the values together in the result. If we imagine a language model to be a self-complexifying function that mixes together all of the data it is given in some way it has found to be useful, matrix multiplication achieves this "mixing everything together" very well.

However this restriction of the dimensions means we only have so many choices for how we can create a feed-forward layer. Ideally, we want for simplicity and the modular nature of this project for each layer to take in this ```S x D``` matrix and output another ```S x D``` matrix. Let's consider the single linear regression first:

*Note that matrix addition here is an element-wise operation, meaning the two matrices being added must be of the same size.

```y = mx + b```
We Know:
x :: (S x D)
m :: (? x S)

Therefore: 
b :: (? x D)
y :: (? x D)

However what we have discovered with these question marks is that the output is not the right size and we cannot get it there here, so we have some freedom to choose a value for this layer. We have some 'internal dimension' to dictate how big the feed-forward layer projects to because we need to do something else to get it back to the right size afterwards. Here we also need to apply our activation function for the reasons mentioned before, which will just entail applying the function element-wise. I'll just call this new function ```y'``` for clarity.

What we need to do to get back to the right size is another linear regression:

```y_after = m' * y' + b'```
We Know:
y_after :: (S x D)
y' :: (D x ?)

Therefore:
m' :: (? x D)
b' :: (D x D)

Putting it all together, we have:

```y = m_1 * x + b_1```
```y' = activation(y)```
```y_after = m_2 * y' + b_2```

This is the structure of the feed-forward layer that can be found in [src/inference/ff_layer.h](src/inference/ff_layer.cpp)!

### Intuition for Backpropogation

I am eliding some details here in this progression, an explanation for how the model generates a prediction via its logit layer is another thing worth explaining at some point. However what what I found more useful for my intuition is backpropagation. The most simple way one could imagine training this model is to simply randomly tune its knobs (the m/b variables in the feed-forward layer for example, as one would for generating a line-of-best-fit), and see if the model gets better or worse. However the very clever thing about neural networks is that they are remarkably ordered despite how chaotic they are in practice. 

The main point here is that we can imagine some "loss gradient". Back in the example of the line-of-best-fit, a simple way to train this line is to determine if tweaking a parameter (the slope/intercept) up would increase or decrease the accuracy, if it would increase it than the parameter should be increased, and vice-versa.

This "increase or decrease" is actually just a derivative. It is the derivative of the loss (how inaccurate we were) with regards to some parameter. If we extend this idea to matrices, we can imagine this in context of a gradient, or the derivative of the loss with regards to each of the parameters in the matrix. The main idea that supports this in a neural network like here is that we can get to every layer the gradient of the loss with regards to the output of the layer via a long string of chain rules.

For brevity I also will assert three very important formulas that are useful here:
```AB = C -> dL/dA = dL/dC * B^T && dL/dB = A^T * dL/dC```
```A + B = C -> dL/dA = dL/dC && dL/dB = dL/dC```
```f(x) = y -> dL/dx = dL/dy * f'(x)```

These are a bit involved to prove, but taking them at face value, you can use them to move backwards through a layer and get the gradients for each of the parameter matrices, and by extension each of the matrices values, which you can then use to update the parameters in the direction that minimizes the loss. This is the core idea behind backpropagation, and it is what allows neural networks to learn from data in a remarkably efficient way. If you take some time to try to fit these formulas into the context of the feed-forward layer described before, you can derive the implementation found in [src/inference/ff_layer.cpp](src/inference/ff_layer.cpp)!

# Concise Overview

## Prerequisites

- **CUDA Toolkit** - Required for GPU acceleration
- **CMake** (≥ 3.25) and **Ninja** build system
- **C++ compiler** with C++23 support
- **Node.js** and **npm** (only if using the web UI)

## Building

```bash
# Release build
./build.sh

# Debug build (with MATRIX_CHECKS enabled)
./build.sh debug

# Force rebuild
./build.sh rebuild
```

The binary will be built in `build/llm` (or `build-debug/llm` for debug).

## Commands

### 1. Train a Tokenizer

```bash
./llm train-tokenizer \
  --corpus <path/to/corpus> \
  --output <output_tokenizer.json> \
  --vocab-size <size> \
  [--dataset-type raw|row-based]
```

### 2. Initialize a Model

```bash
./llm init-model \
  --output <output_model.bin> \
  --tokenizer <tokenizer.json> \
  [--dimensions <n>] \
  [--heads <n>] \
  [--layers <n>] \
  [--activation <leaky_relu|gelu|swiglu>]
```

Default architecture: `--dimensions 128 --heads 4 --layers 4 --activation swiglu`

### 3. Train a Model

```bash
./llm train \
  --data <dataset.txt> \
  --tokenizer <tokenizer.json> \
  --output-model <output_model.bin> \
  --input-model <input_model.bin> \
  [--dataset-type raw|row-based] \
  [-n <number_of_rows>] \
  [--learning-rate <value>]
```

Default learning rate: `0.0001`

### 4. Predict/Generate

```bash
./llm predict \
  --model <model.bin> \
  --tokenizer <tokenizer.json> \
  --prompt "Your prompt here" \
  [--length <num>] \
  [--temperature <value>]
```

- `--length`: Number of tokens to generate (default: 50)
- `--temperature`: Sampling temperature (default: 1.0)

### 5. Performance Benchmark

Runs a few iterations of inference and backpropogation over a training set and reports the time taken on the last iteration. This will be improved
in the future to provide average timing rather than just the last iteration, but for now it serves as a basic framework for benchmarking large 
optimizations.

```bash
./llm perf-model \
  --model <model.bin> \
  --tokenizer <tokenizer.json> \
  --data <dataset.txt> \
  [--dataset-type raw|row-based]
```

## Preparing Datasets

The dataset used most frequently in the development of this project is the TinyStories dataset. It is a collection of very short, simple stories designed for training incredibly small language models. It can be found [here](https://huggingface.co/datasets/roneneldan/TinyStories) on Hugging Face.

```bash
pip install datasets tqdm

python3 download_dataset.py \
  --num_stories 100000 \
  --output_file tinystories_100k.txt
```

The current architecture requires datasets to be stored in plaintext, with rows separated by '<|endoftext|>' tokens. If using the Web UI,
ensure that the dataset can be found in a '.datasets/' directory in the root of the project.

## Web UI

```bash
./start-web-ui.sh
```

Then visit `http://localhost:3000` in your browser.

---

## Project Structure

```
├── src/
│   ├── commands/        # CLI command handlers
│   ├── dataset/         # Dataset loading utilities
│   ├── inference/       # Core transformer architecture
│   ├── kernels/         # CUDA kernels for GPU operations
│   ├── tokenizer/       # Tokenization logic
│   └── util/            # Utilities (logging, matrix, etc.)
├── web-frontend/        # Web UI (TypeScript/Express)
├── CMakeLists.txt
└── build.sh             # Build script
```
