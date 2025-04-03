# Building and Optimizing a Custom Trained GPT2 Model

This project involved building a Language Model (LLM) from scratch in a distributed environment and optimizing it for inference. The process was guided by insights gained from the GPT series video lectures and DDP video tutorials.

## Project Overview

- **Learned From:** GPT series video lectures and DDP video tutorials.
- **Goal:** Build an LLM from scratch on a distributed environment.
- **Model:** Trained a GPT2 model.
- **Training Data:** flytech/python-codes-25k dataset.
- **Training Hardware:** Multi T4 GPUs.

## Inference Optimization

The custom-trained GPT2 model was optimized using quantization techniques to improve inference performance.

### Quantization Methods

- **Static Quantization:** Range of weights and activations was analyzed using a calibration dataset (details not included in this summary).
- **Dynamic Quantization:** During inference, quantization parameters for activations were computed on-the-fly based on the input data's range. Inference was performed with mixed precision (INT8 weights, dynamically quantized activations).

### Validation Results

| Model                     | Validation Loss | Model Size      |
|---------------------------|-----------------|-----------------|
| Original Model            | 1.3223          | 620.33 MB       |
| Dynamic Quantized Model   | 1.3350          | 148.88 MB       |            

The dynamic quantized model shows a slight increase in validation loss compared to the original model.

### Inference Performance

Inference and validation were performed on a system with the following specifications:

- **CPU:** 16gb i5 8 core CPU

### Performance Metrics (for 250 tokens)

| Metric         | Original Model        | Quantized Model       |
|----------------|-----------------------|-----------------------|
| Time Taken     | 84.72966575622559 s | 62.362414598464966 s |
| Tokens/s       | 2.950560441478326   | 4.0088248925203365  |

The dynamic quantized model demonstrates a slight improvement in inference speed (Tokens/s) compared to the original model.

### Output Examples

### Original Model Output (for "Python program to calculate the average")

```python
 Python program to calculate the average age of a group of people people = [
 {'name': 'John', 'age': 30},
 {'name': 'Jane', 'age': 28},
 {'name': 'Jane', 'age': 28}
]
def calculate_average_age(grades):
    total = 0
    for record in people:
        total += record['age']
    return total / len(people)
```

### Dynamic Quantize Model Output (for "Python program to calculate the average")
```python
 Python program to calculate the average age of a person based on their weight and weight in the school
def calculate_average_age(weight):
  total_age = 0
  for person in weight:
    total_age += person['age']
  return total_age
```

- We can see the quantize model made some error in the output code, as these models are autoregressive and most of the token outputs are probabilistic based on the past words. 
- Creating a more sensible outputs which are accurate to the real world is challeneging, will be working on the  advanced methods(Static Quantization, QAT, RoPE and Flash Attention) to improve the model performance.

## KV Cache FLOPS Comparison

KV cache stores the Key and Value projections from the previous tokens. By doing so, at the time of token generation, the model does not need to recompute the Keys and Values for the sequence in each step of generation. This reduces the computational cost and memory bandwidth requirements. When generating the next token, the model only needs to compute the query based on the current token. This query is compared with the cached keys, and the attention scores are used to weight the cached values to produce attention outputs for the current token.

Assuming the following configuration:

* `batch`
* `context_length`
* `emb_dim`
* `num_head`
* `head_dim = emb_dim // num_head`

**1. Query, Key, and Value**

* `X = (batch, context_len, emb_dim)`
* Dimension of `(Wq, Wk, Wv)` -> `(emb_dim, emb_dim)`
* `Q = X.Wq`
* `K = X.Wk`
* `V = X.Wv`

* FLOPS of `(batch, context_len, emb_dim) . (emb_dim, emb_dim)`
    ```
    = 3 * batch * (2 * context_len * emb_dim * emb_dim)
    = 6 * batch * context_length * emb_dim * emb_dim
    ```

**2. For attention**

* `Softmax(Q * K.T / sqrt(emb_dim))`

    * `(batch, num_heads, context_length, head_dim) . (batch, num_heads, head_dim, context_length) -> (batch, num_heads, context_length, context_length)`
    * `batch * (2 * context_length * head_dim * context_length )`

* `Softmax(Q * K.T / sqrt(emb_dim)) * V`

    * `(batch, num_heads, context_length, context_length) * (batch, num_heads, context_length, head_dim) -> (batch, num_heads, context_length, head_dim)`
    * `batch * (2 * context_length * context_length * head_dim)`
    * This is across all the heads.
    * `num_head * batch * (2 * context_length * context_length * head_dim)`

For KV cache, the context length becomes 1, as we will be taking only a single token input, and we keep appending the K and V to the cache.

**3. KV cache**

* `context_len = 1`
* FLOPS of `(batch, context_len, emb_dim) . (emb_dim, emb_dim)`
    ```
    = 3 * batch * (2 * context_len * emb_dim * emb_dim)
    = 6 * batch * emb_dim * emb_dim
    ```
* In the softmax, Q is only a single row (emb_dim column vector).
* `(batch, num_heads, 1, head_dim) . (batch, num_heads, head_dim, KV_CACHE) -> (batch, num_heads, 1, KV_CACHE)`
    * `batch * (2 * context_length * head_dim * KV_CACHE )`
* `(batch, num_heads, 1, KV_CACHE) * (batch, num_heads, KV_CACHE, head_dim) -> (batch, num_heads, 1, head_dim)`
    * `batch * (2 * context_length * KV_CACHE * head_dim)`
    * `num_head * batch * (2 * context_length * KV_CACHE * head_dim)`

python
```
def total_flops_no_cache(batch = 8, context_length = 512, emb_dim = 768, num_heads = 12):
    head_dim = emb_dim//num_heads
    flops = 6 * batch * context_length * emb_dim * emb_dim + \
            batch * (2 * context_length * head_dim * context_length ) + \
            num_heads * batch * (2 * context_length * context_length * head_dim)
    return flops

def total_flops_kv_cache(batch = 8, context_length = 512, emb_dim = 768, num_heads = 12, KV_CACHE = 512): # for larger models KV_CACHE can be 8096 or more
    head_dim = emb_dim//num_heads
    flops = 6 * batch * emb_dim * emb_dim + \
            batch * (2 * context_length * head_dim * KV_CACHE ) + \
            num_heads * batch * (2 * context_length * KV_CACHE * head_dim)

    return flops


no_cache = total_flops_no_cache()
kv_cache = total_flops_kv_cache()
print("Total FLOPS without cache: ", no_cache)
print("4 bytes precision without cache Memory Bandwidth:", no_cache * 4 * 1e-9, "GB/s")
print("Total FLOPS with cache: ", kv_cache)
print("4 bytes precision with kv cache Memory Bandwidth:", kv_cache * 4 * 1e-9, "GB/s")

```

|                                                 |                   |
| :---------------------------------------------- | :---------------- |
| Total FLOPS without cache                       | 17,985,175,552    |
| 4 bytes precision without cache Memory Bandwidth| 71.94 GB/s        |
| Total FLOPS with cache                          | 3,517,972,480     |
| 4 bytes precision with kv cache Memory Bandwidth| 14.07 GB/s        |

- Lesser FLOPS means lesser computations and lesser memory accesses.

#### Resources
1. https://r4j4n.github.io/blogs/posts/kv/
2. https://levelup.gitconnected.com/lets-build-an-optimizer-for-a-gpt-model-from-scratch-in-pytorch-kv-caching-4d3f1f9516fa
