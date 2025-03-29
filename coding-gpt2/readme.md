# Building and Optimizing a Custom Trained GPT2 Model

This project involved building a Language Model (LLM) from scratch in a distributed environment and optimizing it for inference. The process was guided by insights gained from the GPT series video lectures and DDP video tutorials.

## Project Overview

- **Learned From:** GPT series video lectures and DDP video tutorials.
- **Goal:** Build an LLM from scratch on a distributed environment.
- **Model:** Trained a GPT2 model.
- **Training Data:** flytech/python-codes-25k dataset.
- **Training Hardware:** Multi T4 GPUs.

## Model Optimization

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

## Inference Performance

Inference and validation were performed on a system with the following specifications:

- **CPU:** 16gb i5 8 core CPU

### Performance Metrics (for 250 tokens)

| Metric         | Original Model        | Quantized Model       |
|----------------|-----------------------|-----------------------|
| Time Taken     | 84.72966575622559 s | 62.362414598464966 s |
| Tokens/s       | 2.950560441478326   | 4.0088248925203365  |

The dynamic quantized model demonstrates a slight improvement in inference speed (Tokens/s) compared to the original model.

## Output Examples

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
- follow: https://x.com/hetu_1010
