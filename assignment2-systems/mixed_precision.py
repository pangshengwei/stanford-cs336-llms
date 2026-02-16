import torch

s = torch.tensor(0,dtype=torch.float32)
for i in range(1000):
    s += torch.tensor(0.01,dtype=torch.float32)
print(s)

s = torch.tensor(0,dtype=torch.float16)
for i in range(1000):
    s += torch.tensor(0.01,dtype=torch.float16)
print(s)

s = torch.tensor(0,dtype=torch.float32)
for i in range(1000):
    s += torch.tensor(0.01,dtype=torch.float16)
print(s)

s = torch.tensor(0,dtype=torch.float32)
for i in range(1000):
    x = torch.tensor(0.01,dtype=torch.float16)
    s += x.type(torch.float32)
print(s)

""" RESULTS:
tensor(10.0001)
tensor(9.9531, dtype=torch.float16)
tensor(10.0021)
tensor(10.0021)

we can see that float16 has significant precision issues when summing small values, while float32 can maintain much better precision. 
When we add a float16 value to a float32 accumulator, 
the float16 value is first converted to float32, 
which can help mitigate some of the precision loss compared to doing the entire sum in float16.

(a) Data Types with FP16 Autocasting

With PyTorch's autocast using FP16, here are the data types:

Model parameters: FP32 (parameters remain in their original dtype; autocast only affects operations, not stored parameters)
Output of fc1: FP16 (Linear/matmul operations are autocasted to FP16)
Output of LayerNorm: FP32 (LayerNorm is autocasted to FP32 for numerical stability)
Model's predicted logits (fc2 output): FP16 (Linear layer output is FP16)
Loss: FP32 (loss calculations are typically promoted to FP32)
Gradients: FP32 (gradients match the parameter dtype, which is FP32)

(b) LayerNorm Sensitivity to Mixed Precision

LayerNorm involves computing mean, variance, and normalization (division operations), which are sensitive to the limited range of FP16 (especially variance accumulation and division can cause overflow/underflow). 
With BF16, the treatment is less critical because BF16 has the same exponent range as FP32 (8 bits), giving it much better range than FP16, making it less prone to overflow/underflow during variance computation. 
However, PyTorch still typically promotes LayerNorm to FP32 even with BF16 for maximum numerical stability, though the necessity is reduced compared to FP16.
"""