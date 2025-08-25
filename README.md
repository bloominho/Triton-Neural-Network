# Triton Neural Network

June 2025, Andrew Park

A Neural Network with Triton Implementation.

A neural network is composed of multiple layers. Normally, these layers are written with PyTorch. And Pytorch's backend is written with C and CUDA. 

This implementation uses Triton instead. Each layers are rewritten to use Triton instead of CUDA.

## Supported Layers

- Average Pool 2D
- Batch Norm 2D
- Convolution 2D
- Linear
- Max Pool 2D
- ReLU

## Project Structure

The TritonNN folder holds the files that are needed to replace Pytorch Layers with Triton Implemented Layers. 

The following files replace the forward function with Triton Kernels. 
- TritonAvgPool2d.py
- TritonBatchNorm2d.py
- TritonConv2d.py
- TritonLinear.py
- TritonMaxPool2d.py
- TritonReLU.py

The forward functions that are replaced by these files are written inside the 'kernel' folder. The files inside this folder implements the triton kernel of each forward pass.
- AvgPool2d.py
- BatchNorm2d.py
- Conv2d.py
- Linear.py
- MaxPool2d.py
- ReLU.py

## Accuracy
The triton implementation and torch implementation gives equal results. Therefore, the triton implementation can replace the torch implementation in terms of accuracy.

## Performance

Performance was measured based on ResNet18-CIFAR10 task. The time needed for inference was measured using both PyTorch based model and Triton based model. 

### - AvgPool2d
|            | PyTorch | Triton  | Incrementation (%) |
| ---------- | ------- | ------- | ------------------ |
| Time (ms)  | 27.1674 | 28.1231 | 3.5                |

### - BatchNorm2d
|            | PyTorch | Triton  | Incrementation (%) |
| ---------- | ------- | ------- | ------------------ |
| Time (ms)  | 27.1674 | 30.1854 | 11.1               |

### - Conv2d
|            | PyTorch | Triton  | Incrementation (%) |
| ---------- | ------- | ------- | ------------------ |
| Time (ms)  | 27.1674 | 58.5258 | **215**            |

### - Linear
|            | PyTorch | Triton  | Incrementation (%) |
| ---------- | ------- | ------- | ------------------ |
| Time (ms)  | 27.1674 | 29.8572 | 9.9                |

### - MaxPool2d
|            | PyTorch | Triton  | Incrementation (%) |
| ---------- | ------- | ------- | ------------------ |
| Time (ms)  | 27.1674 | 29.1238 | 7.2                |

Overall, we can see that Triton performs slightly poorer compared to Pytorch. This is because CUDA allows high optimization compared to Triton. 

### All Layers Triton
|            | PyTorch | Triton  | Incrementation (%) |
| ---------- | ------- | ------- | ------------------ |
| Time (ms)  | 27.1674 | 59.6288 | 219                |

When all layers are replaced with Triton layers, 2.19 times more time was needed for inference. We can see that the Triton kernel does not give as fast speed as the pytorch kernel does. As we’ve seen from the results before, most of the time increase was from convolution kernel. The convolution kernel cannot be optimized well with triton because of its intrinsic memory access characteristics. As a result, CUDA is a better choice for elaborately optimizing the memory accesses of convolution kernel.

Note that Triton is used for fast implementation with adequate optimizations. 

# Credits
Developed by Andrew Park.
