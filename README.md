# Mixed Precision Training 
### in PyTorch

-----------
Training in FP16 that is in half precision results in slightly faster training in nVidia cards that supports half precision ops. Also the memory requirements of the models weights are almost halved since we use 16-bit format to store the weights instead of 32-bits.

Although training in half precision has it's own caveats.
The problems that is encountered in half precision training are:  
 - Imprecise weight update
 - Gradients underflow
 - Reductions overflow

Below is a discussion on how to deal with these problems.

#### FP16 Basics
IEEE-754 floating point starndard states that given a floating point number *X* if,  
`2^E <= abs(X) < 2^(E+1)`
then the distance from *X* to the next largest representable floating point number *epsilon* is:  
 - `epsilon = 2^(E-52)`     [For a 64-bit float (double precision)]
 - `epsilon = 2^(E-23)`    [For a 32-bit float (single precision)]
 - `epsilon = 2^(E-10)`    [For a 16-bit float (half precision)]

The above equations allow us to compute the following:

- For half precision...

    If you want an accuracy of `+/-0.5 (or 2^-1)`, the maximum size that the number can be is `2^10`. Any larger than this and the distance between floating point numbers is greater than 0.5.

    If you want an accuracy of `+/-0.0005 (about 2^-11)`, the maximum size that the number can be is `1`. Any larger than this and the distance between floating point numbers is greater than 0.0005.

 - For single precision...

    If you want an accuracy of `+/-0.5 (or 2^-1)`, the maximum size that the number can be is `2^23`. Any larger than this and the distance between floating point numbers is greater than 0.5.

    If you want an accuracy of `+/-0.0005 (about 2^-11)`, the maximum size that the number can be is `2^13`. Any larger than this and the distance between floating point numbers is greater than 0.0005.

- For double precision...

  If you want an accuracy of `+/-0.5 (or 2^-1)`, the maximum size that the number can be is `2^52`. Any larger than this and the distance between floating point numbers is greater than 0.5.

  If you want an accuracy of `+/-0.0005 (about 2^-11)`, the maximum size that the number can be is `2^42`. Any larger than this and the distance between floating point numbers is greater than 0.0005.

#### Imprecise Weight Update
Thus while training our network we'll need that added precision, since our weights will go through small updates. For example `1 + 0.0001` will result in:
 - `1.0001` in FP32
 - but in FP16 it will be `1`

What that means is that we risk underflow (attempting to represent numbers so small they clamp to zero) and overflow (numbers so large they become NaN, not a number). With underflow, our network never learns anything, and with overflow, it learns garbage.  
To overcome this we keep a "FP32 master copy". It is a copy of out FP16 model weights in FP32. We use these master params to update our weights and then copy them back to our model. We also update the gradients in the master copy as they are calculated in the model.

#### Gradients Underflow
Gradients are sometime not representable in FP16. This leads to the gradient underflow problem. A way to deal with this problem is to shift the gradient bitwise, so that they are in a range representable by half-precision floats. A way to do this is to multiply the loss by a large number like `2^7` that would shift the computed gradients during `loss.backward()` to the FP16 representable range. Then when we copy these gradients to FP32 master copy, we scale them back down by dividing the gradients with the same scaling factor.

#### Reduction Overflow
Another caveat with half-precision is that while doing large reductions it may overflow. For example consider two tensor: 
- a = `torch.Tensor(4094).fill_(4.0).cuda()`
- b = `torch.Tensor(4095).fill_(4.0).cuda()`

If we were to do `a.sum()` and `b.sum()` it would result in `16376` and `16380` respectively, as expected in single-point precision. But if we did the same ops in half point precision it would result in `16376` and `16384` respectively. To overcome this problem we do the reduction ops like `BatchNorm` and loss calculation in FP32.

All these problems have been kept in mind to help us successfully train with FP16 weights. Implementation of the above ideas can be found in the `train.py` file.

#### Usage Instruction
```
python main.py [-h] [--lr LR] [--steps STEPS] [--gpu] [--fp16] [--loss_scaling] [--model MODEL]

PyTorch (FP16) CIFAR10 Training

optional arguments:
  -h, --help            Show this help message and exit
  --lr LR               Learning Rate
  --steps STEPS, -n STEPS
                        No of Steps
  --gpu, -p             Train on GPU
  --fp16                Train with FP16 weights
  --loss_scaling, -s    Scale FP16 losses
  --model MODEL, -m MODEL
                        Name of Network
```
To run in `FP32` mode, use:  
`python main.py -n 200 -p --model resnet50`

To train with `FP16` weights, use:  
`python main.py -n 200 -p --fp16 -s --model resnet50`  
`-s` flag enables loss scaling.

#### Results
Training on a single P100 GPU, I was able to obtain the following result, while training with ResNet50 with a batch size of 128 over 200 epochs.  

|  | FP32 | Mixed Precision |
|------------|:-----:|:---------------:|
| Time/Epoch | 1m32s | 1m15s |
| Storage | 90 MB | 46 MB |
| Accuracy | 94.50% | 94.43% |

--------------
##### TODO
- [ ] Test with all nets.
- [ ] Test models on Volta GPUs.
- [ ] Test runtimes on multi GPU setup.
--------------
##### Further Explorations:
- Training with INT8 weights. Yes, the weights can be quantanized to 8-bit integers. See [Training and Inference with Integers in Deep Neural Networks](https://arxiv.org/pdf/1802.04680.pdf).
- Pushing all boundaries is quatanizing gradients to 1-bit. You can read about the same in [SIGNSGD: Compressed Optimisation for Non-Convex Problems](https://arxiv.org/pdf/1802.04434.pdf).
--------------
##### Convenience:
nVidia provides the [apex](https://github.com/NVIDIA/apex) library that handles all the caveats of training in mixed precision. It also provides API for multiprocess distributed training with NCCL and `SyncBatchNorm` which reduces stats across processes during multiprocess distributed data parallel training.

---------------
##### Thanks:
The project heavily borrows from @[kuangliu](https://github.com/kuangliu)'s project [pytorch-cifar](https://github.com/kuangliu/pytorch-cifar). The models have been directly borrowed from the repository with minimal change, so thanks to @kuangliu for maintaining such awesome project.
