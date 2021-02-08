# Quanvolutions
Playing around with Quanvolutional Neural Nets

## Background
Quanvolutional neural nets are a quantum analogue of CNNs. They replace the kernel of the convolution with a parameterised quantum circuit. The original paper can be found [here](https://arxiv.org/pdf/1904.04767.pdf). Quanvolutions are thought to be a promising avenue for QML in the near term, as the quantum circuit only needs to have a number of qubits that is of the order of the size of the kernel, which can be reasonably small and still achieve good results (3x3 is a fairly standard kernel size in many models), and so few-qubit quantum processors can feasibly be used to implement hybrid quanvolutional architectures.

## Some of my thoughts
The original paper uses fixed quantum circuits as kernels for the convolution operation. This is somewhat analogous to early CNNs, where different kernels were convolved with images to extract different features (for example

```python
[[-1,0,1],
 [-2,0,2],
 [-1,0,1]]
 ```
 would extract vertical edges in the image (this is the Sobel edge detection kernel). However, the true power of CNNs was realised when we allowed the kernel weights to be learned directly by the model. Therefore, I propose that by using parameterised quantum kernels with learnable parameters, we can benefit from the classically intractable non-linearities introduced by substituting convolutions for quanvolutions in the network architecture, but without having to sacrifice learnability of the kernel.
 
 ## Implementation
 The code here is based on [Pennylane](https://github.com/PennyLaneAI/PennyLane), a great Python library for QML. In `quanv.py`, I provide a working 2D quanvolutional layer in the Pennylane Pytorch interface, and also a currently not-working (due to parameter gradients not being propagated correctly through the network) version for the Pennylane Keras interface. The `torch.ipynb` notebook gives an example of how to build a quanvolutional net and train it on the MNIST dataset. I intend to use this framework to explore how different architectures affect the ability of the model to learn the MNIST data, specifically focusing on how using different quantum circuits as kernels affects model performance. I also hope to soon run some experiments on an actual QPU.
