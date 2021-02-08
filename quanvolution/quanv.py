from pennylane import numpy as np
from tensorflow.keras.layers import Conv2D
import functools
import inspect
from collections.abc import Iterable
from typing import Optional
import pennylane as qml

try:
    import tensorflow as tf
    from tensorflow.keras.layers import Layer
    from pennylane.interfaces.tf import to_tf

    CORRECT_TF_VERSION = int(tf.__version__.split(".")[0]) > 1
except ImportError:
    # The following allows this module to be imported even if TensorFlow is not installed. Users
    # will instead see an ImportError when instantiating the KerasLayer.
    from abc import ABC

    Layer = ABC
    CORRECT_TF_VERSION = False


class Quanv2D(Conv2D):
    """
    Quanvolutional layer for Keras hybrid architecture
    """

    def __init__(
            self, qnode, weight_shapes: dict, output_dim: Iterable, weight_specs: Optional[dict] = None, *args, **kwargs
    ):
        if not CORRECT_TF_VERSION:
            raise ImportError(
                "Quanv2D requires TensorFlow version 2 or above. The latest "
                "version of TensorFlow can be installed using:\n"
                "pip install tensorflow --upgrade\nAlternatively, visit "
                "https://www.tensorflow.org/install for detailed instructions."
            )

        self.weight_shapes = {
            weight: (tuple(size) if isinstance(size, Iterable) else (size,) if size > 1 else ())
            for weight, size in weight_shapes.items()
        }

        if qml.tape_mode_active():
            self._signature_validation_tape_mode(qnode, weight_shapes)
            self.qnode = qnode

            dtype = tf.float32 if tf.keras.backend.floatx() == tf.float32 else tf.float64

            if self.qnode.diff_method != "backprop":
                self.qnode.to_tf(dtype=dtype)
        else:
            self._signature_validation(qnode, weight_shapes)
            self.qnode = to_tf(qnode, dtype=tf.keras.backend.floatx())

        # Output dim must be an iterable with at least 2 dimensions (possibly a third for channels) TODO check this
        if len(output_dim) != 3:
            raise ValueError("Output must have three dimensions (the 2 image dimensions and one for the channels)")
        else:
            self.output_dim = output_dim

        self.weight_specs = weight_specs if weight_specs is not None else {}

        self.qnode_weights = {}

        super().__init__(dynamic=True, *args, **kwargs)

    def _signature_validation_tape_mode(self, qnode, weight_shapes):
        sig = inspect.signature(qnode.func).parameters

        if self.input_arg not in sig:
            raise TypeError(
                "QNode must include an argument with name {} for inputting data".format(
                    self.input_arg
                )
            )

        if self.input_arg in set(weight_shapes.keys()):
            raise ValueError(
                "{} argument should not have its dimension specified in "
                "weight_shapes".format(self.input_arg)
            )

        param_kinds = [p.kind for p in sig.values()]

        if inspect.Parameter.VAR_POSITIONAL in param_kinds:
            raise TypeError("Cannot have a variable number of positional arguments")

        if inspect.Parameter.VAR_KEYWORD not in param_kinds:
            if set(weight_shapes.keys()) | {self.input_arg} != set(sig.keys()):
                raise ValueError("Must specify a shape for every non-input parameter in the QNode")

    def _signature_validation(self, qnode, weight_shapes):
        self.sig = qnode.func.sig

        if self.input_arg not in self.sig:
            raise TypeError(
                "QNode must include an argument with name {} for inputting data".format(
                    self.input_arg
                )
            )

        if self.input_arg in set(weight_shapes.keys()):
            raise ValueError(
                "{} argument should not have its dimension specified in "
                "weight_shapes".format(self.input_arg)
            )

        if qnode.func.var_pos:
            raise TypeError("Cannot have a variable number of positional arguments")

        if qnode.func.var_keyword:
            raise TypeError("Cannot have a variable number of keyword arguments")

        if set(weight_shapes.keys()) | {self.input_arg} != set(self.sig.keys()):
            raise ValueError("Must specify a shape for every non-input parameter in the QNode")

        defaults = {
            name for name, sig in self.sig.items() if sig.par.default != inspect.Parameter.empty
        }

        self.input_is_default = self.input_arg in defaults

        if defaults - {self.input_arg} != set():
            raise TypeError(
                "Only the argument {} is permitted to have a default".format(self.input_arg)
            )

    def build(self, input_shape):
        """Initializes the QNode weights.

        Args:
            input_shape (tuple or tf.TensorShape): shape of input data
        """
        for weight, size in self.weight_shapes.items():
            spec = self.weight_specs.get(weight, {})
            self.qnode_weights[weight] = self.add_weight(name=weight, shape=size, trainable=True, **spec)

        super().build(input_shape)

    @tf.function
    def quanv(self, image, **kwargs):
        """Convolves the input image with many applications of the same quantum circuit."""
        image = tf.expand_dims(image, 0)
        img_shape = image.shape
        out = tf.zeros(self.compute_output_shape(img_shape).as_list())

        print(self.kernel_size)
        print(self.strides)
        print(self.dilation_rate)
        print(self.padding)

        input_patches = tf.image.extract_patches(image, sizes=[1,self.kernel_size[0],self.kernel_size[1],1], strides=[1,self.strides[0],self.strides[1], 1], rates=[1,self.dilation_rate[0], self.dilation_rate[1],1], padding=self.padding.upper())

        input_patches = tf.squeeze(input_patches)
        input_patches = tf.reshape(input_patches, [input_patches.shape[0] * input_patches.shape[1], input_patches.shape[2]])

        def apply_circuit(input_patch):
            print(input_patch)
            q_results = self.qnode(
                input_patch, **kwargs
            )

            return q_results

        def apply_convolution(patches, kernel_function):
            if len(patches.shape) > 1:
                reconstructor = []
                for x in tf.unstack(patches):
                    reconstructor.append(apply_convolution(x, kernel_function))
                return tf.stack(reconstructor)
            else:
                return kernel_function(patches)


        convolved_patches = tf.map_fn(apply_circuit, input_patches)

        print(convolved_patches.shape)

        def extract_patches_inverse(x, y):
            _x = tf.zeros_like(x)
            _y = tf.image.extract_patches(_x, sizes=[1,self.kernel_size[0],self.kernel_size[1],1], strides=[1,self.strides[0],self.strides[1], 1], rates=[1,self.dilation_rate[0], self.dilation_rate[1],1], padding=self.padding.upper())
            grad = tf.gradients(_y, _x)[0]
            # Divide by grad, to "average" together the overlapping patches
            # otherwise they would simply sum up
            return tf.gradients(_y, _x, grad_ys=y)[0] / grad

        reconstructed_convolved_image = extract_patches_inverse(image, convolved_patches)

        return reconstructed_convolved_image


        # # Loop over the coordinates of the top-left pixel of 2X2 squares
        # for j in tf.range(0, img_shape[0], self.strides[0]):
        #     for k in tf.range(0, img_shape[1], self.strides[1]):
        #         # Process a squared 2x2 region of the image with a quantum circuit
        #         q_results = self.qnode(
        #             [
        #                 image[j, k, 0],
        #                 image[j, k + 1, 0],
        #                 image[j + 1, k, 0],
        #                 image[j + 1, k + 1, 0]
        #             ], **kwargs
        #         )
        #         # Assign expectation values to different channels of the output pixel (j/2, k/2)
        #         index_creator = lambda x: tf.constant([j//2, k//2, x])
        #         indices = tf.map_fn(index_creator, tf.range(tf.shape(out)[2]), dtype="int32")
        #         print(indices)
        #         update_creator = lambda x: q_results[x]
        #         updates = tf.cast(tf.map_fn(update_creator, tf.range(self.filters)), "float32")
        #         print(updates)
        #         tf.tensor_scatter_nd_add(out, indices, updates)
        # return out


    def call(self, inputs):
        """Evaluates the QNode on input data using the initialized weights.

        Args:
            inputs (tensor): data to be processed

        Returns:
            tensor: output data
        """
        outputs = []
        for x in inputs:  # iterate over batch

            if qml.tape_mode_active():
                res = self._evaluate_qnode_tape_mode(x)
                outputs.append(res)
            else:
                # The QNode can require some passed arguments to be positional and others to be
                # keyword. The following loops through input arguments in order and uses
                # functools.partial to bind the argument to the QNode.
                qnode = self.qnode

                for arg in self.sig:
                    if arg is not self.input_arg:  # Non-input arguments must always be positional
                        w = self.qnode_weights[arg]
                        qnode = functools.partial(qnode, w)
                outputs.append(self.quanv(x))

        return tf.stack(outputs)


    def _evaluate_qnode_tape_mode(self, x):
        """Evaluates a tape-mode QNode for a single input datapoint.

        Args:
            x (tensor): the datapoint

        Returns:
            tensor: output datapoint
        """
        kwargs = {**{k: 1.0 * w for k, w in self.qnode_weights.items()}}
        print(kwargs)
        return self.quanv(x, **kwargs)


    def compute_output_shape(self, input_shape):
        """Computes the output shape after passing data of shape ``input_shape`` through the
        QNode. Can just use the compute output shape of the Conv2D layer from which we inherit.

        Args:
            input_shape (tuple or tf.TensorShape): shape of input data

        Returns:
            tf.TensorShape: shape of output data
        """
        return super().compute_output_shape(input_shape)


    def __str__(self):
        detail = "<Quanv2D Keras Layer: func={}>"
        return detail.format(self.qnode.func.__name__)


    __repr__ = __str__

    _input_arg = "inputs"


    @property
    def input_arg(self):
        """Name of the argument to be used as the input to the Keras
        `Layer <https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer>`__. Set to
        ``"inputs"``."""
        return self._input_arg


import torch
from torch import nn
import pennylane as qml


class TorchQuanvLayer(nn.Module):
    """
    Quanvolutional layer for Pytorch hybrid architectures
    """
    def __init__(self, qnode_layer, kernel_size=(2,2), stride=2, out_channels=4):
        super(TorchQuanvLayer, self).__init__()

        self.kernel_size = kernel_size
        self.stride = stride
        self.out_channels = out_channels

        self.qnode = qnode_layer

    def quanv(self, inputs, in_channels=1):
        """
        Helper function to perform single channel 2D quanvolution
        :param inputs: One channel of the input batch
        :param in_channels: The number of channels in the original input image (which helps calculate output shape for
        a single channel)
        :return:
        """

        input_patches = torch.nn.functional.unfold(inputs, kernel_size=self.kernel_size, stride=self.stride)

        convolved_patches = torch.stack([self.qnode(inputs=x) for x in torch.unbind(input_patches, dim=-1)])

        out_shape = (inputs.shape[0], self.out_channels // in_channels, (inputs.shape[2] - self.kernel_size[0]) // self.stride + 1,
                     (inputs.shape[3] - self.kernel_size[1]) // self.stride + 1)

        out = convolved_patches.view(out_shape)

        return out

    def forward(self, inputs):
        """
        Applies the quanvolution operation for the forward pass
        Args:
            img: An image

        Returns:

        """
        # unstack the different channels, apply convolutions, restack together
        return torch.cat([self.quanv(x.unsqueeze(1), in_channels=inputs.shape[1]) for x in torch.unbind(inputs,dim=1)], 1)
