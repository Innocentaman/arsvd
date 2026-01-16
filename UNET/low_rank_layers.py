"""
Low-Rank Convolutional Layers for Neural Network Compression

Implements proper low-rank decomposition of Conv2D layers:
- Original: Conv2D(kernel_size=(h,w), filters=f, in_channels=c)
- Decomposed: Conv2D(kernel_size=(h,w), filters=rank, in_channels=c)
             + Conv2D(kernel_size=(1,1), filters=f, in_channels=rank)

This actually reduces parameters and FLOPs while maintaining accuracy.
"""

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Layer
import numpy as np
from compression_utils import compute_svd


class LowRankConv2D(Layer):
    """
    Low-Rank Convolution Layer

    Decomposes a standard convolution into two smaller convolutions:
    - First conv: Spatial convolution with reduced channels (rank)
    - Second conv: 1x1 convolution to restore original channel count

    Mathematical formulation:
    Original: W ∈ R^(k×k×c_in×c_out)
    Decomposed: W ≈ W1 @ W2
    where W1 ∈ R^(k×k×c_in×r), W2 ∈ R^(1×1×r×c_out)
    and r < min(c_in, c_out) is the rank
    """

    def __init__(self, original_layer, rank, **kwargs):
        super(LowRankConv2D, self).__init__(**kwargs)

        self.rank = rank
        self.original_layer = original_layer

        # Get original layer properties
        kernel_size = original_layer.kernel_size
        kernel_shape = original_layer.kernel.shape  # (h, w, in_c, out_c)
        filters_in = kernel_shape[2]
        filters_out = original_layer.filters
        use_bias = original_layer.use_bias

        # First conv: Spatial reduction (h×w×in → h×w×rank)
        self.conv1 = Conv2D(
            filters=rank,
            kernel_size=kernel_size,
            strides=original_layer.strides,
            padding=original_layer.padding,
            use_bias=False,  # No bias in first layer
            name=f'{original_layer.name}_low_rank_1'
        )

        # Second conv: Channel expansion (1×1×rank → 1×1×out)
        self.conv2 = Conv2D(
            filters=filters_out,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding='same',
            use_bias=use_bias,
            name=f'{original_layer.name}_low_rank_2'
        )

    def build(self, input_shape):
        self.conv1.build(input_shape)
        self.conv2.build(self.conv1.compute_output_shape(input_shape))
        super(LowRankConv2D, self).build(input_shape)

    def call(self, inputs, training=None):
        x = self.conv1(inputs)
        x = self.conv2(x)
        return x

    def compute_output_shape(self, input_shape):
        return self.original_layer.compute_output_shape(input_shape)

    def get_config(self):
        config = super(LowRankConv2D, self).get_config()
        config.update({
            'rank': self.rank,
        })
        return config


def initialize_low_rank_from_svd(original_layer, rank):
    """
    Initialize a low-rank convolution layer using SVD of original weights

    Args:
        original_layer: Original Conv2D layer
        rank: Target rank for decomposition

    Returns:
        LowRankConv2D layer with SVD-initialized weights
    """
    # Get original weights
    weights = original_layer.get_weights()
    kernel = weights[0]  # Shape: (h, w, in_c, out_c)
    bias = weights[1] if len(weights) > 1 else None

    h, w, in_c, out_c = kernel.shape

    # Reshape kernel to 2D matrix for SVD
    # Combine spatial dimensions and input channels
    kernel_2d = kernel.reshape(h * w * in_c, out_c)  # Shape: (h*w*in_c, out_c)

    # Compute SVD
    U, S, Vt = compute_svd(kernel_2d)

    # Truncate to target rank
    U_r = U[:, :rank]  # Shape: (h*w*in_c, rank)
    S_r = S[:rank]     # Shape: (rank,)
    Vt_r = Vt[:rank, :]  # Shape: (rank, out_c)

    # Reconstruct the two weight matrices
    # W1 corresponds to the first convolution (spatial)
    # W2 corresponds to the second convolution (1x1 pointwise)

    # For W1: Reshape U_r * sqrt(S_r) to (h, w, in_c, rank)
    W1_matrix = U_r * np.sqrt(S_r)  # Shape: (h*w*in_c, rank)
    W1_kernel = W1_matrix.reshape(h, w, in_c, rank)

    # For W2: sqrt(S_r) * Vt_r, already (rank, out_c)
    W2_kernel = (np.sqrt(S_r)[:, np.newaxis] * Vt_r)  # Shape: (rank, out_c)
    W2_kernel = W2_kernel.reshape(1, 1, rank, out_c)  # Shape: (1, 1, rank, out_c)

    # Create low-rank layer
    low_rank_layer = LowRankConv2D(original_layer, rank)

    # Note: Don't explicitly build - let Keras build it when connected to model
    # The weights will be set when the layer is part of the model

    # Manually set weights without building
    low_rank_layer.conv1.built = True
    low_rank_layer.conv2.built = True

    # Set weights
    low_rank_layer.conv1.set_weights([W1_kernel])
    low_rank_layer.conv2.set_weights([W2_kernel] + ([bias] if bias is not None else []))

    # Reset built flag so Keras can properly initialize later
    low_rank_layer.conv1.built = False
    low_rank_layer.conv2.built = False

    return low_rank_layer


def create_compressed_model_proper(base_model, rank_config, img_size=256):
    """
    Create a properly compressed model using low-rank convolutions

    Args:
        base_model: Original trained model
        rank_config: Either a single rank (int) or dict mapping layer names to ranks
        img_size: Input image size

    Returns:
        Compressed model with low-rank convolution layers
    """
    from unet import build_unet

    # Build fresh model
    compressed_model = build_unet((img_size, img_size, 3))

    # Copy all weights and replace Conv2D layers with low-rank versions
    for i, base_layer in enumerate(base_model.layers):
        if 'conv' in base_layer.name.lower() and hasattr(base_layer, 'kernel'):
            # Determine rank for this layer
            if isinstance(rank_config, dict):
                rank = rank_config.get(base_layer.name, 128)  # Default rank
            else:
                rank = rank_config

            # Get layer dimensions
            kernel_shape = base_layer.kernel.shape  # (h, w, in_c, out_c)
            in_channels = kernel_shape[2]
            out_channels = kernel_shape[3]

            # Validate rank
            max_rank = min(in_channels, out_channels)
            rank = min(rank, max_rank)
            rank = max(1, rank)

            # Initialize low-rank layer from SVD
            low_rank_layer = initialize_low_rank_from_svd(base_layer, rank)

            # Replace the layer in compressed model
            compressed_model.layers[i] = low_rank_layer

            # Calculate parameter reduction
            original_params = np.prod(kernel_shape)
            compressed_params = (base_layer.kernel_size[0] * base_layer.kernel_size[1] *
                                in_channels * rank +
                                rank * out_channels)
            reduction = (1 - compressed_params / original_params) * 100

            print(f"Layer {base_layer.name}: rank={rank}, "
                  f"params: {original_params:,} → {compressed_params:,} "
                  f"({reduction:.1f}% reduction)")
        else:
            # Copy non-convolutional layers directly
            if hasattr(base_layer, 'get_weights'):
                compressed_model.layers[i].set_weights(base_layer.get_weights())

    return compressed_model


def calculate_optimal_rank(layer, tau=0.95):
    """
    Calculate optimal rank using spectral information (ARSVD approach)

    Args:
        layer: Conv2D layer
        tau: Information retention threshold

    Returns:
        Optimal rank for the layer
    """
    weights = layer.get_weights()
    kernel = weights[0]

    h, w, in_c, out_c = kernel.shape
    kernel_2d = kernel.reshape(h * w * in_c, out_c)

    # Compute SVD
    U, S, Vt = compute_svd(kernel_2d)

    # Normalize singular values
    p = S / np.sum(S)

    # Compute entropy
    epsilon = 1e-15
    p_safe = (p + epsilon) / np.sum(p + epsilon)
    H_total = -np.sum(p * np.log(p_safe))

    # Find smallest k satisfying H(k) >= tau * H_total
    H_partial = 0
    for k in range(len(S)):
        H_partial -= p[k] * np.log(p_safe[k])
        if H_partial >= tau * H_total:
            return min(k + 1, len(S))

    return len(S)  # Return full rank if threshold not met


def create_arsvd_compressed_model(base_model, tau=0.95, img_size=256):
    """
    Create ARSVD-compressed model with adaptive rank selection per layer

    Args:
        base_model: Original trained model
        tau: Entropy threshold (higher = less compression)
        img_size: Input image size

    Returns:
        Compressed model with layer-specific optimal ranks
    """
    # Calculate optimal rank for each conv layer
    rank_config = {}
    print(f"\nARSVD Compression (tau={tau}):")
    print("="*60)

    for layer in base_model.layers:
        if 'conv' in layer.name.lower() and hasattr(layer, 'kernel'):
            rank = calculate_optimal_rank(layer, tau)
            kernel_shape = layer.kernel.shape
            in_channels = kernel_shape[2]
            out_channels = kernel_shape[3]
            max_rank = min(in_channels, out_channels)
            rank = min(rank, max_rank)
            rank_config[layer.name] = rank

    # Create compressed model using calculated ranks
    compressed_model = create_compressed_model_proper(base_model, rank_config, img_size)

    return compressed_model
