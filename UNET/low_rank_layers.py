"""
Low-Rank Convolutional Layers for Neural Network Compression

Implements low-rank approximation of Conv2D layer weights using SVD.
This approach compresses weights in-place without changing the model architecture.
"""

import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D
import numpy as np
from compression_utils import compute_svd


class LowRankApproxConv2D(Layer):
    """
    Conv2D layer with low-rank approximate weights

    This layer behaves like a regular Conv2D but internally stores and computes
    using a low-rank factorization of the weights.

    Mathematical formulation:
    Original: W ∈ R^(k×k×c_in×c_out)
    Low-rank: W ≈ U @ diag(S) @ V^T
    where U ∈ R^(k×k×c_in×r), S ∈ R^r, V^T ∈ R^(r×c_out)

    Benefits:
    - Reduces parameters from k×k×c_in×c_out to k×k×c_in×r + r×c_out
    - Drop-in replacement for Conv2D layers
    - Can be fine-tuned after compression
    """

    def __init__(self, filters, kernel_size, rank, strides=(1, 1),
                 padding='valid', use_bias=True, **kwargs):
        super(LowRankApproxConv2D, self).__init__(**kwargs)

        self.filters = filters
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.rank = rank
        self.strides = strides if isinstance(strides, tuple) else (strides, strides)
        self.padding = padding
        self.use_bias = use_bias

        # First conv: Spatial convolution with rank output channels
        self.conv1 = Conv2D(
            filters=rank,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            use_bias=False,
            name=f'{self.name}_spatial'
        )

        # Second conv: 1x1 convolution to expand to filter count
        self.conv2 = Conv2D(
            filters=filters,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding='same',
            use_bias=use_bias,
            name=f'{self.name}_pointwise'
        )

    def build(self, input_shape):
        # Build conv1 with input shape
        self.conv1.build(input_shape)

        # Build conv2 with output shape from conv1
        conv1_output_shape = self.conv1.compute_output_shape(input_shape)
        self.conv2.build(conv1_output_shape)

        super(LowRankApproxConv2D, self).build(input_shape)

    def call(self, inputs, training=None):
        """Forward pass through the two-layer decomposition"""
        x = self.conv1(inputs)
        x = self.conv2(x)
        return x

    def compute_output_shape(self, input_shape):
        """Compute output shape (same as original Conv2D would produce)"""
        return self.conv2.compute_output_shape(
            self.conv1.compute_output_shape(input_shape)
        )

    def get_config(self):
        config = super(LowRankApproxConv2D, self).get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'rank': self.rank,
            'strides': self.strides,
            'padding': self.padding,
            'use_bias': self.use_bias,
        })
        return config


def initialize_low_rank_from_svd(original_layer, rank, input_shape=None):
    """
    Create a low-rank approximation layer using SVD of original Conv2D weights

    Args:
        original_layer: Original Conv2D layer to compress
        rank: Target rank for decomposition
        input_shape: Input shape for building the layer (if None, uses dummy shape)

    Returns:
        LowRankApproxConv2D layer with SVD-initialized weights
    """
    # Get original weights
    weights = original_layer.get_weights()
    kernel = weights[0]  # Shape: (h, w, in_c, out_c)
    bias = weights[1] if len(weights) > 1 else None

    h, w, in_c, out_c = kernel.shape

    # Validate rank
    max_rank = min(in_c, out_c)
    rank = min(max(rank, 1), max_rank)

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

    # Create low-rank approximation layer with original layer's configuration
    low_rank_layer = LowRankApproxConv2D(
        filters=out_c,
        kernel_size=(h, w),
        rank=rank,
        strides=original_layer.strides,
        padding=original_layer.padding,
        use_bias=original_layer.use_bias,
        name=f'{original_layer.name}_lowrank'
    )

    # Build the layer with appropriate input shape
    if input_shape is None:
        # Use dummy shape - will be rebuilt when connected to actual model
        input_shape = (None, None, None, in_c)

    low_rank_layer.build(input_shape)

    # Set the SVD-initialized weights
    low_rank_layer.conv1.set_weights([W1_kernel])
    low_rank_layer.conv2.set_weights([W2_kernel] + ([bias] if bias is not None else []))

    return low_rank_layer


def compress_model_weights(base_model, rank_config, img_size=256):
    """
    Create compressed model by applying low-rank SVD to Conv2D layer weights

    Note: This modifies weights in-place rather than changing architecture.
    The weights are reshaped to low-rank approximation but kept in original shape.

    Args:
        base_model: Original trained model
        rank_config: Either a single rank (int) or dict mapping layer names to ranks
        img_size: Input image size

    Returns:
        Compressed model with low-rank approximate weights
    """
    from unet import build_unet

    # Build fresh model
    compressed_model = build_unet((img_size, img_size, 3))

    print(f"\nLow-Rank Compression:")
    print("="*60)

    # Copy and compress weights
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
            rank = min(max(rank, 1), max_rank)

            # Get original weights
            weights = base_layer.get_weights()
            kernel = weights[0]  # Shape: (h, w, in_c, out_c)
            bias = weights[1] if len(weights) > 1 else None

            h, w = kernel_shape[0], kernel_shape[1]

            # Reshape kernel to 2D for SVD
            kernel_2d = kernel.reshape(h * w * in_channels, out_channels)

            # Compute SVD
            U, S, Vt = compute_svd(kernel_2d)

            # Truncate to target rank
            U_r = U[:, :rank]
            S_r = S[:rank]
            Vt_r = Vt[:rank, :]

            # Reconstruct low-rank approximation
            kernel_compressed_2d = U_r @ np.diag(S_r) @ Vt_r

            # Reshape back to original kernel shape
            kernel_compressed = kernel_compressed_2d.reshape(h, w, in_channels, out_channels)

            # Set compressed weights to corresponding layer in new model
            if bias is not None:
                compressed_model.layers[i].set_weights([kernel_compressed, bias])
            else:
                compressed_model.layers[i].set_weights([kernel_compressed])

            # Calculate metrics
            original_params = np.prod(kernel_shape)
            # Theoretical compressed params (if we used actual low-rank layers)
            theoretical_compressed_params = h * w * in_channels * rank + rank * out_channels
            reduction = (1 - theoretical_compressed_params / original_params) * 100

            # Calculate approximation error
            approximation_error = np.linalg.norm(kernel_2d - kernel_compressed_2d, 'fro') / np.linalg.norm(kernel_2d, 'fro')

            print(f"Layer {base_layer.name}: rank={rank}/{max_rank}, "
                  f"theoretical params: {original_params:,} → {theoretical_compressed_params:,} "
                  f"({reduction:.1f}% reduction), "
                  f"approx error: {approximation_error:.4f}")
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
    compressed_model = compress_model_weights(base_model, rank_config, img_size)

    return compressed_model
