"""
Compression utilities for Neural Network compression
Implements ARSVD and standard SVD compression methods
"""

import numpy as np
import tensorflow as tf
from scipy.linalg import svd


def compute_svd(matrix):
    """
    Compute SVD of a matrix

    Args:
        matrix: Input matrix of shape (m, n)

    Returns:
        U, S, Vt: SVD components
    """
    return svd(matrix, full_matrices=False)


def standard_svd_compress(W, rank):
    """
    Standard SVD compression with fixed rank

    Args:
        W: Weight matrix of shape (m, n)
        rank: Target rank for compression

    Returns:
        compressed_matrix: Approximated matrix
        actual_rank: Actual rank used (min of rank and matrix rank)
    """
    m, n = W.shape
    max_rank = min(m, n)
    k = min(rank, max_rank)

    U, S, Vt = compute_svd(W)

    # Truncate to rank k
    U_k = U[:, :k]
    S_k = S[:k]
    Vt_k = Vt[:k, :]

    # Reconstruct
    W_compressed = U_k @ np.diag(S_k) @ Vt_k

    return W_compressed, k


def arsvd_compress(W, tau):
    """
    Adaptive-Rank SVD compression using spectral entropy

    Args:
        W: Weight matrix of shape (m, n)
        tau: Entropy threshold (0 < tau <= 1)

    Returns:
        compressed_matrix: Approximated matrix
        rank: Selected rank based on entropy
        entropy_info: Dictionary with entropy details
    """
    U, S, Vt = compute_svd(W)
    r = len(S)  # Total number of singular values

    # Normalize singular values to get probability distribution
    total = np.sum(S)
    if total == 0:
        # Handle edge case
        return np.zeros_like(W), 0, {'total_entropy': 0, 'partial_entropy': 0}

    p = S / total

    # Compute total spectral entropy
    # Avoid log(0) by adding small epsilon
    epsilon = 1e-15
    p_safe = p + epsilon
    p_safe = p_safe / np.sum(p_safe)  # Renormalize

    H_total = -np.sum(p * np.log(p_safe))

    # Find smallest k such that H(k) >= tau * H_total
    H_partial = 0
    k = 0

    for i in range(r):
        # Add contribution of i-th singular value
        H_partial -= p[i] * np.log(p_safe[i])

        if H_partial >= tau * H_total:
            k = i + 1
            break
    else:
        # If threshold not met, use all singular values
        k = r

    # Ensure at least rank 1
    k = max(1, k)

    # Truncate to rank k
    U_k = U[:, :k]
    S_k = S[:k]
    Vt_k = Vt[:k, :]

    # Reconstruct
    W_compressed = U_k @ np.diag(S_k) @ Vt_k

    entropy_info = {
        'total_entropy': H_total,
        'partial_entropy': H_partial,
        'threshold': tau * H_total,
        'fraction_retained': H_partial / H_total if H_total > 0 else 0
    }

    return W_compressed, k, entropy_info


def compress_conv2d_layer(layer, method='arsvd', rank=None, tau=None):
    """
    Compress a Conv2D layer using SVD or ARSVD

    Args:
        layer: Keras Conv2D layer
        method: 'arsvd' or 'svd'
        rank: Rank for SVD compression
        tau: Tau threshold for ARSVD compression

    Returns:
        compressed_weights: Dictionary with compressed weight matrices
        compression_info: Dictionary with compression details
    """
    weights = layer.get_weights()

    if len(weights) == 0:
        return None, {'error': 'No weights to compress'}

    kernel = weights[0]  # Shape: (h, w, in_channels, out_channels)
    bias = weights[1] if len(weights) > 1 else None

    h, w, in_c, out_c = kernel.shape

    # Reshape kernel to 2D matrix: (in_channels * h * w, out_channels)
    kernel_2d = kernel.reshape(h * w * in_c, out_c)

    # Compress using specified method
    if method.lower() == 'arsvd':
        if tau is None:
            raise ValueError("tau must be specified for ARSVD compression")

        compressed_kernel_2d, rank_used, entropy_info = arsvd_compress(kernel_2d, tau)
        compression_info = {
            'method': 'arsvd',
            'tau': tau,
            'rank': rank_used,
            'original_shape': kernel.shape,
            'matrix_shape': kernel_2d.shape,
            'entropy_info': entropy_info
        }
    elif method.lower() == 'svd':
        if rank is None:
            raise ValueError("rank must be specified for SVD compression")

        compressed_kernel_2d, rank_used = standard_svd_compress(kernel_2d, rank)
        compression_info = {
            'method': 'svd',
            'rank': rank_used,
            'original_shape': kernel.shape,
            'matrix_shape': kernel_2d.shape
        }
    else:
        raise ValueError(f"Unknown compression method: {method}")

    # Reshape back to kernel shape
    compressed_kernel = compressed_kernel_2d.reshape(h, w, in_c, out_c)

    # Calculate parameter reduction
    original_params = np.prod(kernel.shape)
    compressed_params = rank_used * (h * w * in_c + out_c)
    reduction_pct = (1 - compressed_params / original_params) * 100

    compression_info['original_params'] = original_params
    compression_info['compressed_params'] = compressed_params
    compression_info['reduction_pct'] = reduction_pct

    compressed_weights = {
        'kernel': compressed_kernel,
        'bias': bias
    }

    return compressed_weights, compression_info


def calculate_compression_metrics(layer_infos):
    """
    Calculate overall compression metrics across all layers

    Args:
        layer_infos: List of compression info dictionaries

    Returns:
        summary: Dictionary with overall compression statistics
    """
    total_original = sum(info['original_params'] for info in layer_infos)
    total_compressed = sum(info['compressed_params'] for info in layer_infos)

    summary = {
        'total_original_params': total_original,
        'total_compressed_params': total_compressed,
        'total_reduction_pct': (1 - total_compressed / total_original) * 100,
        'num_layers': len(layer_infos),
        'layer_details': layer_infos
    }

    return summary
