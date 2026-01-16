"""
Compression Experiment Runner
Compares ARSVD vs Standard SVD compression on U-Net model
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import argparse
import json
import csv
import numpy as np
import pandas as pd
from datetime import datetime
import tensorflow as tf
from tqdm import tqdm

from unet import build_unet
from metrics import dice_loss, dice_coef
from compression_utils import compress_conv2d_layer, calculate_compression_metrics
from train import load_dataset, tf_dataset
from tensorflow.keras.utils import CustomObjectScope


def parse_args():
    parser = argparse.ArgumentParser(
        description='Compression Experiment: ARSVD vs SVD on Brain Tumor Segmentation'
    )

    # Dataset parameters
    parser.add_argument('--data_root', type=str, required=True,
                        help='Path to dataset directory')
    parser.add_argument('--img_size', type=int, default=256,
                        help='Image size (H and W)')

    # Compression parameters
    parser.add_argument('--svd_ranks', type=str, default="100,150,200",
                        help='Comma-separated list of SVD ranks to test')
    parser.add_argument('--arsvd_taus', type=str, default="0.95,0.9,0.85,0.8",
                        help='Comma-separated list of ARSVD tau values to test')

    # Training/Evaluation parameters
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for evaluation')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs to train before compression')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')

    # Output parameters
    parser.add_argument('--out_dir', type=str, default='./compression_results',
                        help='Directory to save results')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to pre-trained model (if None, will train)')

    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    return parser.parse_args()


def create_compressed_model(base_model, compression_configs, img_size=256):
    """
    Create a compressed version of U-Net model

    Args:
        base_model: Original trained U-Net model
        compression_configs: List of compression config dicts for each layer
        img_size: Image size

    Returns:
        compressed_model: New model with compressed layers
    """
    # Build new model with same architecture
    inputs = tf.keras.Input(shape=(img_size, img_size, 3))

    # We'll copy the architecture but replace conv layers
    # This is a simplified approach - in practice, you'd need to rebuild
    # the model with low-rank approximation layers

    # For now, we'll directly modify the weights
    compressed_model = tf.keras.models.clone_model(base_model)

    # Apply compression to each conv layer
    config_idx = 0
    for i, layer in enumerate(compressed_model.layers):
        if 'conv' in layer.name.lower() and hasattr(layer, 'kernel'):
            if config_idx < len(compression_configs):
                comp_config = compression_configs[config_idx]

                # Get compressed weights
                original_layer = base_model.layers[i]
                weights = original_layer.get_weights()
                kernel = weights[0]

                # Apply compression
                if comp_config['method'] == 'arsvd':
                    from compression_utils import arsvd_compress

                    h, w, in_c, out_c = kernel.shape
                    kernel_2d = kernel.reshape(h * w * in_c, out_c)

                    compressed_kernel_2d, rank, _ = arsvd_compress(
                        kernel_2d, comp_config['tau']
                    )
                    compressed_kernel = compressed_kernel_2d.reshape(h, w, in_c, out_c)

                elif comp_config['method'] == 'svd':
                    from compression_utils import standard_svd_compress

                    h, w, in_c, out_c = kernel.shape
                    kernel_2d = kernel.reshape(h * w * in_c, out_c)

                    compressed_kernel_2d, _ = standard_svd_compress(
                        kernel_2d, comp_config['rank']
                    )
                    compressed_kernel = compressed_kernel_2d.reshape(h, w, in_c, out_c)

                # Set compressed weights
                if len(weights) > 1:
                    layer.set_weights([compressed_kernel, weights[1]])
                else:
                    layer.set_weights([compressed_kernel])

                config_idx += 1

    return compressed_model


def evaluate_model(model, test_dataset):
    """
    Evaluate model and return comprehensive metrics

    Args:
        model: Model to evaluate
        test_dataset: Test dataset

    Returns:
        metrics: Dictionary with evaluation metrics
    """
    # Metrics
    dice_scores = []
    iou_scores = []
    f1_scores = []
    accuracies = []

    for images, masks in tqdm(test_dataset, desc="Evaluating"):
        # Predict
        predictions = model.predict(images, verbose=0)

        # Calculate metrics for each sample in batch
        for i in range(images.shape[0]):
            pred_mask = predictions[i, :, :, 0]
            true_mask = masks[i, :, :, 0]

            # Convert to numpy if needed
            if hasattr(pred_mask, 'numpy'):
                pred_mask = pred_mask.numpy()
            if hasattr(true_mask, 'numpy'):
                true_mask = true_mask.numpy()

            # Binarize
            pred_binary = (pred_mask >= 0.5).astype(np.float32)
            true_binary = (true_mask >= 0.5).astype(np.float32)

            # Dice
            intersection = np.sum(pred_binary * true_binary)
            dice = (2. * intersection) / (np.sum(pred_binary) + np.sum(true_binary) + 1e-15)
            dice_scores.append(dice)

            # IoU (Jaccard)
            union = np.sum(pred_binary) + np.sum(true_binary) - intersection
            iou = intersection / (union + 1e-15)
            iou_scores.append(iou)

            # F1 Score (same as Dice for binary)
            f1_scores.append(dice)

            # Pixel accuracy
            accuracy = np.mean(pred_binary == true_binary)
            accuracies.append(accuracy)

    metrics = {
        'dice_mean': np.mean(dice_scores),
        'dice_std': np.std(dice_scores),
        'iou_mean': np.mean(iou_scores),
        'iou_std': np.std(iou_scores),
        'f1_mean': np.mean(f1_scores),
        'f1_std': np.std(f1_scores),
        'accuracy_mean': np.mean(accuracies),
        'accuracy_std': np.std(accuracies)
    }

    return metrics


def run_single_experiment(model, test_dataset, method, param_value, experiment_name):
    """
    Run a single compression experiment

    Args:
        model: Base trained model
        test_dataset: Test dataset
        method: 'arsvd' or 'svd'
        param_value: tau for arsvd, rank for svd
        experiment_name: Name for this experiment

    Returns:
        results: Dictionary with experiment results
    """
    print(f"\n{'='*60}")
    print(f"Running: {experiment_name}")
    print(f"{'='*60}")

    # Get compression configs for all conv layers
    compression_configs = []
    layer_count = 0

    for layer in model.layers:
        if 'conv' in layer.name.lower() and hasattr(layer, 'kernel'):
            if method == 'arsvd':
                config = {'method': 'arsvd', 'tau': param_value}
            else:  # svd
                config = {'method': 'svd', 'rank': param_value}

            compression_configs.append(config)
            layer_count += 1

    print(f"Compressing {layer_count} Conv2D layers...")

    # Create compressed model
    compressed_model = create_compressed_model(model, compression_configs)

    # Calculate compression statistics
    compression_summary = {
        'method': method,
        'param_value': param_value,
        'num_layers_compressed': layer_count
    }

    # Evaluate compressed model
    print("Evaluating compressed model...")
    metrics = evaluate_model(compressed_model, test_dataset)

    results = {
        'experiment_name': experiment_name,
        'method': method,
        'param_value': param_value,
        'compression': compression_summary,
        'metrics': metrics
    }

    print(f"\nResults:")
    print(f"  Dice: {metrics['dice_mean']:.4f} ± {metrics['dice_std']:.4f}")
    print(f"  IoU: {metrics['iou_mean']:.4f} ± {metrics['iou_std']:.4f}")
    print(f"  F1: {metrics['f1_mean']:.4f} ± {metrics['f1_std']:.4f}")
    print(f"  Accuracy: {metrics['accuracy_mean']:.4f} ± {metrics['accuracy_std']:.4f}")

    return results


def main():
    args = parse_args()

    # Parse ranks and taus
    svd_ranks = [int(r.strip()) for r in args.svd_ranks.split(',')]
    arsvd_taus = [float(t.strip()) for t in args.arsvd_taus.split(',')]

    print("\n" + "="*80)
    print("COMPRESSION EXPERIMENT: ARSVD vs SVD")
    print("="*80)
    print(f"SVD Ranks: {svd_ranks}")
    print(f"ARSVD Taus: {arsvd_taus}")
    print(f"Dataset: {args.data_root}")
    print(f"Output: {args.out_dir}")
    print("="*80)

    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)

    # Load dataset
    print("\nLoading dataset...")
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_dataset(args.data_root)
    print(f"Train: {len(train_x)}, Valid: {len(valid_x)}, Test: {len(test_x)}")

    test_dataset = tf_dataset(test_x, test_y, batch=args.batch_size)

    # Load or train base model
    if args.model_path and os.path.exists(args.model_path):
        print(f"\nLoading pre-trained model from {args.model_path}")
        with CustomObjectScope({"dice_coef": dice_coef, "dice_loss": dice_loss}):
            base_model = tf.keras.models.load_model(args.model_path)
    else:
        print("\nTraining base model...")
        H = args.img_size
        W = args.img_size

        base_model = build_unet((H, W, 3))
        base_model.compile(loss=dice_loss, optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
                          metrics=[dice_coef])

        train_dataset = tf_dataset(train_x, train_y, batch=args.batch_size)
        valid_dataset = tf_dataset(valid_x, valid_y, batch=args.batch_size)

        base_model.fit(
            train_dataset,
            epochs=args.epochs,
            validation_data=valid_dataset,
            verbose=1
        )

        # Save base model
        model_save_path = os.path.join(args.out_dir, 'base_model.h5')
        base_model.save(model_save_path)
        print(f"Base model saved to {model_save_path}")

    # Evaluate base model
    print("\nEvaluating base model...")
    base_metrics = evaluate_model(base_model, test_dataset)
    print(f"Base Model - Dice: {base_metrics['dice_mean']:.4f}, IoU: {base_metrics['iou_mean']:.4f}")

    # Run experiments
    all_results = []
    all_results.append({
        'experiment_name': 'base_model',
        'method': 'none',
        'param_value': None,
        'metrics': base_metrics
    })

    # SVD experiments
    for rank in svd_ranks:
        result = run_single_experiment(
            base_model, test_dataset, 'svd', rank, f'SVD_rank_{rank}'
        )
        all_results.append(result)

    # ARSVD experiments
    for tau in arsvd_taus:
        result = run_single_experiment(
            base_model, test_dataset, 'arsvd', tau, f'ARSVD_tau_{tau}'
        )
        all_results.append(result)

    # Save results
    results_file = os.path.join(args.out_dir, 'compression_results.json')
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    # Create CSV summary
    summary_data = []
    for result in all_results:
        row = {
            'experiment': result['experiment_name'],
            'method': result['method'],
            'param_value': result['param_value'],
            'dice': result['metrics']['dice_mean'],
            'dice_std': result['metrics']['dice_std'],
            'iou': result['metrics']['iou_mean'],
            'iou_std': result['metrics']['iou_std'],
            'f1': result['metrics']['f1_mean'],
            'f1_std': result['metrics']['f1_std'],
            'accuracy': result['metrics']['accuracy_mean'],
            'accuracy_std': result['metrics']['accuracy_std']
        }
        summary_data.append(row)

    summary_df = pd.DataFrame(summary_data)
    csv_file = os.path.join(args.out_dir, 'compression_summary.csv')
    summary_df.to_csv(csv_file, index=False)

    print(f"\n{'='*80}")
    print("EXPERIMENT COMPLETED!")
    print(f"{'='*80}")
    print(f"Results saved to {args.out_dir}")
    print(f"  - compression_results.json")
    print(f"  - compression_summary.csv")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
