"""
Pipeline script for Brain Tumor Segmentation
Trains the U-Net model and then evaluates it on test data
"""

import os
import sys
import argparse
import subprocess
from datetime import datetime

def run_command(cmd, description):
    """Run a shell command and print status"""
    print(f"\n{'='*60}")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {description}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"❌ Error in {description}")
        sys.exit(1)
    print(f"✅ {description} completed successfully")

def main():
    parser = argparse.ArgumentParser(description='Brain Tumor Segmentation Pipeline')

    # Data parameters
    parser.add_argument('--data_root', type=str, required=True,
                        help='Path to the dataset directory')
    parser.add_argument('--out_dir', type=str, default='./files',
                        help='Directory to save model and logs')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=500,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--img_size', type=int, default=256,
                        help='Image size (height and width)')

    # Callback parameters
    parser.add_argument('--patience', type=int, default=20,
                        help='Early stopping patience')
    parser.add_argument('--lr_patience', type=int, default=5,
                        help='ReduceLROnPlateau patience')

    # Other parameters
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)

    print("\n" + "="*60)
    print("BRAIN TUMOR SEGMENTATION PIPELINE")
    print("="*60)
    print(f"Dataset: {args.data_root}")
    print(f"Output: {args.out_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Learning Rate: {args.lr}")
    print(f"Image Size: {args.img_size}x{args.img_size}")
    print("="*60)

    # Step 1: Training
    train_cmd = (
        f"python UNET/train.py "
        f"--dataset_path {args.data_root} "
        f"--batch_size {args.batch_size} "
        f"--epochs {args.epochs} "
        f"--lr {args.lr} "
        f"--img_size {args.img_size} "
        f"--patience {args.patience} "
        f"--lr_patience {args.lr_patience} "
        f"--seed {args.seed} "
        f"--out_dir {args.out_dir}"
    )
    run_command(train_cmd, "TRAINING")

    # Step 2: Testing
    test_cmd = (
        f"python UNET/test.py "
        f"--dataset_path {args.data_root} "
        f"--img_size {args.img_size} "
        f"--model_path {args.out_dir}/model.h5 "
        f"--results_dir {args.out_dir}/results"
    )
    run_command(test_cmd, "TESTING AND EVALUATION")

    print("\n" + "="*60)
    print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"Model saved to: {args.out_dir}/model.h5")
    print(f"Results saved to: {args.out_dir}/results/")
    print(f"Metrics saved to: {args.out_dir}/score.csv")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
