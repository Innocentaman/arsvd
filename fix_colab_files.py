"""
Quick fix script to update compression_experiment.py for Google Colab
Run this in your Colab notebook to fix the function name issue
"""

import os

# Read the file
file_path = 'UNET/compression_experiment.py'

with open(file_path, 'r') as f:
    content = f.read()

# Fix the function call
content = content.replace(
    'compressed_model = create_compressed_model_proper(model, rank_config=param_value, img_size=img_size)',
    'compressed_model = compress_model_weights(model, rank_config=param_value, img_size=img_size)'
)

# Fix the import
content = content.replace(
    'from low_rank_layers import create_compressed_model_proper, create_arsvd_compressed_model',
    'from low_rank_layers import compress_model_weights, create_arsvd_compressed_model'
)

# Write back
with open(file_path, 'w') as f:
    f.write(content)

print("✅ Fixed compression_experiment.py")
print("Please restart your runtime and try again!")
