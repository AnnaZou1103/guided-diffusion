"""
Create a reference batch (.npz file) from ArtBench image directory for evaluation.

Usage:
    python create_reference_batch.py --data_dir ./artbench_images/impressionism --output reference_impressionism.npz --num_images 10000 --image_size 256
"""

import argparse
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import glob


def center_crop_arr(pil_image, image_size):
    """
    Center crop and resize image to target size.
    """
    # Resize to maintain aspect ratio
    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )
    
    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


def load_images_from_directory(data_dir, num_images=None, image_size=256, use_test_set=False, test_start_idx=5000):
    """
    Load images from directory and convert to numpy array.
    
    Args:
        data_dir: Directory containing images
        num_images: Maximum number of images to load (None for all)
        image_size: Target image size (assumes square images)
        use_test_set: If True, only load test set images (last 1000 images by default)
        test_start_idx: Starting index for test set (default: 5000, assuming first 5000 are training)
    
    Returns:
        numpy array of shape [N, H, W, 3] with dtype uint8, values in [0, 255]
    """
    # Find all image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(data_dir, ext)))
        image_files.extend(glob.glob(os.path.join(data_dir, '**', ext), recursive=True))
    
    image_files = sorted(image_files)
    
    if len(image_files) == 0:
        raise ValueError(f"No image files found in {data_dir}")
    
    print(f"Found {len(image_files)} images in {data_dir}")
    
    # If use_test_set is True, select only test set images
    if use_test_set:
        if len(image_files) < test_start_idx:
            raise ValueError(
                f"Not enough images for test set. Found {len(image_files)} images, "
                f"but test set starts at index {test_start_idx}"
            )
        image_files = image_files[test_start_idx:]
        print(f"Using test set: images from index {test_start_idx} onwards ({len(image_files)} images)")
    
    # Limit number of images if specified
    if num_images is not None:
        if len(image_files) < num_images:
            print(f"Warning: Only {len(image_files)} images available, but {num_images} requested")
        image_files = image_files[:num_images]
        print(f"Loading {len(image_files)} images...")
    else:
        print(f"Loading all {len(image_files)} images...")
    
    images = []
    failed = 0
    
    for img_path in tqdm(image_files, desc="Loading images"):
        try:
            with open(img_path, 'rb') as f:
                pil_image = Image.open(f)
                pil_image.load()
            
            # Convert to RGB if necessary
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            # Center crop and resize
            arr = center_crop_arr(pil_image, image_size)
            
            # Ensure shape is [H, W, 3]
            if len(arr.shape) == 2:
                arr = np.stack([arr] * 3, axis=-1)
            elif arr.shape[2] != 3:
                # Handle grayscale or other channel counts
                if arr.shape[2] == 1:
                    arr = np.repeat(arr, 3, axis=2)
                elif arr.shape[2] == 4:
                    # RGBA -> RGB
                    arr = arr[:, :, :3]
            
            images.append(arr)
        except Exception as e:
            print(f"Warning: Failed to load {img_path}: {e}")
            failed += 1
            continue
    
    if failed > 0:
        print(f"Warning: Failed to load {failed} images")
    
    if len(images) == 0:
        raise ValueError("No images were successfully loaded")
    
    # Stack into [N, H, W, 3] array
    images_array = np.stack(images, axis=0)
    
    # Ensure uint8 dtype and [0, 255] range
    if images_array.dtype != np.uint8:
        images_array = np.clip(images_array, 0, 255).astype(np.uint8)
    
    print(f"Loaded {len(images)} images, shape: {images_array.shape}")
    print(f"Value range: {images_array.min()} - {images_array.max()}")
    
    return images_array


def main():
    parser = argparse.ArgumentParser(
        description="Create reference batch .npz file from image directory"
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        required=True,
        help='Directory containing images (supports recursive search)'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output .npz file path'
    )
    parser.add_argument(
        '--num_images',
        type=int,
        default=1000,
        help='Number of images to include (default: 1000 for ArtBench test set)'
    )
    parser.add_argument(
        '--image_size',
        type=int,
        default=256,
        help='Target image size (default: 256)'
    )
    parser.add_argument(
        '--use_test_set',
        action='store_true',
        help='Only use test set images (assumes first 5000 are training, rest are test)'
    )
    parser.add_argument(
        '--test_start_idx',
        type=int,
        default=5000,
        help='Starting index for test set (default: 5000, assuming first 5000 are training)'
    )
    
    args = parser.parse_args()
    
    # Load images
    images = load_images_from_directory(
        args.data_dir,
        num_images=args.num_images,
        image_size=args.image_size,
        use_test_set=args.use_test_set,
        test_start_idx=args.test_start_idx
    )
    
    # Save as .npz file with 'arr_0' key 
    print(f"\nSaving to {args.output}...")
    np.savez_compressed(args.output, arr_0=images)


    print(f"Verifying saved file...")
    loaded = np.load(args.output)
    print(f"Saved file contains keys: {loaded.files}")
    print(f"Array shape: {loaded['arr_0'].shape}")
    print(f"Array dtype: {loaded['arr_0'].dtype}")
    print(f"Array value range: {loaded['arr_0'].min()} - {loaded['arr_0'].max()}")
    
    print(f"\nSuccessfully created reference batch: {args.output}")
    print(f"  Shape: {images.shape}")
    print(f"  Size: {os.path.getsize(args.output) / (1024**2):.2f} MB")


if __name__ == '__main__':
    main()

