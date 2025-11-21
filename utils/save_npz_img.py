import argparse
import os
import numpy as np
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default='./results/LSUN/samples_24x256x256x3.npz')
parser.add_argument('--output_dir', type=str, default='./results/LSUN/image/')
args = parser.parse_args()

data = np.load(args.input)

print(f"Field labels: {data.files}")  # ['arr_0', 'arr_1']

images = data['arr_0']

print(f"Number of images: {images.shape[0]}")
print(f"Image size: {images.shape[1]}x{images.shape[2]}")
print(f"Number of channels: {images.shape[3]}")

if 'arr_1' in data:
    labels = data['arr_1']

print(f"Value range: {images.min()} - {images.max()}")

if not os.path.exists(args.output_dir):
    os.mkdir(args.output_dir)

for i, img in enumerate(images):
    pil_image = Image.fromarray(img.astype(np.uint8))
    pil_image.save(f'{args.output_dir}image_{i:04d}.png')