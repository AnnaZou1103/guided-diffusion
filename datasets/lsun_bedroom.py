"""
Convert an LSUN lmdb database into a directory of images.
"""

import argparse
import io
import os

from PIL import Image
import lmdb
import numpy as np


def read_images(lmdb_path, image_size, max_images=None, start_index=0):
    env = lmdb.open(lmdb_path, map_size=1099511627776, max_readers=100, readonly=True)
    with env.begin(write=False) as transaction:
        cursor = transaction.cursor()
        count = 0
        skipped = 0
        for _, webp_data in cursor:
            # Skip images before start_index
            if skipped < start_index:
                skipped += 1
                continue
            
            # Stop if we've reached max_images
            if max_images is not None and count >= max_images:
                break
            
            img = Image.open(io.BytesIO(webp_data))
            width, height = img.size
            scale = image_size / min(width, height)
            img = img.resize(
                (int(round(scale * width)), int(round(scale * height))),
                resample=Image.BOX,
            )
            arr = np.array(img)
            h, w, _ = arr.shape
            h_off = (h - image_size) // 2
            w_off = (w - image_size) // 2
            arr = arr[h_off : h_off + image_size, w_off : w_off + image_size]
            yield arr
            count += 1


def dump_images(out_dir, images, prefix, start_index=0):
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    for i, img in enumerate(images):
        # Use start_index for numbering to maintain original indices
        Image.fromarray(img).save(os.path.join(out_dir, f"{prefix}_{start_index + i:07d}.png"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-size", help="new image size", type=int, default=256)
    parser.add_argument("--prefix", help="class name", type=str, default="bedroom")
    parser.add_argument("--max-images", help="maximum number of images to convert", type=int, default=None)
    parser.add_argument("--start-index", help="starting index for images (skip first N images)", type=int, default=0)
    parser.add_argument("lmdb_path", help="path to an LSUN lmdb database")
    parser.add_argument("out_dir", help="path to output directory")
    args = parser.parse_args()

    images = read_images(args.lmdb_path, args.image_size, max_images=args.max_images, start_index=args.start_index)
    dump_images(args.out_dir, images, args.prefix, start_index=args.start_index)


if __name__ == "__main__":
    main()
