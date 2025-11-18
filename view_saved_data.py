"""view_saved_data.py

Small utility to inspect a saved chunk (.npz) produced by data_collection.py.

Usage examples:

# Print keys and shapes and save first 3 frames as PNGs
python view_saved_data.py data/20251115-125117/episode_000/chunk_0000.npz --out preview --num-frames 3

# Display the first frame in a window (requires a display / matplotlib)
python view_saved_data.py path/to/chunk.npz --display

The script will try to handle both formats:
 - new: .npz with keys like 'images', 'pos', ... where 'images' is an array
   shaped (T, n_cameras, H, W, C) or (T, H, W, C) for single camera.
 - old: .npz with a single 'frames' entry that is a list/ndarray of dicts.

Note about security: loading object arrays requires allow_pickle=True. Only
use this on files you trust.
"""

import argparse
import numpy as np
import os
import sys


def try_import_imageio():
    try:
        import imageio
        return imageio
    except Exception:
        return None


def try_import_pil():
    try:
        from PIL import Image
        return Image
    except Exception:
        return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--npz_path", help="Path to the .npz chunk file", default="/home/snaak/CARLA_with_obstacles/PythonAPI/examples/truck-parking/data/20251116-135901/episode_001/chunk_0109.npz")
    p.add_argument("--out", default="preview_images", help="Output directory to save preview PNGs")
    p.add_argument("--num-frames", type=int, default=3, help="Number of time-steps to preview (per camera)")
    p.add_argument("--display", action="store_true", help="Attempt to display the first frame using matplotlib")
    p.add_argument("--allow-pickle", action="store_true", default=True, help="Allow loading object arrays with pickle (only for trusted files)")
    args = p.parse_args()

    if not os.path.exists(args.npz_path):
        print("File not found:", args.npz_path)
        sys.exit(1)

    # load with or without pickle depending on flag
    allow_pickle = bool(args.allow_pickle)
    # If user didn't enable allow_pickle we still attempt a safe load first
    try:
        npz = np.load(args.npz_path, allow_pickle=allow_pickle)
    except Exception as e:
        print("Initial load failed:", e)
        if not allow_pickle:
            print("Retrying with allow_pickle=True (only do this for trusted files)")
            npz = np.load(args.npz_path, allow_pickle=True)
        else:
            raise

    print("Loaded:", args.npz_path)
    print("Keys:", npz.files)

    # Helper to safely print shape
    def shape_or_info(x):
        try:
            return getattr(x, 'shape', type(x))
        except Exception:
            return type(x)

    # Print shapes/dtypes for keys
    for k in npz.files:
        try:
            v = npz[k]
            print(f"  {k}: shape={shape_or_info(v)}, dtype={getattr(v, 'dtype', type(v))}")
        except Exception as e:
            print(f"  {k}: error getting info: {e}")

    images = None

    # Prefer explicit 'images' key
    if 'images' in npz.files:
        images = npz['images']
    elif 'frames' in npz.files:
        frames = npz['frames']
        # frames could be an object array, list, or ndarray
        try:
            if isinstance(frames, np.ndarray) and frames.dtype == object:
                # try to extract images
                images = np.stack([f['images'] for f in frames], axis=0)
            else:
                # frames may be a list of dicts
                frames_list = list(frames)
                if len(frames_list) > 0 and isinstance(frames_list[0], dict):
                    images = np.stack([f['images'] for f in frames_list], axis=0)
        except Exception as e:
            print("Could not auto-extract images from 'frames':", e)
            images = None

    if images is None:
        print("No 'images' key found and couldn't extract images from 'frames'. Exiting.")
        return

    # Normalize images shape to (T, n_cams, H, W, C)
    try:
        images = np.asarray(images)
    except Exception as e:
        print("Failed to convert images to array:", e)
        return

    if images.ndim == 5:
        # (T, n_cams, H, W, C) as expected
        pass
    elif images.ndim == 4:
        # (T, H, W, C) -> add camera dim = 1
        images = images[:, None, ...]
    elif images.ndim == 3:
        # (H, W, C) -> single frame -> make T=1, n_cams=1
        images = images[None, None, ...]
    else:
        print(f"Unexpected images ndim={images.ndim}. Can't proceed.")
        return

    T, n_cams, H, W, C = images.shape
    print(f"Images shape interpreted as T={T}, n_cams={n_cams}, H={H}, W={W}, C={C}")

    # Basic sanity checks
    print("dtype:", images.dtype)
    if np.issubdtype(images.dtype, np.integer):
        vmin, vmax = images.min(), images.max()
        print(f"value range: {vmin} .. {vmax}")
    else:
        print("Non-integer image dtype; values may need scaling for display")

    # Prepare output directory
    os.makedirs(args.out, exist_ok=True)
    imageio = try_import_imageio()
    pil = None
    if imageio is None:
        pil = try_import_pil()
        if pil is None:
            print("Warning: neither imageio nor PIL available. Will attempt to use matplotlib only for display.")

    saved = []
    max_frames = min(args.num_frames, T)
    for t in range(max_frames):
        for c in range(n_cams):
            img = images[t, c]
            # ensure uint8 for saving if floats
            if np.issubdtype(img.dtype, np.floating):
                # normalize to 0..255
                im = img - img.min()
                if im.max() > 0:
                    im = (im / im.max() * 255).astype(np.uint8)
                else:
                    im = (im * 255).astype(np.uint8)
            else:
                im = img.astype(np.uint8)

            fname = os.path.join(args.out, f"frame{t:03d}_cam{c}.png")
            try:
                if imageio is not None:
                    imageio.imwrite(fname, im)
                elif pil is not None:
                    pil.fromarray(im).save(fname)
                else:
                    # fallback: use matplotlib
                    import matplotlib.pyplot as plt
                    plt.imsave(fname, im)
                saved.append(fname)
            except Exception as e:
                print(f"Failed to save {fname}: {e}")

    print(f"Saved {len(saved)} preview images to: {os.path.abspath(args.out)}")

    if args.display:
        try:
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, n_cams, figsize=(4*n_cams, 4))
            if n_cams == 1:
                axes = [axes]
            for c in range(n_cams):
                axes[c].imshow(images[0, c])
                axes[c].axis('off')
            plt.show()
        except Exception as e:
            print("Display failed (are you running headless?). Error:", e)


if __name__ == '__main__':
    main()
