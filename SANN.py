import argparse
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


def shape_aware_downscale(img, dst_w, dst_h, threshold=10, denoise=False):
    src_arr = np.asarray(img.convert("RGBA"))
    src_h, src_w, _ = src_arr.shape

    guidance_l = img.convert("L").filter(ImageFilter.MedianFilter(size=3))
    guidance_arr = np.asarray(guidance_l, dtype=np.int16)

    xs = (np.arange(dst_w) + 0.5) * (src_w / dst_w) - 0.5
    ys = (np.arange(dst_h) + 0.5) * (src_h / dst_h) - 0.5

    x0 = np.clip(np.floor(xs).astype(np.int32), 0, src_w - 2)
    y0 = np.clip(np.floor(ys).astype(np.int32), 0, src_h - 2)

    fx = (xs - x0)[np.newaxis, :]
    fy = (ys - y0)[:, np.newaxis]

    src_padded = np.pad(src_arr, ((1, 2), (1, 2), (0, 0)), mode="edge")
    guidance_padded = np.pad(guidance_arr, ((1, 2), (1, 2)), mode="edge")

    y_idx = y0[:, np.newaxis]
    x_idx = x0[np.newaxis, :]

    yp0, yp1, yp2, yp3 = y_idx, y_idx + 1, y_idx + 2, y_idx + 3
    xp0, xp1, xp2, xp3 = x_idx, x_idx + 1, x_idx + 2, x_idx + 3

    A_g = guidance_padded[yp1, xp1]
    B_g = guidance_padded[yp1, xp2]
    C_g = guidance_padded[yp2, xp1]
    D_g = guidance_padded[yp2, xp2]
    P01 = guidance_padded[yp0, xp1]
    P02 = guidance_padded[yp0, xp2]
    P10 = guidance_padded[yp1, xp0]
    P13 = guidance_padded[yp1, xp3]
    P20 = guidance_padded[yp2, xp0]
    P23 = guidance_padded[yp2, xp3]
    P31 = guidance_padded[yp3, xp1]
    P32 = guidance_padded[yp3, xp2]

    w1 = 4 * np.abs(B_g - C_g) + np.abs(P01 - P10) + np.abs(P23 - P32)
    w2 = 4 * np.abs(A_g - D_g) + np.abs(P02 - P13) + np.abs(P20 - P31)

    max_dist = np.maximum(w1, w2)
    min_dist = np.minimum(w1, w2)
    edge_detected = (max_dist - min_dist) > threshold

    if denoise:
        A_c, B_c, C_c, D_c = A_g[..., np.newaxis], B_g[..., np.newaxis], C_g[..., np.newaxis], D_g[..., np.newaxis]
    else:
        src_rgb = src_padded[..., :3].astype(np.int16)
        A_c = src_rgb[yp1, xp1]
        B_c = src_rgb[yp1, xp2]
        C_c = src_rgb[yp2, xp1]
        D_c = src_rgb[yp2, xp2]

    def c_dist(c1, c2):
        if c1.ndim == 2:
            return np.abs(c1 - c2)
        return np.sum(np.abs(c1 - c2), axis=-1)

    core_variance = c_dist(A_c, B_c) + c_dist(C_c, D_c) + c_dist(A_c, C_c) + c_dist(B_c, D_c)
    is_high_frequency = core_variance > (threshold * 4)

    diag1 = (w1 < w2) & edge_detected & ~is_high_frequency
    diag2 = (w2 < w1) & edge_detected & ~is_high_frequency

    mask_A = (fx < 0.5) & (fy < 0.5)
    mask_B = (fx >= 0.5) & (fy < 0.5)
    mask_C = (fx < 0.5) & (fy >= 0.5)
    mask_D = (fx >= 0.5) & (fy >= 0.5)

    choice = np.zeros((dst_h, dst_w), dtype=np.uint8)
    choice[mask_B] = 1
    choice[mask_C] = 2
    choice[mask_D] = 3

    cond_diag1_A = diag1 & mask_A & (fx + fy > 0.5)
    choice[cond_diag1_A & (fx >= fy)] = 1
    choice[cond_diag1_A & (fy > fx)] = 2

    cond_diag1_D = diag1 & mask_D & (fx + fy < 1.5)
    choice[cond_diag1_D & (fx >= fy)] = 1
    choice[cond_diag1_D & (fy > fx)] = 2

    cond_diag2_B = diag2 & mask_B & (fx - fy < 0.5)
    choice[cond_diag2_B & (fx + fy < 1.0)] = 0
    choice[cond_diag2_B & (fx + fy >= 1.0)] = 3

    cond_diag2_C = diag2 & mask_C & (fy - fx < 0.5)
    choice[cond_diag2_C & (fx - fy < 0.5)] = 0
    choice[cond_diag2_C & (fx + fy >= 1.0)] = 3

    is_fine_grid = core_variance > (threshold * 8)
    if np.any(is_fine_grid):
        if A_c.ndim == 2:
            luma_A, luma_B, luma_C, luma_D = A_c, B_c, C_c, D_c
        else:
            luma_A = A_c.sum(axis=-1)
            luma_B = B_c.sum(axis=-1)
            luma_C = C_c.sum(axis=-1)
            luma_D = D_c.sum(axis=-1)
        darkest_stack = np.stack([luma_A, luma_B, luma_C, luma_D], axis=0)
        choice = np.where(is_fine_grid, np.argmin(darkest_stack, axis=0).astype(np.uint8), choice)

    A = src_padded[yp1, xp1]
    B = src_padded[yp1, xp2]
    C = src_padded[yp2, xp1]
    D = src_padded[yp2, xp2]

    out_arr = np.zeros((dst_h, dst_w, 4), dtype=np.uint8)
    m0 = choice == 0
    m1 = choice == 1
    m2 = choice == 2
    m3 = choice == 3

    out_arr[m0] = A[m0]
    out_arr[m1] = B[m1]
    out_arr[m2] = C[m2]
    out_arr[m3] = D[m3]

    return Image.fromarray(out_arr)


def shape_aware_upscale(img, dst_w, dst_h):
    return img.convert("RGBA").resize((dst_w, dst_h), resample=Image.Resampling.NEAREST)


def resize_sann(img, dst_w, dst_h, threshold=10, denoise=False):
    src_w, src_h = img.size
    if dst_w >= src_w and dst_h >= src_h:
        return shape_aware_upscale(img, dst_w, dst_h)
    return shape_aware_downscale(img, dst_w, dst_h, threshold=threshold, denoise=denoise)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("input")
    p.add_argument("output")
    p.add_argument("width", type=int)
    p.add_argument("height", type=int)
    p.add_argument("--threshold", type=int, default=10)
    p.add_argument("--denoise", action="store_true")
    args = p.parse_args()

    img = Image.open(args.input).convert("RGBA")

    t0 = time.perf_counter()

    out = resize_sann(
        img,
        args.width,
        args.height,
        threshold=args.threshold,
        denoise=args.denoise,
    )

    print(f"{(time.perf_counter() - t0) * 1000:.2f}ms")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    if args.output.lower().endswith((".jpg", ".jpeg")):
        out = out.convert("RGB")
        out.save(args.output, quality=100, subsampling=0)
    else:
        out.save(args.output)


if __name__ == "__main__":
    main()