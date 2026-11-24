import os
import random
import numpy as np
from PIL import Image


def create_mold_stain_mask(
    img_size=256,
    seed=0,
    min_ratio=0.50,
    max_ratio=0.60,
    min_iters=500,
    max_iters=10000
):
    random.seed(seed)
    np.random.seed(seed)

    canvas = np.zeros((img_size, img_size), dtype=np.uint8)
    total_pixels = img_size * img_size
    target_pixels = max(1, int(total_pixels * random.uniform(min_ratio, max_ratio)))
    covered_pixels = 0

    directions = ((0, 1), (0, -1), (1, 0), (-1, 0))

    while covered_pixels < target_pixels:
        x = random.randint(0, img_size - 1)
        y = random.randint(0, img_size - 1)

        if canvas[x, y] == 0:
            canvas[x, y] = 1
            covered_pixels += 1

        iters = random.randint(min_iters, max_iters)
        for _ in range(iters):
            if covered_pixels >= target_pixels:
                break

            dx, dy = random.choice(directions)
            step_len = random.randint(0, 1)
            if step_len == 0:
                continue

            nx = x + dx * step_len
            ny = y + dy * step_len
            if 0 <= nx < img_size:
                x = nx
            if 0 <= ny < img_size:
                y = ny

            if canvas[x, y] == 0:
                canvas[x, y] = 1
                covered_pixels += 1

    mask_img = Image.fromarray(canvas * 255, mode='L')
    return mask_img, covered_pixels / total_pixels


if __name__ == '__main__':
    img_size = 256
    num_masks = 500
    min_ratio, max_ratio = 0.50, 0.60
    save_dir = './masks/50_60'

    os.makedirs(save_dir, exist_ok=True)

    for i in range(num_masks):
        img, ratio = create_mold_stain_mask(
            img_size=img_size,
            seed=i,
            min_ratio=min_ratio,
            max_ratio=max_ratio
        )
        img.save(os.path.join(save_dir, f'{i:06d}.png'))
        if (i + 1) % 100 == 0:
            print(f'Progress: {i + 1}/{num_masks} | Coverage: {ratio:.2%}')