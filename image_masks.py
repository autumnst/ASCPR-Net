import os
import shutil
from pathlib import Path

def rename_masks_to_match_images(
    image_root, 
    mask_root, 
    image_extensions={'jpg', 'jpeg', 'png', 'bmp'},
    mask_extensions={'png', 'bmp', 'jpg'},
    mask_prefix=""  # 掩码文件名前缀（如掩码是"mask_001.png"，则prefix="mask_"）
):
    """
    根据图像文件重命名掩码文件，支持目录结构不同、掩码带前缀的情况
    
    参数:
        mask_prefix: 掩码文件名的固定前缀（如掩码是"mask_001.png"，则设为"mask_"）
    """
    # 1. 收集所有图像文件的"文件名前缀"（不带路径和扩展名）
    # 键：图像文件名前缀（如"001"），值：图像完整路径（用于验证）
    image_name_prefixes = set()
    for dirpath, _, filenames in os.walk(image_root):
        for filename in filenames:
            name, ext = os.path.splitext(filename)
            ext = ext.lower().lstrip('.')
            if ext in image_extensions:
                image_name_prefixes.add(name)  # 只记录文件名前缀（如"001"）
    
    print(f"已发现 {len(image_name_prefixes)} 个图像文件名前缀（含子目录）")
    if not image_name_prefixes:
        print("错误：未找到任何图像文件，请检查图像目录和格式！")
        return

    # 2. 遍历掩码文件，按图像文件名前缀重命名（忽略目录结构差异）
    renamed_count = 0
    skipped_count = 0
    for dirpath, _, filenames in os.walk(mask_root):
        for filename in filenames:
            # 过滤非掩码文件
            mask_name, mask_ext = os.path.splitext(filename)
            mask_ext = mask_ext.lower().lstrip('.')
            if mask_ext not in mask_extensions:
                continue
            
            # 移除掩码前缀（如"mask_001" → "001"）
            if mask_name.startswith(mask_prefix):
                clean_mask_name = mask_name[len(mask_prefix):]
            else:
                clean_mask_name = mask_name  # 无前缀则直接使用原名
            
            # 检查掩码是否有对应的图像（仅通过文件名前缀匹配，忽略路径）
            if clean_mask_name not in image_name_prefixes:
                skipped_count += 1
                # print(f"跳过：掩码 '{filename}' 无对应图像（前缀不匹配）")
                continue
            
            # 生成新文件名（与图像前缀相同，保留掩码扩展名）
            new_mask_filename = f"{clean_mask_name}.{mask_ext}"
            old_mask_path = os.path.join(dirpath, filename)
            new_mask_path = os.path.join(dirpath, new_mask_filename)
            
            # 重命名（避免覆盖已存在的文件）
            if old_mask_path != new_mask_path:
                if not os.path.exists(new_mask_path):
                    os.rename(old_mask_path, new_mask_path)
                    renamed_count += 1
                    print(f"重命名：{filename} → {new_mask_filename}")
                else:
                    skipped_count += 1
                    print(f"跳过：'{new_mask_filename}' 已存在")
    
    print(f"\n重命名完成：成功 {renamed_count} 个，跳过 {skipped_count} 个")

if __name__ == "__main__":
    # --------------------------
    # 请根据实际情况修改以下参数
    # --------------------------
    IMAGE_ROOT = r"/data/lqq/tql/APLRL/results/APLRL_dh_256/corrupt/"  # 图像根目录
    MASK_ROOT = r"/data/lqq/tql/APLRL_data/test_dh/masks/"             # 掩码根目录
    MASK_PREFIX = ""  # 若掩码文件名有前缀（如"mask_001.png"），则设为"mask_"；无则留空
    
    # 检查路径
    if not os.path.isdir(IMAGE_ROOT):
        print(f"错误：图像目录 '{IMAGE_ROOT}' 不存在！")
    elif not os.path.isdir(MASK_ROOT):
        print(f"错误：掩码目录 '{MASK_ROOT}' 不存在！")
    else:
        rename_masks_to_match_images(
            image_root=IMAGE_ROOT,
            mask_root=MASK_ROOT,
            image_extensions={'jpg', 'png'},
            mask_extensions={'png'},
            mask_prefix=MASK_PREFIX  # 关键参数：根据掩码文件名前缀修改
        )