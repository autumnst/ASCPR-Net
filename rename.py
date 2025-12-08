import os

def rename_masks_sequentially(mask_dir, ext='png'):
    """
    将掩码文件夹中的文件按顺序重命名为 00000.xxx、00001.xxx...
    
    参数:
        mask_dir: 掩码文件夹路径
        ext: 掩码文件扩展名（默认png）
    """
    # 1. 获取文件夹中所有指定格式的文件，并按原始顺序排序
    mask_files = []
    for filename in os.listdir(mask_dir):
        # 过滤非目标格式文件
        if filename.lower().endswith(f'.{ext}'):
            # 保存完整路径
            mask_files.append(os.path.join(mask_dir, filename))
    
    # 检查是否有文件
    if not mask_files:
        print(f"错误：在 {mask_dir} 中未找到 .{ext} 格式的文件！")
        return
    
    print(f"找到 {len(mask_files)} 个 .{ext} 格式的掩码文件，准备重命名...")
    
    # 2. 按原始文件名排序（确保顺序一致，如需按修改时间排序可调整此处）
    # 若需要按数字排序（如原文件是1.png、2.png），可改为：
    # mask_files.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    mask_files.sort()  # 按原始文件名字符串排序（默认）
    
    # 3. 依次重命名为 00000.ext、00001.ext...
    renamed_count = 0
    for i, old_path in enumerate(mask_files):
        # 生成5位数字文件名（00000, 00001...）
        new_filename = f"{i:05d}.{ext}"
        new_path = os.path.join(mask_dir, new_filename)
        
        # 跳过已存在的文件（避免覆盖）
        if os.path.exists(new_path):
            print(f"跳过：{new_filename} 已存在")
            continue
        
        # 执行重命名
        os.rename(old_path, new_path)
        renamed_count += 1
        print(f"重命名：{os.path.basename(old_path)} → {new_filename}")
    
    print(f"\n重命名完成：成功 {renamed_count} 个，共 {len(mask_files)} 个文件")

if __name__ == "__main__":
    # --------------------------
    # 请修改为你的掩码文件夹路径
    # --------------------------
    MASK_DIR = r"/data/lqq/tql/APLRL_data/test_dh/masks/"  # 掩码文件夹路径
    MASK_EXT = "png"  # 掩码文件扩展名（无需带点）
    
    # 检查路径是否存在
    if not os.path.isdir(MASK_DIR):
        print(f"错误：掩码文件夹 '{MASK_DIR}' 不存在！")
    else:
        rename_masks_sequentially(MASK_DIR, MASK_EXT)