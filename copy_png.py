import os
import shutil


def copy_first_and_last_500_pngs(source_folder, target_folder):
    """复制源文件夹中前500个和后500个PNG图片到目标文件夹."""
    # 检查源文件夹是否存在
    if not os.path.exists(source_folder):
        print(f"错误：源文件夹 '{source_folder}' 不存在")
        return

    # 创建目标文件夹（如果不存在）
    os.makedirs(target_folder, exist_ok=True)

    # 获取所有PNG文件
    png_files = [
        f
        for f in os.listdir(source_folder)
        if f.lower().endswith(".png") and os.path.isfile(os.path.join(source_folder, f))
    ]

    # 排序文件（确保顺序一致）
    png_files.sort()

    # 计算前500和后500的文件
    total_files = len(png_files)
    first_500 = png_files[:500]
    last_500 = png_files[-500:] if total_files > 500 else png_files

    # 合并文件列表并去重
    files_to_copy = list(set(first_500 + last_500))

    # 复制文件
    for file in files_to_copy:
        source_path = os.path.join(source_folder, file)
        target_path = os.path.join(target_folder, file)
        shutil.copy2(source_path, target_path)

    print(f"成功复制 {len(files_to_copy)} 个PNG文件到 '{target_folder}'")


# 示例用法
if __name__ == "__main__":
    source = "/home/jzyh/xbzl/traffic_light/data_v1_filtered/images/val"  # 替换为实际的源文件夹路径
    target = "/home/jzyh/xbzl/traffic_light/data_v1_filtered/images/val1"  # 替换为实际的目标文件夹路径
    copy_first_and_last_500_pngs(source, target)
