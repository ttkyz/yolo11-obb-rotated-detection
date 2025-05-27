import os
import shutil
from tqdm import tqdm

def copy_image_multiple_times(source_image_path, destination_folder, num_copies=200):
    """
    将一张图片复制指定次数到目标文件夹。

    Args:
        source_image_path (str): 源图片的完整路径。
        destination_folder (str): 目标文件夹的路径。
        num_copies (int): 复制的次数。
    """
    if not os.path.exists(source_image_path):
        print(f"错误：源图片 '{source_image_path}' 不存在。")
        return

    # 创建目标文件夹（如果不存在）
    os.makedirs(destination_folder, exist_ok=True)

    # 获取源图片的文件名和扩展名
    base_name, extension = os.path.splitext(os.path.basename(source_image_path))

    print(f"开始复制图片 '{source_image_path}' 到 '{destination_folder}'，共 {num_copies} 次。")

    # 使用 tqdm 创建进度条
    for i in tqdm(range(1, num_copies + 1), desc="复制进度"):
        destination_image_name = f"{base_name}_{i}{extension}"
        destination_image_path = os.path.join(destination_folder, destination_image_name)
        try:
            shutil.copy2(source_image_path, destination_image_path)
        except Exception as e:
            print(f"复制文件 '{destination_image_name}' 时发生错误: {e}")

    print("复制完成！")

if __name__ == "__main__":
    # --- 请根据您的实际情况修改以下路径和文件名 ---
    # 源图片路径
    source_image = "lunchuan1.jpg"  # 替换为您的图片文件名，如果图片不在当前目录，请提供完整路径

    # 目标文件夹路径
    output_directory = "lunchuan"

    # 执行复制操作
    copy_image_multiple_times(source_image, output_directory, num_copies=50)