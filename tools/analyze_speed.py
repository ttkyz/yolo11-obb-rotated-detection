from ultralytics import YOLO
import os
import glob
import time
import torch # 用于检查CUDA可用性

def efficient_yolo_inference_on_folder(
    model_path: str,
    image_folder_path: str,
    batch_size: int = 16, # 默认批量大小，根据您的GPU显存调整
    conf_threshold: float = 0.25, # 检测置信度阈值
    img_size: int = 640, # 模型输入图像尺寸，对于大图可能需要调整以平衡速度和精度
    save_plotted_images: bool = True, # 是否保存绘制了检测框的图片
    save_labels_txt: bool = True, # 是否保存YOLO格式的txt标签文件
    output_project_name: str = "yolo_large_image_predictions" # 输出项目名称
):


    # 1. 环境设置与模型加载
    print(f"正在加载模型: {model_path}...")
    try:
        model = YOLO(model_path)
        print("模型加载成功。")
    except Exception as e:
        print(f"错误：无法加载模型。请检查模型路径和文件格式。错误信息: {e}")
        return

    # 检查CUDA (GPU) 可用性
    if torch.cuda.is_available():
        print(f"检测到CUDA可用。模型将使用GPU进行推理。当前设备: {torch.cuda.get_device_name(0)}")
    else:
        print("未检测到CUDA")

    print(f"正在检查图像文件夹: {image_folder_path}...")
    if not os.path.isdir(image_folder_path):
        print(f"错误：未在 '{image_folder_path}' 找到指定的图像文件夹。请检查路径。")
        return

    image_files = glob.glob(os.path.join(image_folder_path, "*.jpg")) + \
                  glob.glob(os.path.join(image_folder_path, "*.png")) + \
                  glob.glob(os.path.join(image_folder_path, "*.jpeg")) + \
                  glob.glob(os.path.join(image_folder_path, "*.bmp"))

    if not image_files:
        warmup_source = "https://ultralytics.com/images/bus.jpg"
    else:
        warmup_source = image_files[:min(len(image_files), 5)]

    print(f"找到 {len(image_files)} 张图像文件。")

    print("\n--- 启动模型预热阶段 ---")
    try:
        for i in range(3):
            _ = model.predict(source=warmup_source, verbose=False, conf=0.01, imgsz=img_size)
            print(f"  预热运行 {i+1}/3 完成。")
        print("模型预热成功完成。")
    except Exception as e:
        print(f"错误信息: {e}")


    output_run_name = "run_batch_inference_" + time.strftime("%Y%m%d-%H%M%S")
    results_output_dir = os.path.join("runs", "detect", output_project_name, output_run_name)

    print(f"\n--- 开始对文件夹进行连续批量推理：'{image_folder_path}' ---")
    print(f"配置的批量大小：{batch_size}，置信度阈值：{conf_threshold}，输入尺寸：{img_size}")


    total_inference_time = 0
    processed_images_count = 0

    try:
        results_generator = model.predict(
            source=image_folder_path,
            batch=batch_size,
            stream=True,
            conf=conf_threshold,
            imgsz=img_size,
            save=save_plotted_images,
            save_txt=save_labels_txt,
            project=output_project_name,
            name=output_run_name,
            verbose=False
        )
        for i, result in enumerate(results_generator):
            processed_images_count += 1
            total_inference_time += result.speed['inference']

            if processed_images_count % 10 == 0:
                print(f"  已处理 {processed_images_count} 张图像...")

    except Exception as e:
        return

    print(f"\n--- 推理完成 ---")
    print(f"已使用{model_path}完成对 {processed_images_count} 张图像的推理。")
    print(f"总推理时间（不含预热）：{total_inference_time:.2f} ms。")
    if processed_images_count > 0:
        print(f"平均每张图像时间：{total_inference_time / processed_images_count:.4f} ms。")


# --- 如何使用 ---
if __name__ == "__main__":

    MODEL_PATH = ["ckpts/yolo11n-obb.pt","ckpts/yolo11s-obb.pt","ckpts/yolo11m-obb.pt","ckpts/yolo11l-obb.pt","ckpts/yolo11x-obb.pt"]

    IMAGE_FOLDER = "lunchuan" 

    BATCH_SIZE_CONFIG = 8
    CONF_THRESHOLD_CONFIG = 0.5
    IMG_SIZE_CONFIG = 1280
    for i in range(5):
        efficient_yolo_inference_on_folder(
            model_path=MODEL_PATH[i],
            image_folder_path=IMAGE_FOLDER,
            batch_size=BATCH_SIZE_CONFIG,
            conf_threshold=CONF_THRESHOLD_CONFIG,
            img_size=IMG_SIZE_CONFIG,
            save_plotted_images=False,
            save_labels_txt=False
        )