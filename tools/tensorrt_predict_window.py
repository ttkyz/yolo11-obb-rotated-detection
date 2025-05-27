import argparse
import os
import time
import torch
import cv2
import numpy as np
import math
from ultralytics import YOLO
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

#export OPENCV_IO_MAX_IMAGE_PIXELS=1099511627776
def efficient_yolo_sliced_inference_on_single_image_with_trt(
    model_path: str,
    image_path: str,
    conf_threshold: float = 0.25,  # Detection confidence threshold
    slice_height: int = 1024,      # SAHI slice height, recommended to match model's img_size
    slice_width: int = 1024,       # SAHI slice width, recommended to match model's img_size
    overlap_height_ratio: float = 0.1,  # Vertical overlap ratio (0.0 to 1.0)
    overlap_width_ratio: float = 0.1,  # Horizontal overlap ratio (0.0 to 1.0)
    line_thickness: int = 1,       # Bounding box line thickness, default 1 pixel
    output_project_name: str = "yolo_trt_sahi_single_image_predictions", # Output project name
    output_run_name_prefix: str = "trt_sahi_sliced_inference", # Prefix for the output run directory
    sahi_match_metric: str = 'IOU', # SAHI postprocess match metric
    sahi_match_threshold: float = 0.5, # SAHI postprocess match threshold
    hide_labels: bool = True,      # Hide labels in output image
    hide_conf: bool = True         # Hide confidence scores in output image
):
    """
    Efficiently performs sliced prediction on a single large image using SAHI
    with a YOLO TensorRT engine, and customizes bounding box style (thinner lines, no labels).

    Parameters:
        model_path (str): Path to the YOLO TensorRT engine file (e.g., "yolo11s-obb.engine").
        image_path (str): Path to the single image file to process.
        conf_threshold (float): Minimum confidence threshold; detections below this will be ignored.
        slice_height (int): Height of each slice. Should ideally match the input size
                             your YOLO model was trained with.
        slice_width (int): Width of each slice. Should ideally match the input size
                             your YOLO model was trained with.
        overlap_height_ratio (float): Vertical overlap ratio between slices (0.0 to 1.0).
        overlap_width_ratio (float): Horizontal overlap ratio between slices (0.0 to 1.0).
        line_thickness (int): Thickness of the bounding box lines (in pixels).
        output_project_name (str): Parent directory name for organizing output results.
        output_run_name_prefix (str): Prefix for the individual run directory within the project.
        sahi_match_metric (str): Metric used for SAHI's NMS (e.g., 'IOU', 'IOS').
        sahi_match_threshold (float): Threshold for SAHI's NMS.
        hide_labels (bool): If True, labels will not be drawn on the output image.
        hide_conf (bool): If True, confidence scores will not be drawn on the output image.
    """
    print(f"加载 TensorRT 模型: {model_path}")
    try:
        ultralytics_model = YOLO(model_path)
        
        detection_model = AutoDetectionModel.from_pretrained(
            model=ultralytics_model,
            model_type='ultralytics',
            confidence_threshold=conf_threshold,
            device="cuda:0" if torch.cuda.is_available() else "cpu",
            image_size=slice_height,
        )
    except Exception as e:
        print(f"错误: {e}")
        return
    
    # 读取图像以获取其真实尺寸
    try:
        original_image = cv2.imread(image_path)
        image_height, image_width, _ = original_image.shape
        print(f"图像尺寸: {image_width}x{image_height} 像素.")
    except Exception as e:
        print(f"错误: {e}")
        return
    image_filename = os.path.basename(image_path)
    image_name_without_ext = os.path.splitext(image_filename)[0]
    
    output_run_name = f"{output_run_name_prefix}_{image_name_without_ext}_{slice_height}x{slice_width}"
    results_output_dir = os.path.join("runs", output_project_name, output_run_name)
    os.makedirs(results_output_dir, exist_ok=True)

    print(f"切片尺寸: {slice_width}x{slice_height}, 重叠率: {overlap_width_ratio}x{overlap_height_ratio}")
    print(f"置信度阈值: {conf_threshold}")
    print(f"结果保存至: '{os.path.abspath(results_output_dir)}'")


    # --- 模型预热 ---
    print("模型预热")
    try:
        dummy_image = np.random.randint(0, 256, size=original_image.shape, dtype=np.uint8)
        
        _ = get_sliced_prediction(
            dummy_image,  # 传入 Numpy 图像数组
            detection_model,
            slice_height=slice_height,
            slice_width=slice_width,
            overlap_height_ratio=overlap_height_ratio,
            overlap_width_ratio=overlap_width_ratio,
            postprocess_match_threshold=0.4, # 预热使用通用阈值
            verbose=False,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception as e:
        print(f"预热错误: {e}")
    print("模型完成")
    start_time = time.time() # 记录正式推理开始时间

    try:
        result = get_sliced_prediction(
            image_path,
            detection_model,
            slice_height=slice_height,
            slice_width=slice_width,
            overlap_height_ratio=overlap_height_ratio,
            overlap_width_ratio=overlap_width_ratio,
            postprocess_match_metric=sahi_match_metric,
            postprocess_match_threshold=sahi_match_threshold,
            verbose=False,
        )
        end_time = time.time() # 记录结束时间

        result.export_visuals(
            export_dir=results_output_dir,
            file_name=image_name_without_ext,
            hide_labels=hide_labels,
            hide_conf=hide_conf,
            rect_th=line_thickness,
        )
        print(f"预测结果图像已保存至: '{os.path.abspath(results_output_dir)}'")

    except Exception as e:
        print(f"切片预测错误: {e}")
        print("请检查 SAHI 切片参数和 GPU 内存.")
        return

    total_inference_time = end_time - start_time

    print(f"总预测时间 (不含预热): {total_inference_time:.4f} 秒.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="使用 SAHI 和 YOLO TensorRT 引擎对单张大图进行高效切片预测."
    )

    # Required arguments
    parser.add_argument(
        "--model",
        type=str,
        default="ckpts/yolo11n-obb-half-1024.engine",
        help="YOLO 引擎文件路径 (例如: 'ckpts/yolo11n-obb-half-1024.engine')."
    )
    parser.add_argument(
        "--image",
        type=str,
        default="jointlunchuan.jpg",
        help="待处理的单张大图文件路径."
    )

    # SAHI Slicing and Overlap Parameters
    parser.add_argument(
        "--slice_height",
        type=int,
        default=1024,
        help="SAHI 切片高度."
    )
    parser.add_argument(
        "--slice_width",
        type=int,
        default=1024,
        help="SAHI 切片宽度."
    )
    parser.add_argument(
        "--overlap_ratio",
        type=float,
        default=0.1,
        help="切片间高度和宽度的重叠率."
    )

    # Other Prediction Parameters
    parser.add_argument(
        "--conf_threshold",
        type=float,
        default=0.6,
        help="检测置信度阈值; 低于此值的检测将被忽略."
    )
    parser.add_argument(
        "--line_thickness",
        type=int,
        default=2,
        help="边界框线条粗细 (像素)."
    )
    parser.add_argument(
        "--output_project_name",
        type=str,
        default="yolo_sahi_single_image_predictions",
        help="输出结果的父目录名."
    )
    parser.add_argument(
        "--output_run_name_prefix",
        type=str,
        default="sahi_sliced_inference",
        help="项目内独立运行目录的前缀."
    )
    parser.add_argument(
        "--sahi_match_metric",
        type=str,
        default="IOU",
        choices=['IOU', 'IOS'],
        help="SAHI 后处理 NMS 使用的度量. 选项: 'IOU', 'IOS'."
    )
    parser.add_argument(
        "--sahi_match_threshold",
        type=float,
        default=0.5,
        help="SAHI 后处理 NMS 阈值."
    )
    parser.add_argument(
        "--hide_labels",
        action="store_true",
        default=True,
        help="如果设置, 输出图像上将不绘制标签."
    )
    parser.add_argument(
        "--hide_conf",
        action="store_true",
        default=True,
        help="如果设置, 输出图像上将不绘制置信度分数."
    )

    args = parser.parse_args()

    print("\n--- 单张 1024x1024 纯 TensorRT 推理速度测试 ---")
    single_slice_model = YOLO(args.model)

    dummy_input_image = np.zeros((1024, 1024, 3), dtype=np.uint8)

    # 预热
    for _ in range(10):
        _ = single_slice_model(dummy_input_image, imgsz=1024, conf=0.25, iou=0.7, verbose=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    # 正式计时
    start_time_single = time.time()
    num_runs_single = 100
    for _ in range(num_runs_single):
        _ = single_slice_model(dummy_input_image, imgsz=1024, conf=0.25, iou=0.7, verbose=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    end_time_single = time.time()
    avg_inference_time_single = (end_time_single - start_time_single) / num_runs_single
    print(f"每 1024x1024 切片平均纯 TensorRT 推理时间: {avg_inference_time_single:.4f} 秒")

    # --- 动态计算总切片数 ---
    try:
        temp_img = cv2.imread(args.image)
        image_h, image_w, _ = temp_img.shape
        del temp_img
    except Exception as e:
        image_h, image_w = 30000, 30000
        print(f"使用默认 {image_w}x{image_h} 进行切片估计.")

    stride_h = args.slice_height * (1 - args.overlap_ratio)
    stride_w = args.slice_width * (1 - args.overlap_ratio)

    num_slices_w = math.ceil((image_w - args.slice_width) / stride_w) + 1
    num_slices_h = math.ceil((image_h - args.slice_height) / stride_h) + 1
    
    total_estimated_slices = num_slices_w * num_slices_h
    print(f"基于图像尺寸 {image_w}x{image_h}, 切片尺寸 {args.slice_width}x{args.slice_height}, 重叠 {args.overlap_ratio}:")
    print(f"估计总切片数: {total_estimated_slices} 片.")

    estimated_total_inference_time = avg_inference_time_single * total_estimated_slices
    print(f"理论上, 仅模型推理的总时间 (不含 SAHI 开销): {estimated_total_inference_time:.4f} 秒")

    if not torch.cuda.is_available():
        print("在 CPU 上运行.")
    else:
        print(f"使用设备: {torch.cuda.get_device_name(0)}")

    print("\n--- 开始 SAHI 切片推理 ---")
    efficient_yolo_sliced_inference_on_single_image_with_trt(
        model_path=args.model,
        image_path=args.image,
        conf_threshold=args.conf_threshold,
        slice_height=args.slice_height,
        slice_width=args.slice_width,
        overlap_height_ratio=args.overlap_ratio,
        overlap_width_ratio=args.overlap_ratio,
        line_thickness=args.line_thickness,
        output_project_name=args.output_project_name,
        output_run_name_prefix=args.output_run_name_prefix,
        sahi_match_metric=args.sahi_match_metric,
        sahi_match_threshold=args.sahi_match_threshold,
        hide_labels=args.hide_labels,
        hide_conf=args.hide_conf
    )