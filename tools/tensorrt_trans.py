from ultralytics import YOLO
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="Export YOLO model to TensorRT format and run inference.")
    parser.add_argument("--model", type=str, 
                        help="Path to the YOLO model to be converted.")
    parser.add_argument("--output", type=str, default=None,
                        help="Name of the output TensorRT engine file.")
    parser.add_argument("--imgsz", type=int, default=1024,
                        help="Image size for TensorRT export.")
    parser.add_argument("--half", type=bool, default=True,
                        help="Path to the image for inference.")
    parser.add_argument("--int8", type=bool, default=False,
                        help="Path to the image for inference.")
    parser.add_argument("--workspace", type=int, default=16,
                        help="Workspace size in GB for TensorRT export.")
    parser.add_argument("--inference_image", type=str, default="lunchuan.jpg",
                        help="Path to the image for inference.")

    args = parser.parse_args()

    # Load the YOLO11 model
    model = YOLO(args.model)

    # Define paths
    base_name = os.path.splitext(os.path.basename(args.model))[0]
    default_engine_path = os.path.join(os.path.dirname(args.model), f"{base_name}.engine")
    engine_path = default_engine_path

    # Export the model to TensorRT format
    model.export(format="engine", imgsz=args.imgsz, workspace=args.workspace, half=args.half, int8=args.int8)

    if args.output != None:
        new_engine_path = os.path.join(os.path.dirname(args.model), args.output)
        # Rename the exported engine file
        if os.path.exists(default_engine_path):
            os.rename(default_engine_path, new_engine_path)
            print(f"Renamed default engine '{default_engine_path}' to '{new_engine_path}'")
        else:
            print(f"Warning: Default engine file '{default_engine_path}' not found. Skipping rename.")
        engine_path = new_engine_path


    # Load the exported TensorRT model
    tensorrt_model = YOLO(engine_path)

    # Run inference
    results = tensorrt_model(args.inference_image)
    print("Inference completed successfully.")

if __name__ == "__main__":
    main()