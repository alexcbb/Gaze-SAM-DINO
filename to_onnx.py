from ultralytics import YOLO
import argparse


if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser(
        description='Convert a YOLO model checkpoint to ONNX format'
    )
    parser.add_argument('--model_path', type=str, help='Path to the model checkpoint')

    args = parser.parse_args()

    # Load a model
    model = YOLO(args.model_path)  # build a new model from scratch

    success = model.export(format='onnx', opset=12)  # export the model to ONNX format