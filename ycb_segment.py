from PIL import images 
import os, sys, argparse


if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser(
        description='Extract the segmentation of objects on YCB dataset with Yolo bounding box'
    )
    parser.add_argument('dataset_path', type=str, help='Path to the YCB_dataset')
    parser.add_argument('', type=int, help='')


