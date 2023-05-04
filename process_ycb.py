import argparse

"""
In this script, we create the config.yaml file used to train YOLO.
For further information check : https://docs.ultralytics.com/yolov5/tutorials/train_custom_data/#1-create-dataset
"""
if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser(
        description='Process the YCB-video dataset to train the YOLOv8 model'
    )
    parser.add_argument('--dataset_path', type=int, help='Path to the dataset root folder (insided YCB_Video_Dataset)')
    parser.add_argument('--data_config', type=int, help='.yaml config file used to train the model')
    
    args = parser.parse_args()

    train_set_path = args.dataset_path + "/images_sets/train.txt"
    test_set_path = args.dataset_path + "/images_sets/val.txt"
    classes_path = args.dataset_path + "/images_sets/classes.txt"

    tmp_train_set = []
    # We first split the training set into train and val set
    with open(train_set_path, "r") as f:
        for line in f.readlines():
            line = line.strip()
            image_path =  args.dataset_path + "/data/" + line
            tmp_train_set.append(image_path)

    train_len = int(len(tmp_train_set) / 2)

    # Create two new files containing path to the images

    # We change and recreate bounding box files : x1 y1 x2 y2 --> x y w h



