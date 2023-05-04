import argparse
import os



"""
In this script, we create the config.yaml file used to train YOLO.
For further information check : https://docs.ultralytics.com/yolov5/tutorials/train_custom_data/#1-create-dataset
"""

def write_array_in_file(array, path_file):
    # This function is used to write each value inside an array to a file
    if os.path.exists(path_file):
        os.remove(path_file)
    with open(path_file, "a") as f:
        for val in array:
            f.write(val + "\n")
    f.close()

def change_bbox_format(array, dataset_path):
    # Change the format of the bbox from (x1 y1 x2 y2) to (x y w h) as required by YOLO :
    # https://docs.ultralytics.com/yolov5/tutorials/train_custom_data/#11-create-datasetyaml
    for val in array:
        path_bbox = dataset_path + val + "-box.txt"
        bbox = []
        with open(path_bbox, "r") as f:
            for line in f.readlines():
                line = line.strip()
                bbox.append(line.split(" "))
        f.close()
        new_path_bbox = dataset_path + val + "-color.txt"
        # Remove the old path if already existing
        if os.path.exists(new_path_bbox):
            os.remove(new_path_bbox)
        with open(new_path_bbox, "a") as f:
            for obj in bbox:
                x1, y1, x2, y2 = float(obj[1]), float(obj[2]), float(obj[3]), float(obj[4])
                w = x2 - x1
                h = y2 - y1
                x = x1 + w*0.5
                y = y1 + h*0.5
                f.write(obj[0] + f" {x} {y} {w} {h}\n")
        f.close()

if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser(
        description='Process the YCB-video dataset to train the YOLOv8 model'
    )
    parser.add_argument('--dataset_path', type=str, help='Path to the dataset root folder (insided YCB_Video_Dataset)')
    parser.add_argument('--data_config', type=str, help='.yaml config file used to train the model')
    
    args = parser.parse_args()

    train_set_path = args.dataset_path + "/image_sets/train.txt"
    val_set_path = args.dataset_path + "/image_sets/val.txt"
    classes_path = args.dataset_path + "/image_sets/classes.txt"

    classes_names = []
    # Get all classes names
    with open(classes_path, "r") as f:
        for class_name in f.readlines():
            classes_names.append(class_name)
    f.close()

    train_set = []
    # Get all the images for training
    with open(train_set_path, "r") as f:
        for line in f.readlines():
            line = line.strip()
            image_path = "/data/" + line
            train_set.append(image_path)
    f.close()

    val_set = []
    with open(val_set_path, "r") as f:
        for line in f.readlines():
            line = line.strip()
            image_path = "/data/" + line
            val_set.append(image_path)
    f.close()

    # Create two new files containing path to the images
    write_array_in_file(train_set, args.dataset_path + "/train.txt")
    write_array_in_file(val_set, args.dataset_path + "/val.txt")

    # We change and recreate bounding box files : x1 y1 x2 y2 --> x y w h
    change_bbox_format(train_set, args.dataset_path)
    change_bbox_format(val_set, args.dataset_path)

    # We finally create the .yaml file for the training of YOLO
    with open(args.data_config, "w") as f:
        train_path = args.dataset_path + "/train.txt"
        val_path = args.dataset_path + "/val.txt"
        f.write(f"path: {args.dataset_path}\n") # set path to dataset
        f.write(f"train: {train_path}\n") # set relative path to train .txt file
        f.write(f"val: {val_path}\n") # set relative path to val .txt file

        f.write("\n")
        i = 0
        for class_name in classes_names:
            f.write(f"{i}: {class_name}\n")
    f.close()


