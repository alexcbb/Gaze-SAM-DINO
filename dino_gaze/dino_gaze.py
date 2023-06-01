import torch
from torchvision import transforms as T
from datetime import datetime
from PIL import Image
import numpy as np
import argparse
from segment_anything import SamPredictor, sam_model_registry
from dinov2.dinov2.models.vision_transformer import vit_small
import cv2

from matplotlib import pyplot as plt
# Content of this script obtained from : https://github.com/facebookresearch/dinov2/issues/2 and https://www.kaggle.com/code/stpeteishii/dino-visualize-self-attention-sample/notebook

# Return the id of the patch given the pixel's position
def get_patch_id(patch_size, x, y, img_w, img_h):
    max_x_id = img_w // patch_size 
    id = (x // patch_size) + (y // patch_size) * max_x_id
    return id

def onclickdino(event):
    global img
    if event.button == 1 and event.inaxes is ax1:
        x_pos, y_pos = int(event.xdata), int(event.ydata)
        image_transforms = T.Compose([
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        t_img = image_transforms(img)
        w, h = t_img.shape[1] - t_img.shape[1] % patch_size, t_img.shape[2] - t_img.shape[2] % patch_size
        t_img = t_img[:, :w, :h].unsqueeze(0).to(device)

        w_featmap = t_img.shape[-2] // patch_size
        h_featmap = t_img.shape[-1] // patch_size

    
        start_time = datetime.now()
        with torch.no_grad():
            attentions = model.get_last_self_attention(t_img) 
        end_time = datetime.now()
        time_difference = (end_time - start_time).total_seconds() * 10**3
        print(f"Inference time : {time_difference}ms")
        
        nh = attentions.shape[1]
        patch_id = get_patch_id(patch_size, x_pos, y_pos, w, h)
        print(f"Id of the clicked patch :{patch_id}")

        attentions = attentions[0, :, patch_id, 1:].reshape(nh, -1) # TODO : change the value 155 with the right one 

        val, idx = torch.sort(attentions)
        val /= torch.sum(val, dim=1, keepdim=True)
        cumval = torch.cumsum(val, dim=1)

        threshold = 0.5
        th_attn = cumval > (1 - threshold)
        idx2 = torch.argsort(idx)
        for head in range(nh):
            th_attn[head] = th_attn[head][idx2[head]]
            
        th_attn = th_attn.reshape(nh, w_featmap, h_featmap).float()

        th_attn = torch.nn.functional.interpolate(th_attn.unsqueeze(0), scale_factor=patch_size, mode="nearest")[0].cpu().numpy()
        th_attn_mean = np.mean(th_attn, axis=0)

        attentions = attentions.reshape(nh, w_featmap, h_featmap)
        attentions = torch.nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=patch_size, mode="nearest")[0].cpu().numpy()
        attentions_mean = np.mean(attentions, axis=0)

        ax2.imshow(attentions_mean)
        ax3.imshow(th_attn_mean)
        plt.draw()  

def onclicksam(event):
    global img
    global blured_image
    if event.button == 1 and event.inaxes is ax1:
        x_pos, y_pos = int(event.xdata), int(event.ydata)
        
        input_point = np.array([[x_pos, y_pos]])
        input_label = np.array([1])
    
        start_time = datetime.now()
        model.set_image(img)
        masks, _, _ = model.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )
        end_time = datetime.now()
        time_difference = (end_time - start_time).total_seconds() * 10**3
        print(f"Inference time : {time_difference}ms")
        mask = update_distance(masks[0].astype(np.float32), x_pos, y_pos)
        blured_image = apply_focus_blur(img, mask, blur_amount=25)
        ax2.imshow(mask)
        ax3.imshow(blured_image)
        plt.draw()  

def update_distance(mask, fade_x, fade_y, fade_factor=0.5):
    height, width = mask.shape[:2]
    max_distance = max(height, width) / 4 # TODO : change this parameter to a better one ?

    for y in range(height):
        for x in range(width):
            distance = np.sqrt((x - fade_x) ** 2 + (y - fade_y) ** 2)

            if distance > max_distance:
                mask[y, x] = 0
            else:
                fade = (max_distance - distance) / max_distance
                mask[y, x] *= fade_factor * fade

    return mask

def apply_focus_blur(image, mask, blur_amount):
    blurred_image = cv2.GaussianBlur(image, (blur_amount, blur_amount), 0)
    result = image.copy()
    result[mask<=0] = blurred_image[mask<=0]
    return result

if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser(
        description='Extract the saliency of objects on images with gaze information'
    )
    parser.add_argument('--sam_path', type=str, default="./../sam_vit_b_01ec64.pth", help='Path to the SAM model')
    parser.add_argument('--dino_path', type=str, default="./../dinov2_vits14_pretrain.pth", help='Path to the dino model')
    parser.add_argument('--model', type=str, default="dino", help='Model to use')
    parser.add_argument('--image_path', type=str, default="./../90139.png", help='Image path')

    args = parser.parse_args()
    patch_size = 14
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    image_path = args.image_path
    
    if args.model == 'dino':
        # Load DINO model
        model = vit_small(patch_size=patch_size,init_values=1.0, img_size=526, block_chunks=0)
        model.load_state_dict(torch.load(args.dino_path))
        for p in model.parameters():
            p.requires_grad = False
        model.eval()
        model.to(device)

        # Load image
        img = Image.open(image_path)

        # Show images
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        ax1.imshow(img) 
        ax2.imshow(img)
        ax3.imshow(img)

        # Connect canvas to a clicking event for visualisation
        cid = fig.canvas.mpl_connect('button_press_event', onclickdino)
        plt.show()
    else:
        if args.sam_path.split('/')[-1] == "sam_vit_b_01ec64.pth":
            model_type = "vit_b"
        elif args.sam_path.split('/')[-1] == "sam_vit_h_4b8939.pth":
            model_type = "default"
        else:
            model_type = "vit_l"
        sam = sam_model_registry[model_type](checkpoint=args.sam_path) # To use this model, you need to download it here : https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
        sam.to(device)
        model = SamPredictor(sam)

        # Load image
        original_image = cv2.imread(image_path)
        img = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

        # Show images
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        ax1.imshow(img) 
        ax2.imshow(img)
        ax3.imshow(img)

        # Connect canvas to a clicking event for visualisation
        cid = fig.canvas.mpl_connect('button_press_event', onclicksam)
        plt.show()
    