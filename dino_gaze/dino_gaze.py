import torch
from torchvision import transforms as T
from datetime import datetime
from PIL import Image
import numpy as np


from transformers import ViTImageProcessor, ViTModel
from matplotlib import pyplot as plt
# Content of this script obtained from : https://github.com/facebookresearch/dinov2/issues/2 and https://www.kaggle.com/code/stpeteishii/dino-visualize-self-attention-sample/notebook

# Return the id of the patch given the pixel's position
def get_patch_id(patch_size, pos):
    return (pos[0]//patch_size[0], pos[1]//patch_size[1])

# TODO : create a function that extract the attention matrix + the final attention
def get_last_attention_layer(model, image):
    x = model.patch_embed(image)
    nb_blocks = len(model.blocks)
    for i in range(nb_blocks-1):
        x = model.blocks[i](x)
    x = model.blocks[-1].norm1(x)
    attn = model.blocks[-1].attn(x)
    return attn

if __name__ == '__main__':
    """
    batch_size = 1
    num_attention_heads = 6
    input_height, input_width = 242, 940
    image = Image.open("./img.jpg")

    processor = ViTImageProcessor.from_pretrained('facebook/dino-vits8')
    model = ViTModel.from_pretrained('facebook/dino-vits8')

    print(model.config)

    inputs = processor(images=image, return_tensors="pt") # https://huggingface.co/docs/transformers/model_doc/vit#transformers.ViTImageProcessor
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True) # https://huggingface.co/docs/transformers/model_doc/vit#transformers.ViTImageProcessor
        last_attention_map = outputs.attentions[0]

    resized_attention_maps = np.resize(last_attention_map, (batch_size, num_attention_heads, input_height, input_width))
    normalized_attention_maps = resized_attention_maps / np.max(resized_attention_maps, axis=(2, 3), keepdims=True)

    fig, axes = plt.subplots(nrows=num_attention_heads, ncols=1)

    for i, ax in enumerate(axes):
        ax.imshow(resized_attention_maps[0, i], interpolation='nearest')
        ax.set_title(f'Attention Head {i+1}')

    plt.show()

    #w_feature_map = h_feature_map = 224 // 8 # = 28
    #last_attention = last_attention.reshape(6, w_feature_map, h_feature_map)



    """
    patch_size = 14
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    #model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    model.eval()
    model.to(device)

    # TODO load images or get video stream
    img = Image.open("./ycb1.png")
    
    image_transforms = T.Compose([
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    t_img = image_transforms(img)
    print(t_img.shape)
    w, h = t_img.shape[1] - t_img.shape[1] % patch_size, t_img.shape[2] - t_img.shape[2] % patch_size
    t_img = t_img[:, :w, :h].unsqueeze(0).to(device)
    print(t_img.shape)

    w_featmap = t_img.shape[-2] // patch_size
    h_featmap = t_img.shape[-1] // patch_size
    print(h_featmap)
    print(w_featmap)

    #print(model.blocks[-1].attn.qkv) # extract qkv values !
    print(model.blocks[-1])

    start_time = datetime.now()
    with torch.no_grad():
        attentions = get_last_attention_layer(model, t_img) # TODO : find a way to do the same with DINOv2
    end_time = datetime.now()
    time_difference = (end_time - start_time).total_seconds() * 10**3
    print("Inference time: ", time_difference, "ms")
    """start_time = datetime.now()
    with torch.no_grad():
        attentions = model.get_last_selfattention(t_img) # TODO : find a way to do the same with DINOv2
    end_time = datetime.now()
    time_difference = (end_time - start_time).total_seconds() * 10**3
    print("Inference time: ", time_difference, "ms")"""
    
    nh = attentions.shape[1]
    print(attentions.shape)

    # Only keep the output patch attention
    # [1, 6, 901, 901] => [6, 900]
    attentions = attentions[0, :, 786, 1:].reshape(nh, -1) # TODO : change the value 155 with the right one 

    # Only keep a certain purcentage of the mass
    val, idx = torch.sort(attentions)
    val /= torch.sum(val, dim=1, keepdim=True)
    cumval = torch.cumsum(val, dim=1)

    # We visualize masks obtained by thresholding 
    # the self-attention maps to keep xx% of the mass.
    threshold = 0.2 
    th_attn = cumval > (1 - threshold)
    idx2 = torch.argsort(idx)
    for head in range(nh):
        th_attn[head] = th_attn[head][idx2[head]]
        
    th_attn = th_attn.reshape(nh, w_featmap, h_featmap).float()

    # interpolate
    th_attn = torch.nn.functional.interpolate(th_attn.unsqueeze(0), scale_factor=patch_size, mode="nearest")[0].cpu().numpy()

    attentions = attentions.reshape(nh, w_featmap, h_featmap)
    attentions = torch.nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=patch_size, mode="nearest")[0].cpu().numpy()
    attentions_mean = np.mean(attentions, axis=0)

    plt.figure(figsize=(15,15), dpi=200)

    plt.subplot(3, 1, 1)
    plt.title("Original",size=6)
    plt.imshow(t_img.cpu().numpy()[0].transpose((1, 2, 0)))
    plt.axis("off")

    plt.subplot(3, 1, 2)
    plt.title("Attentions Mean",size=6)
    plt.imshow(attentions_mean)
    plt.axis("off")

    plt.subplot(3, 1, 3)
    plt.title("Attention",size=6)
    plt.imshow(attentions[-1])
    plt.axis("off")
    plt.show()
    