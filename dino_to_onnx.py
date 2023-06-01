import torch
from torchvision import transforms as T
from datetime import datetime
from PIL import Image

class xyz_model(torch.nn.Module): 
    def __init__(self, model): 
        super().__init__() 
        self.model = model  

    def forward(self, tensor): 
        ff = self.model(tensor) 
        return ff  
    
# Content of this script obtained from : https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html
if __name__ == '__main__':
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    mm = xyz_model(model).to(device) 

    mm.eval() 
    input_data = torch.randn(1, 3, 224, 224).to(device) 
    output = mm(input_data) 

    torch.onnx.export(mm, input_data, "onnx_models/dinov2.onnx", input_names = ['input'])
    print("ok")