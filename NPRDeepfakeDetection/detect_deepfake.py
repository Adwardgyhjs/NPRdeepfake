import argparse
import torch
from torchvision import transforms
from PIL import Image
from .networks.base_model import BaseModel
from .options.test_options import TestOptions
from .data import create_dataloader
from .networks.resnet import resnet50
import os
import numpy as np
import random

vals = ['progan', 'stylegan', 'stylegan2', 'biggan', 'cyclegan', 'stargan', 'gaugan', 'deepfake']
multiclass = [1, 1, 1, 0, 1, 0, 0, 0]

def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False



def detect_deepfake(image_path, model_path):
    seed_torch(100)
    opt = TestOptions().parse(print_options=False)

    # get model
    model = resnet50(num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location='cpu'), strict=False)
    model.cuda()
    model.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    image = Image.open(image_path).convert('RGB')
    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(image).unsqueeze(0).to(device)


    with torch.no_grad():
        output = model(input_tensor).sigmoid().flatten().tolist()
    print("output is ",output)
    index=np.argmax(np.array(output))
    print("output max index:",index)
    #print(f'The image is {vals[index]}')

    print(f'该照片 {"是深度伪造的" if output[0]>0.52 and output[1]<0.499 else "不是深度伪造的"}')
    return output[0]  # Assuming 1 indicates a deepfake


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Detect deepfake in an image')
    parser.add_argument('--image', type=str, required=True, help='Path to the input image')
    parser.add_argument('--model', type=str, required=True, help='Path to the deepfake detection model')
    args = parser.parse_args()


    is_deepfake = detect_deepfake(args.image, args.model)
    print(f'The image is {"a deepfake" if is_deepfake else "not a deepfake"}')
