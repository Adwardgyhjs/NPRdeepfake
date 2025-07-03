import os
import argparse
from SRCNN_pytorch.super_resolve import super_resolve
from DeepfakeDetection.detect_deepfake import detect_deepfake

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Super-resolve an image and detect if it is a deepfake')
    parser.add_argument('--image', type=str, required=True, help='Path to the input image')
    parser.add_argument('--srcnn_model', type=str, required=True, help='Path to the SRCNN model')
    parser.add_argument('--deepfake_model', type=str, required=True, help='Path to the deepfake detection model')
    args = parser.parse_args()


    sr_image_path = super_resolve(args.image, args.srcnn_model, scale_factor=3)
    print(f'Super-resolved image saved at {sr_image_path}')


    is_deepfake = detect_deepfake(sr_image_path, args.deepfake_model)
    print(f'The super-resolved image is {"a deepfake" if is_deepfake else "not a deepfake"}')
