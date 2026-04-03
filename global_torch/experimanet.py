from PIL import Image
import numpy as np
import torch
import clip
from StyleCLIP import GetStyleDirection, GetClipDirection
from manipulate import Manipulator
import os

os.makedirs("results", exist_ok=True)

if __name__ == "__main__":
    print("Script starting...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device, jit=False)

    network_pkl = "ffhq.pkl"
    device = torch.device("cuda")
    M = (
        Manipulator()
    )  # Which has the generator and the mapping network and other utilities for manipulation
    M.device = device
    G = M.LoadModel(
        network_pkl, device
    )  # Load the pretrained StyleGAN2 model from the pkl file
    M.G = G  # Set the generator in the Manipulator class
    M.SetGParameters()  # Set parameters related to the generator, such as number of layers, image size, etc.
    num_img = 100_000

    M.GenerateS(
        num_img=num_img
    )  # Generate style vectors (dlatents) for a large number of random latent codes. This will be used to compute the mean and std of the style space.
    print(
        f"Style vectors shape: {len(M.dlatents)}, {M.dlatents[0].shape[0]}, {M.dlatents[0].shape[1]}"
    )

    M.GetStyleVecMS()  # Compute the mean and std of the style vectors
    print(f"Mean:  {len(M.code_mean)}, { M.code_mean[0].shape[0]}")
    print(f"Std:  {len(M.code_std)}, { M.code_std[0].shape[0]}")

    file_path = "global_torch/npy/ffhq/"
    fs3_matrix = np.load(
        file_path + "fs3.npy"
    )  # Load the precomputed style space x CLIP direction matrix (fs3) for the FFHQ dataset

    img_indexs = np.arange(8)  # image count | n = 8

    style_vecs_initial = [
        tmp[img_indexs] for tmp in M.dlatents
    ]  # get top n style vectors | shape = 26, n, 512
    M.num_images = len(img_indexs)
    print(
        f"Selected style vectors shape: {len(style_vecs_initial)}, {style_vecs_initial[0].shape[0]}, {style_vecs_initial[0].shape[1]}"
    )

    M.step = 1  # step size for manipulation

    paras = np.array(
        [
            ["original", "person", "original", 0, 0],
            ["white_hair", "face with black hair", "face with white hair", 0.15, 4],
            ["white_hair", "face with black hair", "face with white hair", 0.25, 4],
            ["white_hair", "face with black hair", "face with white hair", 0.35, 4],
        ]
    )

    for i in range(len(paras)): # for each text prompt
        filename, neutral, target, beta, alpha = paras[i]

        beta = np.float32(beta)
        alpha = np.float32(alpha)
        M.alpha = [alpha]

        print()
        print(filename, beta, alpha)
        classnames = [target, neutral]

        clip_direction = GetClipDirection(
            classnames, model
        )  # Get the CLIP direction vector for the target - neutral classes

        style_direction, num_c = GetStyleDirection(fs3_matrix, clip_direction, M, threshold=beta) # Get the style space direction corresponding to the CLIP direction

        style_vectors = M.CalcStyleVectors(style_vecs_initial, style_direction) # Manipulate the style vectors in the direction of the style_direction with the given alpha and beta

        out = M.GenerateImg(style_vectors)
        print(f"Generated image shape: {out.shape}")

        for i in range(len(out)):
            img = out[i][0]
            img = (img * 255).astype(np.uint8)

            img = 255 - img

            path = f"results/{i}_{filename}_beta{beta:.2f}_alpha{alpha:.2f}_{i}.png"

            Image.fromarray(img).save(path)
            print(f"Saved: image {i} - {filename}")
