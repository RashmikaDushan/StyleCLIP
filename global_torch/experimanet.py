
import torch
import clip
from manipulate import Manipulator

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device,jit=False)
    
    network_pkl='D:/FYP/StyleGAN2/stylegan2-ada-pytorch/ffhq.pkl'
    device = torch.device('cuda')
    M=Manipulator()
    M.device=device
    G=M.LoadModel(network_pkl,device)
    M.G=G
    M.SetGParameters()
    num_img=100
    M.GenerateS(num_img=num_img)
    print(f"Style vectors shape: {len(M.dlatents)}, {M.dlatents[0].shape[0]}, {M.dlatents[0].shape[1]}")