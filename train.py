import os
import sys
import torch
from model import GAN

if os.path.exists('./model.pt'):
    print("Loading model...")
    model = torch.load('./model.pt')
else:
    print("Initializing model..")
    model = GAN()

# training
device = torch.device("cuda:0") if torch.cuda.is_available else torch.device('cpu')
model.train(sys.argv[1:], num_epoch=10, batch_size=64, device=device, max_len=5000, lr=1e-4, max_resolution=256)
