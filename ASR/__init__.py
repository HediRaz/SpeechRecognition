import torch


device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print(f"Working on {device}")
