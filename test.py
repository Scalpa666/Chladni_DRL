import torch

# Check if GPU is available
if torch.cuda.is_available():
    device = torch.device('cuda')  # Use GPU
    print("Using GPU")
else:
    device = torch.device('cpu')  # Fall back to CPU
    print("Using CPU")
print(torch.__version__)
print(torch.version.cuda)
print(torch.backends.cudnn.enabled)
print("cuDNN Version:", torch.backends.cudnn.version())