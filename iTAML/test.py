import torch
print("CUDA Available: " + str(torch.cuda.is_available()))
print("CUDA Version: " + torch.version.cuda)
print("GPU Count: " + str(torch.cuda.device_count()))
for i in range(1000): print(i)
print("\n")