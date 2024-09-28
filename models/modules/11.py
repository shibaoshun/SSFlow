
import cv2
import torch
gt=cv2.imread("E:\ZCZ/2023\LLF-main\dataset/train\high/0000/0010.png")
gt=torch.from_numpy(gt)
gt=gt.permute(2,0,1)
gt = gt.unsqueeze(0)
fea_out = gt
fea_out = fea_out / (fea_out.sum(dim=1, keepdims=True) + 1e-4)
fea_out = fea_out.squeeze(0)
# fea_out=fea_out.permute(1,2,0)
fea_out = fea_out.cpu().detach().numpy()
fea_out = fea_out.transpose(1, 2, 0)
fea_out = fea_out * 255.0
import numpy as np

fea_out = fea_out.astype(np.uint8)
#fea_out = cv2.cvtColor(fea_out, cv2.COLOR_BGR2RGB)
cv2.imwrite("E:\ZCZ/2023\LLF-main4/results/kernel_1.png", fea_out)