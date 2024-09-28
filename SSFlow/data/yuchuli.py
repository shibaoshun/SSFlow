

import cv2,torch


def gradient( x):
    def sub_gradient(x):
        left_shift_x, right_shift_x, grad = torch.zeros_like(
            x), torch.zeros_like(x), torch.zeros_like(x)
        left_shift_x[:, :, 0:-1] = x[:, :, 1:]
        right_shift_x[:, :, 1:] = x[:, :, 0:-1]
        grad = 0.5 * (left_shift_x - right_shift_x)
        return grad

    return sub_gradient(x), sub_gradient(torch.transpose(x, 2, 3)).transpose(2, 3)

def hiseq_color_cv2_img(img):
    (b, g, r) = cv2.split(img)
    bH = cv2.equalizeHist(b)
    gH = cv2.equalizeHist(g)
    rH = cv2.equalizeHist(r)
    result = cv2.merge((bH, gH, rH))
    return result


input1=cv2.imread('E:\ZCZ/2023\LLF-main\dataset/test\low/0012/0089.png')
input2=input1.transpose(2,0,1)
input=torch.tensor(input2)
input=input.unsqueeze(0)
his=hiseq_color_cv2_img(input1)
color_map = input / (input.sum(dim=1, keepdims=True) + 1e-4)
dx, dy = gradient(color_map)
noise_map = torch.max(torch.stack([dx.abs(), dy.abs()], dim=0), dim=0)[0]

color_map=color_map.numpy()
color_map=color_map.squeeze(0)
color_map=color_map.transpose(1,2,0)
color_map=color_map*255.0

noise_map=noise_map.numpy()
noise_map=noise_map.squeeze(0)
noise_map=noise_map.transpose(1,2,0)
noise_map=noise_map*255.0

# cv2.imwrite("E:\ZCZ/2023\LLF-main28/yuchuli/his.png",his)
# cv2.imwrite("E:\ZCZ/2023\LLF-main28/yuchuli/color.png",color_map)
cv2.imwrite("E:\ZCZ/2023\LLF-main28/yuchuli/noise.png",noise_map+100)