from typing import List, Any

from PIL import Image, ImageFilter
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter

from PIL import Image
import numpy as np


# 读取图像并转换为灰度图像
im = Image.open('1.jpg').convert('L')
# im.show()

# 高斯滤波，去除噪声
sigma = 1.0  # 高斯标准差
filter_extent = int(4 * sigma)  # 计算滤波器长度
x = np.arange(-filter_extent, filter_extent + 1)
# 生成一维高斯核
c = 1 / (np.sqrt(2 * np.pi) * sigma)
gauss_kernel = c * np.exp(-(x ** 2) / (2 * sigma ** 2))
# 标准化
gauss_kernel = gauss_kernel / np.sum(gauss_kernel)
# 对图像进行高斯滤波平滑
im_smooth = gaussian_filter(im, sigma=sigma)
im_smooth_show = Image.fromarray(im_smooth)
# im_smooth_show.show()


# 分别计算x,y方向上的梯度，Sobel算子
dx = np.array([-1, 0, 1, -2, 0, 2, -1, 0, 1])
dy = np.array([-1, -2, -1, 0, 0, 0, 1, 2, 1])
gx = np.abs(im_smooth_show.filter(ImageFilter.Kernel((3, 3), dx,1,0)).getdata())
gx_re = gx.reshape(im_smooth_show.size[1],im_smooth_show.size[0])
gy = np.abs(im_smooth_show.filter(ImageFilter.Kernel((3, 3), dy,1,0)).getdata())
gy_re = gy.reshape(im_smooth_show.size[1],im_smooth_show.size[0])
# x,y梯度图像的融合
gradient = np.sqrt(gx_re**2 + gy_re**2)
# 融合梯度图像的展示
gray_image = Image.fromarray(gradient)
# gray_image.show()
# 梯度方向
gradient_direction = np.arctan2(gy_re, gx_re) * 180 / np.pi
gradient_direction[gradient_direction < 0] += 180
gradient_direction = np.round(gradient_direction / 45) * 45



# 非极大值抑制
nms = np.zeros(gradient.shape)
for i in range(1, gradient.shape[0]-1):
    for j in range(1, gradient.shape[1]-1):
        if gradient_direction[i][j] == 0:
            # 梯度方向为0度时左右3列比对
            nms[i][j] = max(gradient[i][j-1:j+2])
        elif gradient_direction[i][j] == 45:
            # 梯度方向为45度时8邻域比对
            nms[i][j] = max(gradient[i-1:i+2, j-1:j+2].flatten())
        elif gradient_direction[i][j] == 90:
            # 梯度方向为90度时上下3行比对
            nms[i][j] = max(gradient[i-1:i+2, j])
        elif gradient_direction[i][j] == 135:
            # 梯度方向为135度时右斜向下比对
            nms[i][j] = max(gradient[i-1:i+2, j-1:j+2][::-1].flatten())

# # 得到单像素边缘信息
# for i in range(1, gradient.shape[0]-1):
#     for j in range(1, gradient.shape[1]-1):
#         if gradient_direction[i][j] == 0:
#             # 梯度方向为0度时左右2列比对
#             left_pixel = gradient[i][j-1]
#             right_pixel = gradient[i][j+1]
#             if gradient[i][j] >= max(left_pixel, right_pixel):
#                 nms[i][j] = gradient[i][j]
#         elif gradient_direction[i][j] == 45:
#             # 梯度方向为45度时2个对角线方向比对
#             up_left_pixel = gradient[i-1][j-1]
#             down_right_pixel = gradient[i+1][j+1]
#             if gradient[i][j] >= max(up_left_pixel, down_right_pixel):
#                 nms[i][j] = gradient[i][j]
#         elif gradient_direction[i][j] == 90:
#             # 梯度方向为90度时上下2行比对
#             up_pixel = gradient[i-1][j]
#             down_pixel = gradient[i+1][j]
#             if gradient[i][j] >= max(up_pixel, down_pixel):
#                 nms[i][j] = gradient[i][j]
#         elif gradient_direction[i][j] == 135:
#             # 梯度方向为135度时左斜向下和右斜向上比对
#             up_right_pixel = gradient[i-1][j+1]
#             down_left_pixel = gradient[i+1][j-1]
#             if gradient[i][j] >= max(up_right_pixel, down_left_pixel):
#                 nms[i][j] = gradient[i][j]

# 双阈值
high_threshold = np.mean(gradient) * 2
low_threshold = high_threshold / 2

# 双阈值分割
# 将大于高阈值的点认为是强边缘置1
strong_edges = (nms > high_threshold)
# 介于中间的像素点需进行进一步的检查
weak_edges = (nms >= low_threshold) & (nms <= high_threshold)

edge_map = strong_edges.astype(np.uint8) * 255

# edge_map = strong_edges.astype(np.uint8) * 255 + weak_edges.astype(np.uint8) * 128

# 边缘连接
# def connect_edges(i, j):
#     if edge_map[i][j] == 255:
#         for ii in range(max(0, i - 1), min(edge_map.shape[0], i + 2)):
#             for jj in range(max(0, j - 1), min(edge_map.shape[1], j + 2)):
#                 if edge_map[ii][jj] == 128:
#                     edge_map[ii][jj] = 255
#                     connect_edges(ii, jj)
#
#
# for i in range(edge_map.shape[0]):
#     for j in range(edge_map.shape[1]):
#         if edge_map[i][j] == 255:
#             connect_edges(i, j)

# dfs判断连通性
def dfs(i, j, visited, edges, group):
    stack = [(i, j)]
    while stack:
        x, y = stack.pop()
        if not visited[x][y]:
            visited[x][y] = True
            group.append((x, y))
            for p, q in ((x-1,y),(x+1,y),(x,y-1),(x,y+1)):
                if 0 <= p < edges.shape[0] and 0 <= q < edges.shape[1] and not visited[p][q] and edges[p][q] == 255:
                    stack.append((p, q))



# 定义标记函数
def mark_group(visited, edges):
    groups = []
    for i in range(visited.shape[0]):
        for j in range(visited.shape[1]):
            if visited[i][j] or edges[i][j] == 0:
                continue
            group = []
            dfs(i, j, visited, edges, group)
            groups.append(group)
    return groups

def remove_small_components(edges, threshold=2000):
    visited = np.zeros_like(edges, dtype=bool)
    for i in range(edges.shape[0]):
        for j in range(edges.shape[1]):
            if edges[i][j] == 255 and not visited[i][j]:
                group = []
                dfs(i, j, visited, edges, group)
                if len(group) <= threshold:
                    for x, y in group:
                        edges[x][y] = 0
    edges[edges > 0] = 255

remove_small_components(edge_map)

# 标记连通性并显示结果
visited = np.zeros(edge_map.shape, dtype=bool)
groups = mark_group(visited, edge_map)
max_area = 0
mean_area = 0
print("共检测到%d个连通分量：" % len(groups))
for i, group in enumerate(groups):
    print("第%d个连通分量包含%d个像素" % (i + 1, len(group)))
    if(len(group)>max_area):
        max_area = len(group)
    mean_area += len(group)
mean_area /= len(groups)
# 显示结果

remove_small_components(edge_map,mean_area)

im_edge = Image.fromarray(edge_map)


# # 分别计算x,y方向上的梯度，Sobel算子
# dx = np.array([-1, 0, 1, -2, 0, 2, -1, 0, 1])
# dy = np.array([-1, -2, -1, 0, 0, 0, 1, 2, 1])
# gx = np.abs(im_edge.filter(ImageFilter.Kernel((3, 3), dx,1,0)).getdata())
# gx_re = gx.reshape(im_edge.size[1],im_edge.size[0])
# gy = np.abs(im_edge.filter(ImageFilter.Kernel((3, 3), dy,1,0)).getdata())
# gy_re = gy.reshape(im_edge.size[1],im_edge.size[0])
# # x,y梯度图像的融合
# gradient = np.sqrt(gx_re**2 + gy_re**2)
# # 融合梯度图像的展示
# gray_image = Image.fromarray(gradient)
# # gray_image.show()
# # 梯度方向
# gradient_direction = np.arctan2(gy_re, gx_re) * 180 / np.pi
# gradient_direction[gradient_direction < 0] += 180
# gradient_direction = np.round(gradient_direction / 45) * 45
#
#
# # 得到单像素边缘信息
# for i in range(1, gradient.shape[0]-1):
#     for j in range(1, gradient.shape[1]-1):
#         if gradient_direction[i][j] == 0:
#             # 梯度方向为0度时左右2列比对
#             left_pixel = gradient[i][j-1]
#             right_pixel = gradient[i][j+1]
#             if gradient[i][j] >= max(left_pixel, right_pixel):
#                 nms[i][j] = gradient[i][j]
#         elif gradient_direction[i][j] == 45:
#             # 梯度方向为45度时2个对角线方向比对
#             up_left_pixel = gradient[i-1][j-1]
#             down_right_pixel = gradient[i+1][j+1]
#             if gradient[i][j] >= max(up_left_pixel, down_right_pixel):
#                 nms[i][j] = gradient[i][j]
#         elif gradient_direction[i][j] == 90:
#             # 梯度方向为90度时上下2行比对
#             up_pixel = gradient[i-1][j]
#             down_pixel = gradient[i+1][j]
#             if gradient[i][j] >= max(up_pixel, down_pixel):
#                 nms[i][j] = gradient[i][j]
#         elif gradient_direction[i][j] == 135:
#             # 梯度方向为135度时左斜向下和右斜向上比对
#             up_right_pixel = gradient[i-1][j+1]
#             down_left_pixel = gradient[i+1][j-1]
#             if gradient[i][j] >= max(up_right_pixel, down_left_pixel):
#                 nms[i][j] = gradient[i][j]
#
# # 双阈值
# high_threshold = np.mean(gradient) * 2
# low_threshold = high_threshold / 2
#
# # 双阈值分割
# # 将大于高阈值的点认为是强边缘置1
# strong_edges = (nms > high_threshold)
# # 介于中间的像素点需进行进一步的检查
# weak_edges = (nms >= low_threshold) & (nms <= high_threshold)
#
# edge_map = strong_edges.astype(np.uint8) * 255 + weak_edges.astype(np.uint8) * 128
#
# # 边缘连接
# def connect_edges(i, j):
#     if edge_map[i][j] == 255:
#         for ii in range(max(0, i - 1), min(edge_map.shape[0], i + 2)):
#             for jj in range(max(0, j - 1), min(edge_map.shape[1], j + 2)):
#                 if edge_map[ii][jj] == 128:
#                     edge_map[ii][jj] = 255
#                     connect_edges(ii, jj)
#
#
# for i in range(edge_map.shape[0]):
#     for j in range(edge_map.shape[1]):
#         if edge_map[i][j] == 255:
#             connect_edges(i, j)
#
# im_edge = Image.fromarray(edge_map)
im_edge.show()

im_source = Image.open('1.jpg').convert('RGB')

img_color = np.array(im_source)

Image.fromarray(img_color).show()

# 应用二值化，将图像转换为二值图像
# thresh = cv2.threshold(edge_map, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]

# 找到图像中所有的轮廓
contours, hierarchy = cv2.findContours(edge_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# 遍历每个轮廓
for cnt in contours:
    # # 如果当前轮廓的面积小于10个像素，则说明该轮廓为内部像素点，需要清除
    # if cv2.contourArea(cnt) < 10:
    #     # 在二值图像上绘制当前轮廓，并用黑色填充
    #     cv2.drawContours(edge_map, [cnt], 0, (0,0,0), -1)
    # # 否则，绘制当前轮廓的外部轮廓，并用白色填充
    # else:
        edge_one = cv2.drawContours(edge_map, cnt, 0, (0 , 0, 255), 1)

for i in range(edge_one.shape[0]):
    for j in range(edge_one.shape[1]):
        if edge_one[i][j] == 255:
            img_color[i][j][0] = 255  # 边缘像素为红色
            img_color[i][j][1] = 0
            img_color[i][j][2] = 0

Image.fromarray(img_color).show()

print(max_area)
print(mean_area)
