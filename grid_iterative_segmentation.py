import numpy as np
import cv2


# 定义网格迭代分割函数，返回分割结果——一个存储着每一层单位化灰度值的列表
# 若传入的是网格列表，则返回一组单位化灰度值的列表
def grid_iterative_segmentation(grid_points_list_input, binary_image_input):
    # 定义存储一组单位化灰度值列表的数组
    gray_value_list_array = []

    # 遍历网格列表，对每一个网格进行迭代分割
    for grid_points in grid_points_list_input:
        # 定义存储每一层单位化灰度值的列表
        gray_value_list = []

        # 获取网格的四个顶点坐标，可以获得该网格的长和宽
        grid_length = grid_points[1][0] - grid_points[0][0]
        grid_width = grid_points[2][1] - grid_points[0][1]

        # 迭代分割网格，直到网格的长grid_length<=2或者宽grid_width<=2
        while grid_length > 2 and grid_width > 2:
            # 首先计算当前四个顶点连成的四边形的单位化灰度值
            # 即累加四个顶点连成的四边形内每个像素点的灰度值，然后除以四边形面积
            # 然后将该单位化灰度值存储到gray_value_list中
            mask = np.zeros_like(binary_image_input)
            cv2.fillPoly(mask, [np.array([grid_points])], 255)
            gray_value = np.mean(binary_image_input[mask > 0])
            gray_value_list.append(gray_value)

            # 再然后将网格的长和宽都-2，得到新的网格，计算新的网格的单位化灰度值，添加到gray_value_list中
            # 重复上述过程，直到网格的长grid_length<=2或者宽grid_width<=2，
            # 返回gray_value_list

            # 更新网格的长和宽
            grid_length -= 2
            grid_width -= 2

            # 重新计算网格的四个顶点坐标
            grid_points = [(grid_points[0][0] + 1, grid_points[0][1] + 1),
                           (grid_points[1][0] - 1, grid_points[1][1] + 1),
                           (grid_points[2][0] + 1, grid_points[2][1] - 1),
                           (grid_points[3][0] - 1, grid_points[3][1] - 1)]

        # 将gray_value_list添加到gray_value_list_array中
        gray_value_list_array.append(gray_value_list)

    # 返回gray_value_list_array
    print(gray_value_list_array)
    return gray_value_list_array

