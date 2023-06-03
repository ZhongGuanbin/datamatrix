# 定义网格划分函数
def grid_division(peak_list_x, peak_list_y, is_point=False):
    # 存储列信息，即每列的左右端点坐标
    column_list = []

    # 遍历peak_list_x
    for i in range(len(peak_list_x) - 1):
        # 暂存每列的左右端点坐标
        temp_list = [peak_list_x[i], peak_list_x[i + 1]]

        # 将暂存的每列的左右端点坐标添加到column_list中
        column_list.append(temp_list)

    # 存储行信息，即每行的上下端点坐标
    row_list = []

    # 遍历peak_list_y
    for i in range(len(peak_list_y) - 1):
        # 暂存每行的上下端点坐标
        temp_list = [peak_list_y[i], peak_list_y[i + 1]]

        # 将暂存的每行的上下端点坐标添加到row_list中
        row_list.append(temp_list)

    # 如果is_point为True，则说明是点状二维码
    # 点状二维码只取奇数行和奇数列，删除column_list和row_list的偶数行和偶数列
    if is_point:
        # 删除column_list的偶数列, 即索引为奇数的列
        column_list = column_list[::2]

        # 删除row_list的偶数行, 即索引为奇数的行
        row_list = row_list[::2]

    # 定义矩阵grid_points，存储网格划分后各网格的四顶点坐标
    grid_points = []

    # 遍历行
    for i in range(len(row_list)):
        # 记录每一行的宽度
        row_width = row_list[i][1] - row_list[i][0]

        # 遍历列
        for j in range(len(column_list)):
            # 记录每一列的宽度
            column_width = column_list[j][1] - column_list[j][0]

            # 计算每个网格的四个顶点坐标
            # 左上角顶点坐标
            left_top = (column_list[j][0], row_list[i][0])

            # 右上角顶点坐标
            right_top = (column_list[j][1], row_list[i][0])

            # 左下角顶点坐标
            left_bottom = (column_list[j][0], row_list[i][1])

            # 右下角顶点坐标
            right_bottom = (column_list[j][1], row_list[i][1])

            # 将每个网格的四个顶点坐标存储到grid_points中
            grid_points.append([left_top, right_top, left_bottom, right_bottom])

    # 返回网格划分后各网格的四顶点坐标
    return grid_points
