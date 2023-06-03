import cv2
import numpy as np
from scipy.stats import gaussian_kde

# 采用cv2读取路径为D:\datamatrix\2393的文件夹下有一张名为input.png的PNG图片，将其灰度化
image = cv2.imread(r"D:\datamatrix\2393\input.png", cv2.IMREAD_GRAYSCALE)

# 采用cv2进行灰度均衡处理
equalized_image = cv2.equalizeHist(image)

# 采用cv2的双边滤波方法，去噪
denoised_image = cv2.bilateralFilter(equalized_image, 9, 75, 75)

# 采用cv2进行二值化，二值化的阈值用OTSU算法自动计算
ret, binary_image = cv2.threshold(denoised_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# 缩放到400*400的像素空间中
resized_binary_image = cv2.resize(binary_image, (400, 400), interpolation=cv2.INTER_LINEAR)

# 计算每个像素点与右侧一个像素点的灰度差值
Gx = np.abs(np.diff(np.asarray(resized_binary_image), axis=1))

# 统计每一列像素的差值和，最右侧的像素点则与0做差值
col_sum = np.sum(Gx, axis=0)
col_sum = np.append(col_sum, 0)

# 横纵坐标数组
x = np.arange(len(col_sum))
x = np.linspace(0, len(x) - 1, 400)
y = col_sum

# 去基线，y＜8000的认为是噪音，扣除
y_sub = np.where(y < 8000, 0, y)

# 对y进行差分
y_diff = np.diff(y_sub)

# 差分后的横坐标数组
x_diff = x[:-1]

# 对y_diff求符号
y_diff_sign = np.sign(y_diff)

# 对y_diff_sign再求一次差分
y_diff_sign_diff = np.diff(y_diff_sign)

# 遍历y_diff_sign_diff，将其中y_diff_sign_diff=-2或y_diff_sign_diff[i] == -1的i值变为i+1后存储到数组peak_x_list中
peak_x_list = []
for i in range(len(y_diff_sign_diff)):
    if y_diff_sign_diff[i] == -2 or y_diff_sign_diff[i] == -1:
        peak_x_list.append(i + 1)

# 对peak_x_list做差分，查看两个峰值之间的间隔，保存在数组peak_x_diff中
peak_x_diff_origin = np.diff(peak_x_list)

# 剔除小于7的间隔值
peak_x_diff = peak_x_diff_origin[peak_x_diff_origin > 7]

# 计算众数
peak_x_diff_mode = np.argmax(np.bincount(peak_x_diff))

# 给予7到10的间隔值很小的权重
weights = np.ones_like(peak_x_diff)
weights[(peak_x_diff >= 7) & (peak_x_diff <= 10)] = 0.0001

# 根据样本点与众数之间的距离赋予不同的权重
weights = np.exp(-np.abs(peak_x_diff - peak_x_diff_mode) / 2)

# 估算概率密度函数
kde = gaussian_kde(peak_x_diff, weights=weights)
x = np.linspace(peak_x_diff.min(), peak_x_diff.max(), 10000)
y = kde(x)

# 计算概率密度函数的最大值
idx = np.argmax(y)
most_likely_value = x[idx]

# 遍历peak_x_list，记后一项减去前一项的值为x_diff，即做差分，做如下判断:
peak_x_list_find_peak = peak_x_list.copy()

# # 从后往前遍历,避免删除元素后，索引值发生变化而引发IndexError 错误
# for i in range(len(peak_x_list_find_peak) - 2, -1, -1):
#
#     # 记后一项减去前一项的值为x_diff，即做差分
#     x_diff = peak_x_list_find_peak[i + 1] - peak_x_list_find_peak[i]
#
#     # 如果x_diff小于max(5, most_likely_value/3)，则删除后一项
#     if x_diff < max(5, most_likely_value / 3):
#         peak_x_list_find_peak.pop(i + 1)
#
#     # 如果x_diff大于2*most_likely_value，则在两项中间插入一个值。例如两项分别为x,y，则插入两项的中间值（x+y）/2
#     elif x_diff > 2 * most_likely_value:
#         new_val = (peak_x_list_find_peak[i] + peak_x_list_find_peak[i + 1]) / 2
#         peak_x_list_find_peak.insert(i + 1, new_val)

# 遍历peak_x_list_find_peak，将重峰放在一个数组内
peak_x_list_find_peak_group = []
temp_list = [peak_x_list_find_peak[0]]

for i in range(len(peak_x_list_find_peak)-1):
    x_diff = peak_x_list_find_peak[i+1] - peak_x_list_find_peak[i]

    if x_diff <= max(5, most_likely_value/3):
        temp_list.append(peak_x_list_find_peak[i+1])
    else:
        peak_x_list_find_peak_group.append(temp_list)
        temp_list = [peak_x_list_find_peak[i+1]]

# 如果是最后一组峰（单峰），则将其添加到peak_x_list_find_peak_group中，避免漏掉尾峰
if peak_x_list_find_peak[-1] - temp_list[-1] == 0:
    peak_x_list_find_peak_group.append(temp_list)

# 遍历peak_x_list_find_peak_group，找出每组峰中的最大值
peak_x_list_find_peak_group_max = []
for i in range(len(peak_x_list_find_peak_group)):

    # 如果只有一个峰，则直接将其添加到peak_x_list_find_peak_group_max中
    if len(peak_x_list_find_peak_group[i]) == 1:
        peak_x_list_find_peak_group_max.append(peak_x_list_find_peak_group[i][0])

    # 如果有多个峰，则找出最大值所对应的j值
    else:
        peak_y_list = []
        for j in range(len(peak_x_list_find_peak_group[i])):
            peak_y_list.append(y_sub[int(peak_x_list_find_peak_group[i][j])])
        max_y_sub = max(peak_y_list)

        # 如果只有1个max_y_sub，则取最大值所对应的x值
        if peak_y_list.count(max_y_sub) == 1:
            max_y_sub_idx = peak_y_list.index(max_y_sub)
            peak_x_list_find_peak_group_max.append(peak_x_list_find_peak_group[i][max_y_sub_idx])

        # 如果出现了多个峰的y_sub值相同的情况，则取多个y_sub对应的x值的平均值
        else:
            max_y_sub_idx = [i for i, x in enumerate(peak_y_list) if x == max_y_sub]
            temp = 0
            for j in range(len(max_y_sub_idx)):
                temp += peak_x_list_find_peak_group[i][max_y_sub_idx[j]]
            peak_x_list_find_peak_group_max.append(temp/len(max_y_sub_idx))

# # 遍历peak_x_list_find_peak_group，奇数分割线往左，偶数分割线往右
# peak_x_list_find_peak_group_lr = []
# for i in range(len(peak_x_list_find_peak_group)):
#
#     # 如果只有一个峰，则直接将其添加到peak_x_list_find_peak_group_lr中
#     if len(peak_x_list_find_peak_group[i]) == 1:
#         peak_x_list_find_peak_group_lr.append(peak_x_list_find_peak_group[i][0])
#
#     # 如果有多个峰，则判断是奇数还是偶数，奇数选【0】，偶数选【-1】
#     else:
#         if i % 2 == 0:
#             peak_x_list_find_peak_group_lr.append(peak_x_list_find_peak_group[i][0])
#         else:
#             peak_x_list_find_peak_group_lr.append(peak_x_list_find_peak_group[i][-1])

print(peak_x_list_find_peak_group_max)
