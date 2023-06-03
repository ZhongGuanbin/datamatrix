import cv2
import numpy as np
import plotly.graph_objs as go
import plotly.offline as pyo
from scipy.signal import savgol_filter, butter, filtfilt

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

# 对y进行归一化差分
y_diff_norm = np.where(np.diff(y_sub) > 0, 1, np.where(np.diff(y_sub) == 0, 0, -1))

# 将末尾差分值直接设为0，即认为尾点S[n]之后保持不变,以防止漏掉尾峰
y_diff_norm[-1] = 0

# 差分后的横坐标数组
x_diff = x[:-1]

trace = go.Scatter(x=x_diff, y=y_diff_norm, mode='lines', name='Differentiated Normalization')
data = [trace]
layout = go.Layout(title='Differentiated Normalization Plot')
fig = go.Figure(data=data, layout=layout)
pyo.plot(fig, filename='my_plot_diff_norm.html')
