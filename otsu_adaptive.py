import cv2

# 采用cv2读取路径为D:\datamatrix\2393的文件夹下有一张名为input.png的PNG图片，将其灰度化
image = cv2.imread(r"D:\datamatrix\2393\input.png", cv2.IMREAD_GRAYSCALE)

# 采用cv2进行灰度均衡处理
equalized_image = cv2.equalizeHist(image)

# 采用cv2的双边滤波方法，去噪
denoised_image = cv2.bilateralFilter(equalized_image, 9, 75, 75)

# 应用OTSU算法来计算整个图像的阈值
otsu_thresh, _ = cv2.threshold(denoised_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# 应用自适应阈值分割算法来计算每个子块的阈值
adaptive_thresh = cv2.adaptiveThreshold(denoised_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

# 将得到的每个子块的阈值进行平均，得到最终的阈值
final_thresh = (otsu_thresh + adaptive_thresh.mean()) / 2

# 使用cv2.threshold()函数将整个图像进行二值化处理，得到最终的二值图像
ret, binary_image = cv2.threshold(denoised_image, final_thresh, 255, cv2.THRESH_BINARY)

# 保存图像
cv2.imwrite(r"D:\datamatrix\2393\output\test.png", binary_image)
