import cv2

# 采用cv2读取路径为D:\datamatrix\2393的文件夹下有一张名为input.png的PNG图片，将其灰度化
image = cv2.imread(r"D:\datamatrix\2393\input.png", cv2.IMREAD_GRAYSCALE)

# 采用cv2进行灰度均衡处理
equalized_image = cv2.equalizeHist(image)

# 采用cv2的中值滤波方法，5*5的方形窗口，去噪
# denoised_image = cv2.medianBlur(equalized_image, 1)

# 采用cv2的高斯滤波方法，去噪
# denoised_image = cv2.GaussianBlur(equalized_image, (15, 15), 0)

# 采用cv2的双边滤波方法，去噪
denoised_image = cv2.bilateralFilter(equalized_image, 9, 75, 75)

# 采用cv2的均值滤波方法，去噪
# denoised_image = cv2.blur(equalized_image, (9, 9))

# 采用cv2的形态学去噪方法，去噪
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 13))
# denoised_image = cv2.morphologyEx(equalized_image, cv2.MORPH_OPEN, kernel)

# 采用cv2的非局部均值去噪方法，去噪
# denoised_image = cv2.fastNlMeansDenoising(equalized_image, None, 29, 7, 21)

# 采用cv2的K近邻滤波方法，去噪
# denoised_image = cv2.fastNlMeansDenoisingColored(equalized_image, None, 10, 10, 7, 21)

# 采用cv2进行二值化，二值化的阈值用OTSU算法自动计算
ret, binary_image = cv2.threshold(denoised_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# 显示图像
# cv2.imshow("binary_image", binary_image)

# 保存图像
cv2.imwrite(r"D:\datamatrix\2393\output\output-bilateral-cv-9-otsu.png", binary_image)
