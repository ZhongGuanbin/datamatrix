from PIL import Image, ImageOps, ImageFilter

# 打开图像
image = Image.open(r"D:\datamatrix\input.png")

# 将图像转换为灰度图像
gray_image = image.convert('L')

# 灰度均衡处理
equalized_image = ImageOps.equalize(gray_image)

# 采用5*5的方形窗口进行中值滤波，去噪
# denoised_image = equalized_image.filter(ImageFilter.MedianFilter(13))

# 采用高斯滤波，去噪
denoised_image = equalized_image.filter(ImageFilter.GaussianBlur(9))

# 采用最大类间方差算法进行二值化
binary_image = denoised_image.point(lambda x: 0 if x < 128 else 255, '1')

# 显示图像
# binary_image.show()

# 保存图像
binary_image.save(r"D:\datamatrix\output\output-ga-9-otsu.png")
