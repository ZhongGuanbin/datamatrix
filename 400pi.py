import asyncio
import websockets
from io import BytesIO
from PIL import Image, ImageOps, ImageDraw
import numpy as np
import cv2
from scipy.spatial import distance as dist


async def handler(websocket):
    async for message in websocket:
        initial_data = message

        # 使用 BytesIO 将二进制数据转换成可读写的内存流对象
        img_stream = BytesIO(initial_data)

        # 使用 PIL 库的 Image 模块打开内存流中的图片数据
        img = Image.open(img_stream)

        # 将图像转换为灰度图
        gray_img = ImageOps.grayscale(img)

        # 对灰度图进行二值化处理
        binary_img = gray_img.point(lambda x: 0 if x < 128 else 255)

        # 查找边缘并返回边缘像素点坐标
        edge_coords = np.argwhere(np.array(binary_img) == 0)

        # 寻找 DataMatrix 二维码的四个角点
        contours, _ = cv2.findContours((np.array(binary_img) == 0).astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            continue
        cnt = max(contours, key=cv2.contourArea)

        # 找到最小的矩形，该矩形可能有方向 angle
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        # 确保顺序为左上，右上，右下，左下
        x_sorted = box[np.argsort(box[:, 0]), :]
        left_most = x_sorted[:2, :]
        right_most = x_sorted[2:, :]
        left_most = left_most[np.argsort(left_most[:, 1]), :]
        (tl, bl) = left_most
        D = dist.cdist(tl[np.newaxis], right_most, "euclidean")[0]
        (br, tr) = right_most[np.argsort(D)[::-1], :]
        top_left, top_right, bottom_right, bottom_left = tl, tr, br, bl

        # 透视变换
        src_points = np.float32([top_left, top_right, bottom_right, bottom_left])
        dst_points = np.float32([[0, 0], [400, 0], [400, 400], [0, 400]])
        perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        result_img = cv2.warpPerspective(np.array(img), perspective_matrix, (400, 400))

        # 双线性插值
        result_img = cv2.resize(result_img, (400, 400), interpolation=cv2.INTER_LINEAR)

        # 对透视变换后的图像去噪处理
        denoised_img = cv2.fastNlMeansDenoising(result_img, None, h=10)

        # 对去噪后的图像进行滤波处理
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened_img = cv2.filter2D(denoised_img, -1, kernel)

        # 将滤波后的图像转换为 PIL 图像
        filtered_img = Image.fromarray(sharpened_img)

        # 将图片转换为二进制数据并发送回客户端
        img_byte_arr = BytesIO()
        filtered_img.save(img_byte_arr, format='JPEG')
        await websocket.send(img_byte_arr.getvalue())


async def main():
    async with websockets.serve(handler, "localhost", 8081):
        print("WebSocket server started")
        await asyncio.Future()  # 防止程序退出


if __name__ == '__main__':
    asyncio.run(main())
