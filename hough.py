import asyncio
import websockets
from io import BytesIO
from PIL import Image, ImageOps, ImageDraw
import numpy as np
import cv2


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

        # 使用 Hough 变换查找 L 边
        edges = []
        minLineLength = 300
        maxLineGap = 5
        img_np = np.array(binary_img)
        lines = cv2.HoughLinesP(img_np, rho=1, theta=np.pi / 180, threshold=100, minLineLength=minLineLength,
                                maxLineGap=maxLineGap)

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if abs(x2 - x1) < abs(y2 - y1):
                    edges.append(line[0])

        # 在二值化图像上绘制 L 边
        l_img = binary_img.copy()
        draw = ImageDraw.Draw(l_img)
        for edge in edges:
            x1, y1, x2, y2 = edge
            draw.line((x1, y1, x2, y2), fill=0, width=2)

        # 将二值化图像和 L 边的处理结果显示
        binary_img.show()
        l_img.show()

        # 将二进制数据发送回客户端
        img_byte_arr = BytesIO()
        l_img.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        await websocket.send(img_byte_arr)


async def main():
    async with websockets.serve(handler, "localhost", 8081):
        print("WebSocket server started")
        await asyncio.Future()  # 防止程序退出


if __name__ == '__main__':
    asyncio.run(main())
