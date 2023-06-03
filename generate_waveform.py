import asyncio
from io import BytesIO

import cv2
import matplotlib.pyplot as plt
import numpy as np
import websockets
from PIL import Image
from PIL import ImageOps
from scipy.interpolate import make_interp_spline
from scipy.spatial import distance as dist


def generate_waveform(img):
    # 计算每个像素点与右侧一个像素点的灰度差值
    Gx = np.abs(np.diff(np.asarray(img), axis=1))

    # 统计每一列像素的差值和，最右侧的像素点则与0做差值
    col_sum = np.sum(Gx, axis=0)
    col_sum = np.append(col_sum, 0)

    # 统计每一行像素的差值和，最下方的像素点则与0做差值
    Gy = np.abs(np.diff(np.asarray(img), axis=0))
    row_sum = np.sum(Gy, axis=1)
    row_sum = np.append(row_sum, 0)

    # 绘制水平方向拟合曲线
    # fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(8, 8))
    fig = plt.figure(frameon=False)
    ax1 = plt.subplot2grid((3, 3), (2, 0), colspan=2)
    ax2 = plt.subplot2grid((3, 3), (0, 2), rowspan=2)
    ax3 = plt.subplot2grid((3, 3), (0, 0), colspan=2, rowspan=2)
    ax1.set_axis_off()
    ax2.set_axis_off()
    ax3.set_axis_off()
    # ax1.patch.set_alpha(0.0)
    # ax2.patch.set_alpha(1.0)

    # img_ax3 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_rgb = img.convert('RGB')
    img_ax3_temp = np.array(img_rgb)[:, :, ::-1]
    img_ax3 = Image.fromarray(np.uint8(img_ax3_temp))
    ax3.imshow(img_ax3, aspect='auto')

    x = np.arange(len(col_sum))
    y = col_sum
    x_new = np.linspace(0, len(x) - 1, 300)
    spl = make_interp_spline(x, y, k=3)
    y_new = spl(x_new)
    ax1.plot(x_new, y_new, linewidth=1, c='r', alpha=0.5, label='horizontal')

    ax1.set_xlim([0, len(x) - 1])
    # ax1.xaxis.set_tick_params(labelsize=8, rotation=45)

    # ax1.set_ylim([np.min(y)*4, np.max(y)*4])
    ax1.set_ylim([np.min(y), np.max(y)])
    # ax1.yaxis.set_tick_params(labelsize=8)

    ax1.invert_yaxis()

    # ax1.set_xlabel('column index', fontsize=10)
    # ax1.set_ylabel('pixel gradient', fontsize=10)
    # ax1.legend(fontsize=8)

    # 绘制垂直方向拟合曲线
    x = np.arange(len(row_sum))
    y = row_sum
    x_new = np.linspace(0, len(x) - 1, 300)
    spl = make_interp_spline(x, y, k=3)
    y_new = spl(x_new)
    ax2.plot(y_new, x_new, linewidth=1, c='g', alpha=0.5, label='vertical')

    # ax2.set_xlim([np.min(y)*4, np.max(y)*4])
    ax2.set_xlim([np.min(y), np.max(y)])
    # ax2.xaxis.set_tick_params(labelsize=8)

    ax2.set_ylim([len(x) - 1, 0])
    # ax2.yaxis.set_tick_params(labelsize=8)
    #
    # ax2.set_xlabel('pixel gradient', fontsize=10)
    # ax2.set_ylabel('row index', fontsize=10)
    # ax2.legend(fontsize=8)

    # 将两个图形合并
    # ax3 = ax1.twinx()
    # ax4 = ax2.twiny()
    # ax3.set_ylim(ax1.get_ylim())
    # ax4.set_xlim(ax2.get_xlim())

    fig.tight_layout()
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0, hspace=0)
    # 转换图像并返回二进制数据
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf)
    output_buffer = BytesIO()
    img.save(output_buffer, format='png')
    return output_buffer.getvalue()


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
        result_img = cv2.warpPerspective(np.array(ImageOps.grayscale(img)), perspective_matrix, (400, 400))

        # 双线性插值
        result_img = cv2.resize(result_img, (400, 400), interpolation=cv2.INTER_LINEAR)

        # 对透视变换后的图像去噪处理
        denoised_img = cv2.fastNlMeansDenoising(result_img, None, h=10)

        # 对去噪后的图像进行滤波处理
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened_img = cv2.filter2D(denoised_img, -1, kernel)

        # 将滤波后的图像转换为 PIL 图像
        filtered_img = Image.fromarray(sharpened_img)

        img_to_client = generate_waveform(filtered_img)

        # 将图片转换为二进制数据并发送回客户端
        await websocket.send(img_to_client)


async def main():
    async with websockets.serve(handler, "localhost", 8081):
        print("WebSocket server started")
        await asyncio.Future()  # 防止程序退出


if __name__ == '__main__':
    asyncio.run(main())
