import asyncio
import websockets
from io import BytesIO
from PIL import Image

global initial_data


async def handler(websocket):
    async for message in websocket:
        global initial_data
        initial_data = message

        # 使用 BytesIO 将二进制数据转换成可读写的内存流对象
        img_stream = BytesIO(initial_data)

        # 使用 PIL 库的 Image 模块打开内存流中的图片数据
        img = Image.open(img_stream)

        # 将二进制数据发送回客户端
        await websocket.send(message)


async def main():
    async with websockets.serve(handler, "localhost", 8081):
        print("WebSocket server started")
        await asyncio.Future()  # 防止程序退出

if __name__ == '__main__':
    asyncio.run(main())