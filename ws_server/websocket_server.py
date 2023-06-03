# --coding:utf-8--
# 创建websocket服务端，接收数据流处理后返回前端

import asyncio
import websockets


class WebSocketServer:
    """ 输入url和端口号创建websocket服务端,
    数据处理另开线程
    """

    def __int__(self, ws_handler, host: str = 'localhost', port: int = 8081):
        self.ws_handler = ws_handler
        self.host = host
        self.port = port

    async def start(self):
        async with websockets.serve(self.ws_handler, 'localhost', 8081):
            await asyncio.Future()
