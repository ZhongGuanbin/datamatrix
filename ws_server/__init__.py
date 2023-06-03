import asyncio

from ws_server.websocket_server import WebSocketServer


async def init_ws_handler(websocket):
    data = await websocket.recv()
    await websocket.send(data)


if __name__ == '__main__':
    websocket_server = WebSocketServer()
    websocket_server.ws_handler = init_ws_handler
    asyncio.run(websocket_server.start())
