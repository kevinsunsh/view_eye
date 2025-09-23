import os
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from view import handle_position, label_objects
load_dotenv()

app = FastAPI(
    title="View Eye Service",
    description="基于View Eye的异步任务执行服务",
    version="1.0.0"
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Hello, World!"}

# /v1/ping 
@app.get("/v1/ping")
async def ping():
    return {"message": "pong"}

@app.post("/v1/view_eye")
async def view_eye(request: Request):
    body = await request.json()
    chat_id = body.get("chat_id", "")
    user_id = body.get("user_id", "")
    position = handle_position(user_id, chat_id)
    return {"message": "success", "position": position}

if __name__ == "__main__":
    # 端口优先级：BYTEFAAS_FUNC_PORT > PORT > SERVER_PORT > 8000
    port_str = os.getenv("BYTEFAAS_FUNC_PORT") or os.getenv("PORT") or os.getenv("SERVER_PORT") or "8000"
    try:
        port = int(port_str)
    except ValueError:
        port = 8000
    log_level = (os.getenv("LOG_LEVEL", "info").lower())
    print(f"Starting uvicorn on 0.0.0.0:{port} (LOG_LEVEL={log_level})")
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=port,
        log_level=log_level,
        access_log=True
    )
