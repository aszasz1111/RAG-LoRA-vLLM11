from fastapi import FastAPI
from pydantic import BaseModel
from vllm import LLM, SamplingParams

app = FastAPI()

# ====== 1. 加载模型 ======
# 这里填你的模型路径（本地 or HF）
MODEL_PATH = "/path/to/your/model"

llm = LLM(
    model=MODEL_PATH,
    trust_remote_code=True
)

# ====== 2. 请求格式 ======
class Request(BaseModel):
    prompt: str
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9

# ====== 3. 推理接口 ======
@app.post("/generate")
def generate(req: Request):
    sampling_params = SamplingParams(
        temperature=req.temperature,
        top_p=req.top_p,
        max_tokens=req.max_tokens
    )

    outputs = llm.generate([req.prompt], sampling_params)

    return {
        "response": outputs[0].outputs[0].text
    }

# ====== 4. 启动入口 ======
# uvicorn inference.vllm_server:app --host 0.0.0.0 --port 8000