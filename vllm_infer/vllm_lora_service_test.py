# ===================== 1. 查看 vLLM 服务日志（排查启动失败） =====================
!tail -n 100 /content/vllm_server.log

# ===================== 2. 测试服务连通性（验证vLLM是否存活） =====================
import requests
# 访问vLLM的OpenAI兼容接口，获取模型列表
r = requests.get("http://127.0.0.1:8000/v1/models")
print("状态码：", r.status_code)  # 200=服务正常
print("模型列表：", r.json())     # 能看到weather_lora=LoRA加载成功

# ===================== 3. 单轮对话测试（核心功能验证） =====================
from openai import OpenAI

# 连接本地vLLM服务（OpenAI兼容格式）
client = OpenAI(
    api_key="EMPTY",        # vLLM无需真实key，填EMPTY即可
    base_url="http://127.0.0.1:8000/v1"
)

# 调用你的LoRA模型：天气领域信息提取
response = client.chat.completions.create(
    model="weather_lora",   # 必须和你启动时的lora-modules名称一致
    messages=[
        {
            "role": "user",
            "content": "请从以下资料中提取研究主题、研究对象、数据资料、研究方法、核心结论：某研究利用双偏振天气雷达资料，对弱垂直风切变背景下的局地微下击暴流过程进行分析，重点研究KDP核、ZDR柱和ZDR槽的演变特征。结果表明，融化层附近KDP核增强和ZDR柱减弱下降，对下沉气流爆发具有一定指示意义。"
        }
    ],
    temperature=0.1,        # 低温度=输出更稳定
    max_tokens=512          # 最大生成长度
)

print("\n===== 模型输出结果 =====")
print(response.choices[0].message.content)

# ===================== 4. 串行性能测试（连续请求5次，测单请求速度） =====================
import time

test_prompt = """
请从以下资料中提取研究主题、研究对象、数据资料、研究方法、核心结论：
某研究利用双偏振天气雷达资料，对弱垂直风切变背景下的局地微下击暴流过程进行分析，
重点研究KDP核、ZDR柱和ZDR槽的演变特征。
结果表明，融化层附近KDP核增强和ZDR柱减弱下降，对下沉气流爆发具有一定指示意义。
"""

times = []
print("\n===== 串行性能测试 =====")
for i in range(5):
    start = time.time()
    response = client.chat.completions.create(
        model="weather_lora",
        messages=[{"role": "user", "content": test_prompt}],
        temperature=0.1,
        max_tokens=512
    )
    elapsed = time.time() - start
    times.append(elapsed)
    print(f"第 {i+1} 次耗时：{elapsed:.3f} 秒")

print(f"\n平均响应耗时：{sum(times)/len(times):.3f} 秒")

# ===================== 5. 并发性能测试（3路同时请求，测vLLM高并发能力） =====================
from concurrent.futures import ThreadPoolExecutor

def send_request(i):
    start = time.time()
    response = client.chat.completions.create(
        model="weather_lora",
        messages=[{"role": "user", "content": test_prompt}],
        temperature=0.1,
        max_tokens=512
    )
    elapsed = time.time() - start
    return i, elapsed, response.choices[0].message.content[:30]

print("\n===== 并发性能测试（3路并发） =====")
start_all = time.time()
with ThreadPoolExecutor(max_workers=3) as executor:
    results = list(executor.map(send_request, range(1, 4)))
total_elapsed = time.time() - start_all

for i, elapsed, preview in results:
    print(f"请求 {i} 耗时：{elapsed:.3f} 秒 | 输出预览：{preview}...")

print(f"\n3 路并发总耗时：{total_elapsed:.3f} 秒")
