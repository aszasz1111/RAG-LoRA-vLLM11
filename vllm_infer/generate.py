from vllm import LLM, SamplingParams

MODEL_PATH = "/path/to/your/model"

llm = LLM(model=MODEL_PATH, trust_remote_code=True)

sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=256
)

prompt = "你好，介绍一下你自己"

outputs = llm.generate([prompt], sampling_params)

print(outputs[0].outputs[0].text)