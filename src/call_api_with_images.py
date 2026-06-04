import requests
import json
import base64
import os
from together import Together
from openai import OpenAI

keys = {
    "ta": os.getenv("TOGETHER_API_KEY", ""),
    "cp": os.getenv("CP_API_KEY", ""),
    "or": os.getenv("OPENROUTER_API_KEY", ""),
}

def encode_image_to_base64(image_path):
    """将图片编码为base64格式"""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(f"编码图片失败: {e}")
        return None

def call_llm_api_with_images(messages, model_name, keys=keys, temperature=0.0, images=None):
    """
    支持图片的LLM API调用
    messages: 消息列表，可以包含图片
    model_name: 模型名称
    keys: API密钥
    temperature: 温度参数
    images: 图片路径列表（可选）
    """
    
    api_source_mapping = {
        # 视觉语言模型 (支持图片)
        "qwen2.5-vl-7b": ["or", "qwen/qwen-2.5-vl-7b-instruct"],  # 新增qwen7b
        "qwen2.5-vl-32b": ["or", "qwen/qwen-2.5-vl-32b-instruct"],
        "qwen2.5-vl-72b": ["or", "qwen/qwen-2.5-vl-72b-instruct"],
        "gemini-2.5-flash": ["or", "google/gemini-2.0-flash-001"],
        "gpt-4o": ["or", "openai/gpt-4o"],
        "gemma3-12b": ["or", "google/gemma-3-12b-it"],
        "gemma3-27b": ["or", "google/gemma-3-27b-it"],
        
        # 其他模型
        "gpt4o": ["cp","gpt-4o"],
        "gpt35turbo": ["cp","gpt-3.5-turbo"],
        "gpt4omini": ["cp","gpt-4o-mini"],
        "gem15flash": ["or","google/gemini-flash-1.5"],
        "gem15pro": ["or","google/gemini-pro-1.5"],
        "gem20flash": ["or", "google/gemini-2.0-flash-001"],
        "gem20flasht": ["or", "google/gemini-2.0-flash-thinking-exp:free"],
        "dsr1zero": ["or","deepseek/deepseek-r1-zero:free"],
        "dsr1llama70b": ["ta","deepseek-ai/DeepSeek-R1-Distill-Llama-70B"],
        "r18b": ["nv","nvdev/deepseek-ai/deepseek-r1-distill-llama-8b"],
        "o1mini": ["cp","o1-mini"],
        "o3mini": ["cp","o3-mini"],
        "o1": ["cp","o1-2024-12-17"],
        "qwq-32b": ["ta",'Qwen/QwQ-32B'],
        "dsr1": ["nv",'nvdev/deepseek-ai/deepseek-r1'],
        "llama33-70b": ["nv","nvdev/meta/llama-3.3-70b-instruct"],
        "llama31-8b": ["ta","meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"],
        "llama31-70b": ["ta", "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"],
        "llama31-405b": ["ta", "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo"],
        "qwen25-7b": ["ta","Qwen/Qwen2.5-7B-Instruct-Turbo"],
        "qwen25-32b": ["ta","stonetzheng/Qwen/Qwen2.5-32B-Instruct-f553fd65"],
        "qwen25-72b": ["ta","Qwen/Qwen2.5-72B-Instruct-Turbo"],
        "gemma2-9b": ["ta","google/gemma-2-9b-it"],
        "gemma2-27b": ["ta","google/gemma-2-27b-it"],
        "dsv3": ["ta","deepseek-ai/DeepSeek-V3"],
        "dsv3new": ["or","deepseek/deepseek-chat-v3-0324"],
        "mistral-7b": ["nv", "nvdev/mistralai/mistral-7b-instruct-v0.3"],
        "mistral-24b": ["ta", "mistralai/Mistral-Small-24B-Instruct-2501"],
    }
    
    if model_name not in api_source_mapping:
        raise ValueError(f"模型 '{model_name}' 不在映射表中")
    
    api_source, full_model_name = api_source_mapping[model_name]
    
    # 处理图片消息
    processed_messages = []
    if images:
        for i, message in enumerate(messages):
            if isinstance(message, dict) and message.get("role") == "user":
                # 如果有图片，将图片添加到消息中
                if i < len(images) and images[i]:
                    image_path = images[i]
                    if os.path.exists(image_path):
                        base64_image = encode_image_to_base64(image_path)
                        if base64_image:
                            # 创建包含图片的消息
                            content = []
                            if isinstance(message["content"], str):
                                content.append({"type": "text", "text": message["content"]})
                            elif isinstance(message["content"], list):
                                content = message["content"]
                            
                            content.append({
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}"
                                }
                            })
                            
                            processed_messages.append({
                                "role": message["role"],
                                "content": content
                            })
                        else:
                            processed_messages.append(message)
                    else:
                        processed_messages.append(message)
                else:
                    processed_messages.append(message)
            else:
                processed_messages.append(message)
    else:
        processed_messages = messages
    
    while True:
        try:
            if api_source == "cp":
                Skey = keys['cp']
                url = "https://api.claude-Plus.top" + "/v1/chat/completions"
                headers = {
                    'Accept': 'application/json',
                    'Authorization': f'Bearer {Skey}',
                    'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
                    'Content-Type': 'application/json'
                }  
                payload = json.dumps({
                    "model": full_model_name,
                    "messages": processed_messages,
                    "temperature": temperature,
                })
                response = requests.request("POST", url, headers=headers, data=payload)
                data = response.json()
                print(f"[{model_name}] 响应: {data}")

                content = data['choices'][0]['message']['content']
                tokens = data['usage']['completion_tokens']
            
            elif api_source == "or":
                response = requests.post(
                    url="https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {keys['or']}",
                        "Content-Type": "application/json",
                        "HTTP-Referer": "http://localhost:3000",
                        "X-Title": "Dixit VLM Test",
                    },
                    data=json.dumps({
                        "model": full_model_name,
                        "messages": processed_messages,
                        "temperature": temperature,
                        "response_format": {"type": "json_object"}  # 强制JSON格式
                    })
                )
                data = response.json()
                print(f"[{model_name}] 响应: {data}")
                
                if 'choices' in data and len(data['choices']) > 0:
                    content = data['choices'][0]['message']['content']
                    tokens = data.get('usage', {}).get('completion_tokens', 0)
                else:
                    raise Exception(f"API响应格式错误: {data}")

            elif api_source == "ta":
                stopwords = {
                    "ll": ["<|eot_id|>","<|eom_id|>"],
                    "qw": ["<|im_end|>","<|endoftext|>"],
                    "ge": ["<eos>","<end_of_turn>"],
                    "ds": ["<｜end▁of▁sentence｜>"],
                    "mi": ["[/INST]","</s>"],
                }
                
                stopword = stopwords.get(model_name[:2], [])
                
                ta_client = Together(api_key=keys['ta'])
                response = ta_client.chat.completions.create(
                    model=full_model_name,
                    messages=processed_messages,
                    max_tokens=4096 if model_name[:2] == "ge" else None,
                    temperature=temperature,
                    top_p=0.9,
                    repetition_penalty=1,
                    stop=stopword,
                    stream=False
                )
                data = json.loads(response.model_dump_json())
                print(f"[{model_name}] 响应: {data}")
                content = data['choices'][0]['message']['content']
                tokens = data['usage']['completion_tokens']
            
            elif api_source == "nv":
                nvidia_api_key = os.getenv("NVIDIA_API_KEY", "")

                client = OpenAI(
                    base_url = "https://integrate.api.nvidia.com/v1",
                    api_key = nvidia_api_key
                )

                completion = client.chat.completions.create(
                    model=full_model_name,
                    messages=processed_messages,
                    temperature=temperature,
                    top_p=0.7,
                    stream=False
                )
                print(f"[{model_name}] 响应: {completion.model_dump_json()}")
                data = json.loads(completion.model_dump_json())
                content = data['choices'][0]['message']['content']
                tokens = data['usage']['completion_tokens']
            
            return content, tokens
            
        except Exception as e:
            print(f"[{model_name}] 错误: {e}")
            continue

def test_qwen7b_with_images():
    """测试qwen7b的图片处理能力"""
    print("🧪 测试qwen2.5-vl-7b图片处理...")
    
    # 测试消息
    messages = [
        {
            "role": "user", 
            "content": "请以JSON格式回复，包含reasoning和answer字段。请描述这张图片的内容，answer字段填写'Image processed'"
        }
    ]
    
    # 测试图片
    test_image = "images/1.png"  # 使用第一张图片
    
    if os.path.exists(test_image):
        try:
            content, tokens = call_llm_api_with_images(
                messages=messages,
                model_name="qwen2.5-vl-7b",
                images=[test_image],
                temperature=0.1
            )
            
            print(f"✅ 成功处理图片!")
            print(f"   响应: {content}")
            print(f"   Tokens: {tokens}")
            
            # 尝试解析JSON
            try:
                result = json.loads(content)
                print(f"   Reasoning: {result.get('reasoning', 'N/A')}")
                print(f"   Answer: {result.get('answer', 'N/A')}")
            except:
                print("   响应不是有效JSON格式")
                
        except Exception as e:
            print(f"❌ 处理失败: {e}")
    else:
        print(f"❌ 测试图片不存在: {test_image}")

if __name__ == "__main__":
    test_qwen7b_with_images()
