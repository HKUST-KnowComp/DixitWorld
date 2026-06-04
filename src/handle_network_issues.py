# -*- coding: utf-8 -*-
"""
网络问题处理脚本
当遇到SSL或连接错误时，提供自动恢复机制
"""

import time
import random
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

def create_robust_session():
    """创建更健壮的session"""
    session = requests.Session()
    
    # 更激进的重试策略
    retry_strategy = Retry(
        total=10,  # 总重试次数
        backoff_factor=3,  # 重试间隔倍数
        status_forcelist=[429, 500, 502, 503, 504, 520, 521, 522, 523, 524],
        allowed_methods=["HEAD", "GET", "PUT", "DELETE", "OPTIONS", "TRACE", "POST"],
        raise_on_status=False  # 不立即抛出状态码异常
    )
    
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    # 设置更长的超时时间
    session.timeout = (30, 180)  # 连接30秒，读取180秒
    
    return session

def safe_api_call(messages, model_name, max_attempts=5):
    """安全的API调用，带有多重错误处理"""
    from call_api_openrouter import api_source_mapping, keys
    
    if model_name not in api_source_mapping:
        raise ValueError(f"Model '{model_name}' not found in api_source_mapping.")
    
    api_source, full_model_name = api_source_mapping[model_name]
    
    for attempt in range(max_attempts):
        try:
            print(f"🔄 尝试第 {attempt + 1}/{max_attempts} 次API调用...")
            
            session = create_robust_session()
            
            # 添加随机延迟，避免请求过于频繁
            if attempt > 0:
                delay = random.uniform(1, 5) + (attempt * 2)
                print(f"⏳ 等待 {delay:.1f} 秒...")
                time.sleep(delay)
            
            response = session.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {keys['or']}",
                    "Content-Type": "application/json",
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
                },
                json={
                    "model": full_model_name,
                    "messages": messages,
                    "temperature": 0.7,
                    "response_format": {"type": "json_object"}
                },
                timeout=(30, 180),
                verify=True
            )
            
            response.raise_for_status()
            
            # 解析响应
            data = response.json()
            if 'choices' in data and len(data['choices']) > 0:
                content = data['choices'][0]['message']['content']
                
                try:
                    import json
                    json_content = json.loads(content)
                    reasoning = json_content.get('reasoning', '')
                    answer = json_content.get('answer', '')
                    
                    if not answer:
                        raise ValueError("Missing 'answer' field in JSON response")
                    
                    print(f"✅ API调用成功！")
                    return {'reasoning': reasoning, 'answer': answer}, None, data.get('usage', {}).get('completion_tokens', 0)
                    
                except json.JSONDecodeError:
                    # 如果JSON解析失败，返回原始内容
                    return {'reasoning': '', 'answer': content}, None, data.get('usage', {}).get('completion_tokens', 0)
            else:
                raise ValueError("Invalid response format")
                
        except requests.exceptions.SSLError as e:
            print(f"🔒 SSL错误 (尝试 {attempt + 1}): {e}")
            if attempt < max_attempts - 1:
                continue
            else:
                raise e
                
        except requests.exceptions.ConnectionError as e:
            print(f"🌐 连接错误 (尝试 {attempt + 1}): {e}")
            if attempt < max_attempts - 1:
                continue
            else:
                raise e
                
        except requests.exceptions.Timeout as e:
            print(f"⏰ 超时错误 (尝试 {attempt + 1}): {e}")
            if attempt < max_attempts - 1:
                continue
            else:
                raise e
                
        except requests.exceptions.RequestException as e:
            print(f"📡 请求错误 (尝试 {attempt + 1}): {e}")
            if attempt < max_attempts - 1:
                continue
            else:
                raise e
                
        except Exception as e:
            print(f"❌ 未知错误 (尝试 {attempt + 1}): {e}")
            if attempt < max_attempts - 1:
                continue
            else:
                raise e
    
    raise ValueError(f"API调用失败，已重试 {max_attempts} 次")

def test_network_connectivity():
    """测试网络连接性"""
    print("🔍 测试网络连接性...")
    
    test_urls = [
        "https://www.google.com",
        "https://www.github.com", 
        "https://openrouter.ai"
    ]
    
    for url in test_urls:
        try:
            response = requests.get(url, timeout=10)
            print(f"✅ {url} - 状态码: {response.status_code}")
        except Exception as e:
            print(f"❌ {url} - 错误: {e}")

def main():
    """主函数"""
    print("🛠️ 网络问题处理工具")
    print("=" * 40)
    
    # 测试网络连接
    test_network_connectivity()
    
    print("\n💡 如果遇到SSL错误，可以尝试以下方法:")
    print("1. 运行网络诊断: python network_diagnostic.py")
    print("2. 更新网络库: pip install --upgrade requests urllib3")
    print("3. 检查防火墙设置")
    print("4. 尝试使用VPN")
    print("5. 重启网络适配器")

if __name__ == "__main__":
    main()
