import os
from datetime import datetime
import json
import requests
import re
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dotenv import load_dotenv

# Load environment variables from a .env file if it exists
load_dotenv()

# Try to import fix-busted-json, fallback to custom implementation if not available
try:
    from fix_busted_json import repair_json, first_json
    HAS_FIX_BUSTED_JSON = True
except ImportError:
    HAS_FIX_BUSTED_JSON = False
    print("Warning: fix-busted-json not installed. Using custom JSON repair implementation.")

# 创建带重试机制的session
def create_retry_session():
    session = requests.Session()
    retry_strategy = Retry(
        total=5,  # 增加总重试次数到5次
        backoff_factor=2,  # 增加重试间隔
        status_forcelist=[429, 500, 502, 503, 504],  # 需要重试的HTTP状态码
        allowed_methods=["HEAD", "GET", "PUT", "DELETE", "OPTIONS", "TRACE", "POST"],  # 允许重试的方法
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


keys = {
    'or': os.getenv("OPENROUTER_API_KEY", ""),
}

api_source_mapping = {
    # 原有模型
    "claude35h": ["or", "anthropic/claude-3.5-haiku"],
    "gem25f": ["or", "google/gemini-2.5-flash"],
    "qwen3-8b": ["or", "qwen/qwen3-8b"],               
    "dsv3": ["or", "deepseek/deepseek-chat-v3-0324"],

    # 新增的六个模型 - 使用正确的OpenRouter模型名称
    "qwen2.5-vl-7b": ["or", "qwen/qwen-2.5-vl-7b-instruct"],
    "qwen2.5-vl-32b": ["or", "qwen/qwen-2.5-vl-32b-instruct"],  # 这个模型可能不可用
    "gemma3-12b": ["or", "google/gemma-2-12b-it"],  # 这个模型可能不可用
    "gemma3-27b": ["or", "google/gemma-2-27b-it"],
    "gemini-2.5-flash": ["or", "google/gemini-2.5-flash"],  # 修正模型名称
    "gpt-4o": ["or", "openai/gpt-4o"],
}


def custom_repair_json(json_string):
    """
    Custom JSON repair implementation when fix-busted-json is not available.
    Handles common JSON malformation issues from LLMs.
    """
    if not json_string or not json_string.strip():
        return None, "Empty or None JSON string"
    
    # Step 1: Extract JSON from text
    start_idx = json_string.find('{')
    if start_idx == -1:
        return None, "No opening brace found"
    
    end_idx = json_string.rfind('}')
    if end_idx == -1:
        return None, "No closing brace found"
    
    json_string = json_string[start_idx:end_idx + 1]
    
    # Step 2: Try direct parsing first
    try:
        return json.loads(json_string), None
    except json.JSONDecodeError:
        pass
    
    # Step 3: Apply common fixes
    try:
        # Fix missing quotes around keys
        json_string = re.sub(r'([{,])\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1 "\2":', json_string)
        
        # Fix single quotes to double quotes
        json_string = re.sub(r"'([^']*)'", r'"\1"', json_string)
        
        # Fix trailing commas
        json_string = re.sub(r',\s*([}\]])', r'\1', json_string)
        
        # Fix missing commas
        json_string = re.sub(r'"\s*"', r'", "', json_string)
        
        # Fix Python True/False/None
        json_string = json_string.replace('True', 'true').replace('False', 'false').replace('None', 'null')
        
        # Try parsing again
        return json.loads(json_string), None
        
    except json.JSONDecodeError as e:
        return None, f"Failed to repair JSON: {str(e)}"


def robust_json_parse(response_text):
    """
    Robustly parse JSON from LLM response, handling malformed JSON.
    """
    if HAS_FIX_BUSTED_JSON:
        try:
            # Try to extract and repair JSON using fix-busted-json
            repaired_json = repair_json(response_text)
            return repaired_json, None
        except Exception as e:
            print(f"fix-busted-json failed: {e}")
            # Fallback to custom implementation
            return custom_repair_json(response_text)
    else:
        return custom_repair_json(response_text)


def safe_json_parse(response_text):
    """
    Safely parse JSON from response, with fallback to text extraction.
    """
    # First try to parse as regular JSON
    try:
        return json.loads(response_text), None
    except json.JSONDecodeError:
        pass
    
    # If that fails, try robust parsing
    parsed_data, error = robust_json_parse(response_text)
    if parsed_data:
        return parsed_data, None
    
    # If all parsing fails, return the raw text
    return response_text, f"JSON parsing failed: {error}"


def call_llm_api(messages, model_name, keys=keys, temperature=0.4, trial_info=None, numtokens=8192):
    """
    Calls a large language model API based on the specified model name.
    Now includes robust JSON parsing for malformed responses and retry mechanism.
    Returns JSON response with reasoning and answer fields.
    """
    if model_name not in api_source_mapping:
        raise ValueError(f"Model '{model_name}' not found in api_source_mapping.")

    api_source, full_model_name = api_source_mapping[model_name]
    
    if not keys.get(api_source):
        raise ValueError(f"API key for source '{api_source}' is not set. Please set the corresponding environment variable.")

    trial_id = trial_info.get('trial_id', "unknown") if trial_info else "unknown"
    reasoning_content = None

    if api_source == "or":
        # 使用重试机制
        session = create_retry_session()
        max_retries = 5  # 增加重试次数
        
        for attempt in range(max_retries):
            try:
                print(f"[Trial {trial_id}] 尝试第 {attempt + 1} 次API调用...")
                
                # 构建请求数据
                request_data = {
                    "model": full_model_name,
                    "messages": messages,
                    "temperature": temperature
                }
                
                # 只有支持的模型才添加JSON格式要求
                json_supported_models = [
                    "qwen/qwen-2.5-vl-7b-instruct",
                    "qwen/qwen-2.5-vl-32b-instruct", 
                    "google/gemma-2-12b-it",
                    "google/gemma-2-27b-it",
                    "google/gemini-2.0-flash-exp",
                    "openai/gpt-4o"
                ]
                
                if full_model_name in json_supported_models:
                    request_data["response_format"] = {"type": "json_object"}
                
                # 添加更长的超时时间和SSL验证设置
                response = session.post(
                    url="https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {keys['or']}",
                        "Content-Type": "application/json",
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                    },
                    data=json.dumps(request_data),
                    timeout=(30, 120),  # 连接超时30秒，读取超时120秒
                    verify=True,  # 启用SSL验证
                    stream=False  # 禁用流式传输
                )
                
                # 检查响应状态
                if response.status_code != 200:
                    print(f"[Trial {trial_id}] HTTP错误: {response.status_code}")
                    print(f"[Trial {trial_id}] 响应内容: {response.text[:500]}")
                    if response.status_code == 400:
                        print(f"[Trial {trial_id}] 400错误可能原因:")
                        print(f"  - 模型不支持JSON格式: {full_model_name}")
                        print(f"  - 请求参数错误")
                        print(f"  - 消息格式问题")
                    response.raise_for_status()
                
                # Try to parse JSON response
                try:
                    data = response.json()
                    # Check if response has expected structure
                    if 'choices' in data and len(data['choices']) > 0:
                        content = data['choices'][0]['message']['content']
                        
                        # Parse the JSON content to extract reasoning and answer
                        try:
                            json_content = json.loads(content)
                            reasoning = json_content.get('reasoning', '')
                            answer = json_content.get('answer', '')
                            
                            # Validate JSON structure
                            if not answer:
                                raise ValueError("Missing 'answer' field in JSON response")
                            
                            tokens = data.get('usage', {}).get('completion_tokens', len(content.split()))
                            print(f"[Trial {trial_id}] API调用成功！返回JSON格式")
                            return {'reasoning': reasoning, 'answer': answer}, None, tokens
                            
                        except json.JSONDecodeError as e:
                            # If the content is not valid JSON, treat it as plain text
                            print(f"[Trial {trial_id}] 警告：API返回的不是有效JSON，尝试解析为纯文本")
                            # Try to extract reasoning and answer using regex
                            reasoning_match = re.search(r'reasoning["\s]*:["\s]*([^"]+)', content, re.IGNORECASE)
                            answer_match = re.search(r'answer["\s]*:["\s]*([^"]+)', content, re.IGNORECASE)
                            
                            reasoning = reasoning_match.group(1) if reasoning_match else ''
                            answer = answer_match.group(1) if answer_match else content
                            
                    tokens = data.get('usage', {}).get('completion_tokens', len(content.split()))
                            return {'reasoning': reasoning, 'answer': answer}, None, tokens
                            
                elif 'content' in data:
                    # Alternative response format with direct 'content' key
                    content = data['content']
                        # Try to parse as JSON
                        try:
                            json_content = json.loads(content)
                            reasoning = json_content.get('reasoning', '')
                            answer = json_content.get('answer', '')
                            if not answer:
                                raise ValueError("Missing 'answer' field in JSON response")
                            tokens = data.get('usage', {}).get('completion_tokens', len(content.split()))
                            return {'reasoning': reasoning, 'answer': answer}, None, tokens
                        except:
                            # Fallback to plain text
                    tokens = data.get('usage', {}).get('completion_tokens', len(content.split()))
                            return {'reasoning': '', 'answer': content}, None, tokens
                            
                elif 'response' in data:
                        # Alternative response format with direct 'response' key
                    content = data['response']
                        try:
                            json_content = json.loads(content)
                            reasoning = json_content.get('reasoning', '')
                            answer = json_content.get('answer', '')
                            if not answer:
                                raise ValueError("Missing 'answer' field in JSON response")
                            tokens = data.get('usage', {}).get('completion_tokens', len(content.split()))
                            return {'reasoning': reasoning, 'answer': answer}, None, tokens
                        except:
                            tokens = data.get('usage', {}).get('completion_tokens', len(content.split()))
                            return {'reasoning': '', 'answer': content}, None, tokens
                    else:
                        # Fallback: return raw response as answer
                        content = str(data)
                    tokens = data.get('usage', {}).get('completion_tokens', len(content.split()))
                        return {'reasoning': '', 'answer': content}, None, tokens
                        
                except json.JSONDecodeError as e:
                    print(f"[Trial {trial_id}] JSON解析失败: {e}")
                    if attempt < max_retries - 1:
                        print(f"[Trial {trial_id}] 重试中...")
                        time.sleep(2 ** attempt)  # Exponential backoff
                        continue
                    else:
                        raise ValueError(f"Failed to parse API response as JSON after {max_retries} attempts")
                        
            except requests.exceptions.SSLError as e:
                print(f"[Trial {trial_id}] SSL错误 (尝试 {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    wait_time = min(2 ** attempt, 30)  # 最大等待30秒
                    print(f"[Trial {trial_id}] 等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)
                    continue
                else:
                    raise e
            except requests.exceptions.ConnectionError as e:
                print(f"[Trial {trial_id}] 连接错误 (尝试 {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    wait_time = min(2 ** attempt, 30)  # 最大等待30秒
                    print(f"[Trial {trial_id}] 等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)
                    continue
                else:
                    raise e
            except requests.exceptions.Timeout as e:
                print(f"[Trial {trial_id}] 超时错误 (尝试 {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    wait_time = min(2 ** attempt, 30)  # 最大等待30秒
                    print(f"[Trial {trial_id}] 等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)
                    continue
                else:
                    raise e
        except requests.exceptions.RequestException as e:
                print(f"[Trial {trial_id}] 请求失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    wait_time = min(2 ** attempt, 30)  # 最大等待30秒
                    print(f"[Trial {trial_id}] 等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)
                    continue
                else:
            raise e

        raise ValueError(f"API调用失败，已重试 {max_retries} 次")
        
    else:
        raise ValueError(f"Unsupported API source: {api_source}")


def extract_content_from_raw_response(raw_text, trial_id="unknown"):
    """
    Extract content from raw response using multiple strategies.
    Returns (content, tokens, method_used)
    """
    # Strategy 1: Try JSON repair
    try:
        parsed_data, error = safe_json_parse(raw_text)
        if parsed_data and isinstance(parsed_data, dict):
            if 'choices' in parsed_data and len(parsed_data['choices']) > 0:
                content = parsed_data['choices'][0]['message']['content']
                tokens = parsed_data.get('usage', {}).get('completion_tokens', len(content.split()))
                print(f"[Trial {trial_id}] ✓ JSON repair successful")
                return content, tokens, "JSON repair"
    except Exception as e:
        pass
    
    # Strategy 2: Try regex extraction with multiple patterns
    patterns = [
        r'"content":\s*"([^"]*)"',
        r'"content":\s*"([^"]*(?:\\"[^"]*)*)"',  # Handle escaped quotes
        r'"content":\s*"([^"]*)"[^}]*"completion_tokens":\s*(\d+)',
        r'"content":\s*"([^"]*)"[^}]*"total_tokens":\s*(\d+)'
    ]
    
    for i, pattern in enumerate(patterns):
        try:
            match = re.search(pattern, raw_text, re.DOTALL)
            if match:
                content = match.group(1)
                # Handle escaped quotes
                content = content.replace('\\"', '"')
                tokens = int(match.group(2)) if len(match.groups()) > 1 else len(content.split())
                print(f"[Trial {trial_id}] ✓ Regex extraction successful (pattern {i+1})")
                return content, tokens, f"Regex pattern {i+1}"
        except Exception as e:
            continue
    
    # Strategy 3: Manual parsing for common response formats
    try:
        # Look for content between quotes after "content":
        content_start = raw_text.find('"content":')
        if content_start != -1:
            # Find the opening quote
            quote_start = raw_text.find('"', content_start + 10)
            if quote_start != -1:
                # Find the closing quote, handling escaped quotes
                quote_end = quote_start + 1
                while quote_end < len(raw_text):
                    if raw_text[quote_end] == '"' and raw_text[quote_end-1] != '\\':
                        break
                    quote_end += 1
                
                if quote_end < len(raw_text):
                    content = raw_text[quote_start+1:quote_end]
                    content = content.replace('\\"', '"')
                    tokens = len(content.split())
                    print(f"[Trial {trial_id}] ✓ Manual parsing successful")
                    return content, tokens, "Manual parsing"
    except Exception as e:
        pass
    
    # Strategy 4: Fallback - return raw text
    print(f"[Trial {trial_id}] ⚠ All extraction methods failed, using raw text")
    return raw_text, len(raw_text.split()), "Raw text fallback"


# ===================================================================
# Main execution block to test all models sequentially
# ===================================================================
if __name__ == '__main__':
    sample_messages = [{"role": "user", "content": "Hi."}]

    print("--- Starting Model Tests ---")
    print("This will iterate through all defined models and test the ones with available API keys.\n")

    # Iterate through every model in the mapping
    for model_name, model_info in api_source_mapping.items():
        api_source = model_info[0]

        # Check if the required API key for this model's source exists
        if not keys.get(api_source):
            # If you want to be notified of which are skipped, uncomment the next line
            print(f"SKIPPING: {model_name} (API key for '{api_source}' is not set)")
            continue

        # If the key exists, try to call the model
        print(f"--- Testing model: {model_name} (Source: {api_source.upper()}) ---")
        try:
            content, reasoning_content, tokens = call_llm_api(sample_messages, model_name)
            print(f"✅ SUCCESS")
            print(f"Response: {content}")
            print(f"Response (Reason): {reasoning_content}")
            print(f"Tokens: {tokens}\n")
        except Exception as e:
            print(f"❌ FAILED: An error occurred while calling model {model_name}.")
            print(f"   Error details: {e}\n")

    print("--- All Model Tests Complete ---")

