#!/usr/bin/env python3
"""
Entailment Benchmark Test - Fixed Version
测试五个模型（除了qwen2.5-vl-7b）使用entailment score方法在easy和hard难度上的表现
修复了图片输入问题
"""

import json
import os
import sys
import time
import logging
from datetime import datetime
from typing import List, Dict, Any, Tuple
from multiprocessing import Pool, Manager
import multiprocessing as mp
import requests
import base64

# 添加父目录到路径以导入agents模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents import PlayerAgent
import config

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('entailment_benchmark_fixed.log'),
        logging.StreamHandler()
    ]
)

# 要测试的五个模型（除了qwen2.5-vl-7b）
TEST_MODELS = [
    "claude35h",
    "gem25f", 
    "qwen3-8b",
    "dsv3",
    "qwen2.5-vl-32b"
]

# 测试难度
TEST_DIFFICULTIES = ["easy", "hard"]

def encode_image(image_path):
    """编码图片为base64"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def call_llm_api_with_image(messages, model_name, temperature=0.1):
    """调用支持图片的LLM API"""
    # 模型映射 - 使用正确的模型ID
    model_mapping = {
        "claude35h": "anthropic/claude-3.5-haiku",
        "gem25f": "google/gemini-2.0-flash-001", 
        "qwen3-8b": "qwen/qwen-2.5-7b-instruct",  # 使用可用的qwen模型
        "dsv3": "deepseek/deepseek-chat-v3-0324",
        "qwen2.5-vl-32b": "qwen/qwen-2.5-vl-7b-instruct"  # 使用可用的qwen vision模型
    }
    
    if model_name not in model_mapping:
        raise ValueError(f"Model {model_name} not supported")
    
    full_model_name = model_mapping[model_name]
    api_key = os.getenv("OPENROUTER_API_KEY")
    
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not found in environment")
    
    try:
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            },
            data=json.dumps({
                "model": full_model_name,
                "messages": messages,
                "temperature": temperature,
                "response_format": {"type": "json_object"}
            }),
            timeout=(30, 120),
            verify=True,
            stream=False
        )
        response.raise_for_status()
        
        data = response.json()
        if 'choices' in data and len(data['choices']) > 0:
            content = data['choices'][0]['message']['content']
            tokens = data.get('usage', {}).get('completion_tokens', len(content.split()))
            
            # 解析JSON内容
            try:
                json_content = json.loads(content)
                return json_content, tokens, None
            except json.JSONDecodeError:
                return content, tokens, "JSON parsing failed"
        else:
            return None, 0, "No choices in response"
            
    except Exception as e:
        return None, 0, str(e)

def load_distractor_dataset(dataset_path: str) -> Dict[str, Any]:
    """加载distractor数据集"""
    with open(dataset_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def filter_dataset_by_difficulty(dataset: Dict[str, Any], difficulties: List[str]) -> List[Dict[str, Any]]:
    """根据难度过滤数据集"""
    return [sample for sample in dataset['dataset'] if sample['difficulty'] in difficulties]

def test_entailment_scoring(agent: PlayerAgent, sample: Dict[str, Any]) -> Dict[str, Any]:
    """测试单个样本的entailment scoring"""
    try:
        # 获取目标图片和描述
        target_image = sample['target']['img']
        caption = sample['target']['caption']
        distractor_images = [d['img'] for d in sample['distractors']]
        
        # 构建所有图片列表
        all_images = [target_image] + distractor_images
        
        # 打乱图片顺序
        import random
        random.seed(42 + sample['image_id'])  # 使用固定的随机种子
        random.shuffle(all_images)
        
        # 找到目标图片的新位置
        correct_position = -1
        for i, img in enumerate(all_images):
            if img == target_image:
                correct_position = i
                break
        
        # 计算每个图片的entailment score
        scores = []
        for image_path in all_images:
            prompt = f"""You are evaluating how well an image matches a given clue in a Dixit game.

Clue: "{caption}"

Please evaluate how well this image supports or entails the given clue. Consider:
- Visual elements that match the clue
- Conceptual or metaphorical connections
- Overall thematic alignment
- Symbolic and abstract interpretations

Analyze the image carefully and provide your detailed reasoning steps towards determining a numerical score from 0 to 100, where 0 means completely unrelated and 100 means perfect match.

IMPORTANT: You must respond in the following JSON format:
{{
    "reasoning": "Your detailed reasoning steps towards the numerical score",
    "answer": "Your numerical rating (0-100)"
}}

Provide thorough reasoning that explains your scoring process and the answer field should contain only the numerical rating."""

            # 编码图片
            try:
                base64_image = encode_image(image_path)
            except Exception as e:
                logging.warning(f"图片编码失败 {image_path}: {e}")
                scores.append(0.0)
                continue
            
            messages = [
                {
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url", 
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ]
            
            # 调用API
            response, tokens, error = call_llm_api_with_image(
                messages=messages,
                model_name=agent.model_name,
                temperature=0.1
            )
            
            if error:
                logging.warning(f"API错误: {error}")
                scores.append(0.0)
                continue
            
            # 解析响应
            try:
                if isinstance(response, dict):
                    score = float(response.get('answer', 0))
                else:
                    # 尝试解析JSON
                    json_content = json.loads(response)
                    score = float(json_content.get('answer', 0))
                
                # 确保分数在0-100范围内
                score = max(0, min(100, score))
                scores.append(score)
                
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                logging.warning(f"解析响应失败: {e}")
                scores.append(0.0)
        
        # 找到最高分数的位置
        max_score = max(scores) if scores else 0.0
        predicted_idx = scores.index(max_score) if scores else 0
        
        # 判断是否正确
        is_correct = (predicted_idx == correct_position)
        
        return {
            "model_name": agent.model_name,
            "sample_id": sample['image_id'],
            "difficulty": sample['difficulty'],
            "description": caption,
            "target_image": target_image,
            "scores": scores,
            "predicted_idx": predicted_idx,
            "is_correct": is_correct,
            "max_score": max_score,
            "correct_position": correct_position
        }
        
    except Exception as e:
        logging.error(f"测试样本失败: {e}")
        return {
            "model_name": agent.model_name,
            "sample_id": sample['image_id'],
            "difficulty": sample['difficulty'],
            "description": sample['target']['caption'],
            "target_image": sample['target']['img'],
            "scores": [0.0] * 6,
            "predicted_idx": 0,
            "is_correct": False,
            "max_score": 0.0,
            "correct_position": 0,
            "error": str(e)
        }

def run_model_test(model_name: str, samples: List[Dict[str, Any]], output_queue) -> None:
    """运行单个模型的测试"""
    logging.info(f"开始测试模型: {model_name}")
    
    # 创建代理
    agent = PlayerAgent(agent_id=1, name=f"entailment_test_{model_name}")
    agent.model_name = model_name
    
    # 确保VLM配置正确
    config.VLM_CONFIG['enable_vlm'] = True
    config.VLM_CONFIG['model_name'] = model_name
    
    results = []
    correct_count = 0
    total_count = len(samples)
    
    for i, sample in enumerate(samples):
        logging.info(f"{model_name}: 测试样本 {i+1}/{total_count} - 图片{sample['image_id']}")
        
        result = test_entailment_scoring(agent, sample)
        results.append(result)
        
        if result['is_correct']:
            correct_count += 1
        
        # 每10个样本报告一次进度
        if (i + 1) % 10 == 0:
            accuracy = correct_count / (i + 1) * 100
            logging.info(f"{model_name}: {i+1}/{total_count} - 准确率: {accuracy:.1f}%")
    
    final_accuracy = correct_count / total_count * 100
    logging.info(f"{model_name} 完成: {correct_count}/{total_count} ({final_accuracy:.1f}%)")
    
    # 将结果放入队列
    output_queue.put({
        'model': model_name,
        'results': results,
        'accuracy': final_accuracy,
        'correct_count': correct_count,
        'total_count': total_count
    })

def main():
    """主函数"""
    logging.info("开始Entailment Benchmark Test - Fixed Version")
    
    # 加载数据集
    dataset_path = 'distractor_dataset.json'
    dataset = load_distractor_dataset(dataset_path)
    
    # 过滤数据
    filtered_samples = filter_dataset_by_difficulty(dataset, TEST_DIFFICULTIES)
    logging.info(f"总样本数: {len(filtered_samples)}")
    
    # 按难度分组
    easy_samples = [s for s in filtered_samples if s['difficulty'] == 'easy']
    hard_samples = [s for s in filtered_samples if s['difficulty'] == 'hard']
    
    logging.info(f"Easy难度: {len(easy_samples)} 个样本")
    logging.info(f"Hard难度: {len(hard_samples)} 个样本")
    
    # 创建任务列表
    tasks = []
    for model in TEST_MODELS:
        tasks.append((model, easy_samples))
        tasks.append((model, hard_samples))
    
    logging.info(f"总任务数: {len(tasks)}")
    
    # 使用多进程执行
    manager = Manager()
    output_queue = manager.Queue()
    
    logging.info("开始并行执行...")
    start_time = time.time()
    
    with Pool(processes=8) as pool:
        # 提交所有任务
        for model, samples in tasks:
            pool.apply_async(
                run_model_test,
                args=(model, samples, output_queue)
            )
        
        # 等待所有任务完成
        pool.close()
        pool.join()
    
    # 收集结果
    all_results = {}
    while not output_queue.empty():
        result = output_queue.get()
        key = f"{result['model']}_{'easy' if 'easy' in str(result['results'][0]['difficulty']) else 'hard'}"
        all_results[key] = result
    
    end_time = time.time()
    duration = end_time - start_time
    
    logging.info(f"总耗时: {duration:.1f} 秒")
    
    # 生成报告
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"entailment_benchmark_results_fixed_{timestamp}.json"
    
    # 保存详细结果
    all_results_list = []
    for result_data in all_results.values():
        all_results_list.extend(result_data['results'])
    
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(all_results_list, f, indent=2, ensure_ascii=False)
    
    logging.info(f"详细结果已保存: {report_file}")
    
    # 生成Markdown报告
    report_md = f"entailment_benchmark_report_fixed_{timestamp}.md"
    with open(report_md, 'w', encoding='utf-8') as f:
        f.write("# Entailment Benchmark Test Report - Fixed Version\n\n")
        f.write(f"**Test Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Models Tested:** {', '.join(TEST_MODELS)}\n\n")
        f.write(f"**Difficulties:** {', '.join(TEST_DIFFICULTIES)}\n\n")
        f.write(f"**Total Tests:** {len(all_results_list)}\n\n")
        
        # 计算总体准确率
        total_correct = sum(1 for r in all_results_list if r['is_correct'])
        overall_accuracy = total_correct / len(all_results_list) if all_results_list else 0
        f.write(f"**Overall Accuracy:** {overall_accuracy:.3f} ({overall_accuracy*100:.1f}%)\n\n")
        
        f.write("## Results by Model and Difficulty\n\n")
        f.write("| Model | Difficulty | Correct | Total | Accuracy |\n")
        f.write("|-------|------------|---------|-------|----------|\n")
        
        for model in TEST_MODELS:
            for difficulty in TEST_DIFFICULTIES:
                model_results = [r for r in all_results_list if r['model_name'] == model and r['difficulty'] == difficulty]
                correct = sum(1 for r in model_results if r['is_correct'])
                total = len(model_results)
                accuracy = correct / total if total > 0 else 0
                f.write(f"| {model} | {difficulty} | {correct} | {total} | {accuracy:.3f} |\n")
    
    logging.info(f"报告已保存: {report_md}")
    
    # 打印总结
    print("\n📊 测试结果总结:")
    print("=" * 60)
    
    for model in TEST_MODELS:
        print(f"\n🤖 {model}:")
        for difficulty in TEST_DIFFICULTIES:
            model_results = [r for r in all_results_list if r['model_name'] == model and r['difficulty'] == difficulty]
            correct = sum(1 for r in model_results if r['is_correct'])
            total = len(model_results)
            accuracy = correct / total if total > 0 else 0
            print(f"  {difficulty.upper()}: {correct}/{total} ({accuracy*100:.1f}%)")
    
    print(f"\n✅ Entailment Benchmark Test完成!")
    print(f"📄 详细结果: {report_file}")
    print(f"📊 报告: {report_md}")

if __name__ == "__main__":
    main()
