#!/usr/bin/env python3
"""
简单的Entailment测试 - 使用agents.py中的PlayerAgent
"""

import json
import os
import time
import logging
from datetime import datetime
from multiprocessing import Pool, Manager
import multiprocessing as mp

from agents import PlayerAgent
import config

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('entailment_test_simple.log'),
        logging.StreamHandler()
    ]
)

# 要测试的六个模型
TEST_MODELS = [
    "gpt-4o",
    "gemma3-12b",
    "qwen2.5-vl-32b",
    "gemini-2.5-flash",
    "gemma3-27b",
    "qwen2.5-vl-7b"
]

# 测试难度
TEST_DIFFICULTIES = ["easy", "hard"]

def load_distractor_dataset(dataset_path: str):
    """加载distractor数据集"""
    with open(dataset_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def filter_dataset_by_difficulty(dataset, difficulties):
    """根据难度过滤数据集"""
    return [sample for sample in dataset['dataset'] if sample['difficulty'] in difficulties]

def test_entailment_sample(sample, model_name, shuffle_seed=None):
    """测试单个样本的entailment scoring"""
    try:
        # 创建代理
        agent = PlayerAgent(agent_id=1, name=f"entailment_test_{model_name}")
        agent.model_name = model_name
        
        # 确保VLM配置正确
        config.VLM_CONFIG['enable_vlm'] = True
        config.VLM_CONFIG['model_name'] = model_name
        
        # 获取目标图片和描述
        target_image = sample['target']['img']
        caption = sample['target']['caption']
        distractor_images = [d['img'] for d in sample['distractors']]
        
        # 构建所有图片列表
        all_images = [target_image] + distractor_images
        
        # 打乱图片顺序
        import random
        if shuffle_seed is not None:
            random.seed(shuffle_seed)
        random.shuffle(all_images)
        
        # 找到目标图片的新位置
        correct_position = -1
        for i, img in enumerate(all_images):
            if img == target_image:
                correct_position = i
                break
        
        # 使用agents.py中的compute_entailment_scores方法
        scores = agent.compute_entailment_scores(caption, all_images)
        
        # 找到最高分数的位置
        max_score = max(scores) if scores else 0.0
        predicted_idx = scores.index(max_score) if scores else 0
        
        # 判断是否正确
        is_correct = (predicted_idx == correct_position)
        
        return {
            "model_name": model_name,
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
            "model_name": model_name,
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

def run_model_test(model_name, samples, output_queue):
    """运行单个模型的测试"""
    logging.info(f"开始测试模型: {model_name}")
    
    results = []
    correct_count = 0
    total_count = len(samples)
    
    for i, sample in enumerate(samples):
        logging.info(f"{model_name}: 测试样本 {i+1}/{total_count} - 图片{sample['image_id']}")
        
        result = test_entailment_sample(sample, model_name, shuffle_seed=42 + sample['image_id'])
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
    logging.info("开始Entailment Test - Simple Version")
    
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
    
    with Pool(processes=4) as pool:  # 减少进程数避免API限制
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
    report_file = f"entailment_test_results_simple_{timestamp}.json"
    
    # 保存详细结果
    all_results_list = []
    for result_data in all_results.values():
        all_results_list.extend(result_data['results'])
    
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(all_results_list, f, indent=2, ensure_ascii=False)
    
    logging.info(f"详细结果已保存: {report_file}")
    
    # 生成Markdown报告
    report_md = f"entailment_test_report_simple_{timestamp}.md"
    with open(report_md, 'w', encoding='utf-8') as f:
        f.write("# Entailment Test Report - Simple Version\n\n")
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
    
    print(f"\n✅ Entailment Test完成!")
    print(f"📄 详细结果: {report_file}")
    print(f"📊 报告: {report_md}")

if __name__ == "__main__":
    main()
