#!/usr/bin/env python3
"""
Entailment Scoring 测试 v2
使用agents.py中的正确实现
测试4个模型（除了qwen7b）在Easy和Hard难度上的表现
"""
import json
import time
import multiprocessing as mp
from datetime import datetime
import os
import random
from agents import PlayerAgent
import config

def test_entailment_sample(sample, model_name, shuffle_seed=None):
    """
    测试单个样本的entailment scoring
    """
    try:
        # 创建代理
        agent = PlayerAgent(agent_id=1, name=f"{model_name}_agent")
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
        if shuffle_seed is not None:
            random.seed(shuffle_seed)
        random.shuffle(all_images)
        
        # 找到目标图片的新位置
        correct_position = -1
        for i, img in enumerate(all_images):
            if img == target_image:
                correct_position = i + 1  # 1-indexed
                break
        
        # 使用entailment scoring
        scores = agent.compute_entailment_scores(caption, all_images)
        
        # 找到最高分数的位置
        max_score = max(scores)
        predicted_position = scores.index(max_score) + 1  # 1-indexed
        
        # 判断是否正确
        correct = (predicted_position == correct_position)
        
        return {
            'success': True,
            'correct': correct,
            'predicted_position': predicted_position,
            'correct_position': correct_position,
            'all_scores': scores,
            'max_score': max_score,
            'reasoning': f"Entailment scoring with {model_name}",
            'tokens': 0,  # agents.py中没有返回token信息
            'image_id': sample['image_id'],
            'difficulty': sample['difficulty'],
            'caption': caption
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'predicted_position': None,
            'correct_position': 1,
            'all_scores': [],
            'max_score': 0,
            'reasoning': '',
            'tokens': 0,
            'image_id': sample['image_id'],
            'difficulty': sample['difficulty'],
            'caption': sample['target']['caption']
        }

def run_entailment_scoring_for_model(model_name, difficulty, dataset_subset, output_queue):
    """
    为单个模型运行entailment scoring测试
    """
    print(f"🚀 开始测试 {model_name} - {difficulty} 难度")
    
    results = []
    correct_count = 0
    total_count = len(dataset_subset)
    
    for i, sample in enumerate(dataset_subset):
        print(f"  📋 {model_name}-{difficulty}: {i+1}/{total_count} - 图片{sample['image_id']}")
        
        result = test_entailment_sample(
            sample, 
            model_name, 
            shuffle_seed=42 + i  # 每个样本使用不同的随机种子
        )
        
        results.append(result)
        
        if result['success'] and result['correct']:
            correct_count += 1
        
        # 每10个样本报告一次进度
        if (i + 1) % 10 == 0:
            accuracy = correct_count / (i + 1) * 100
            print(f"  📊 {model_name}-{difficulty}: {i+1}/{total_count} - 准确率: {accuracy:.1f}%")
    
    final_accuracy = correct_count / total_count * 100
    print(f"✅ {model_name}-{difficulty} 完成: {correct_count}/{total_count} ({final_accuracy:.1f}%)")
    
    # 将结果放入队列
    output_queue.put({
        'model': model_name,
        'difficulty': difficulty,
        'results': results,
        'accuracy': final_accuracy,
        'correct_count': correct_count,
        'total_count': total_count
    })

def main():
    """主函数"""
    print("🎯 开始Entailment Scoring测试 v2")
    print("使用agents.py中的正确实现")
    
    # 加载数据集
    with open('distractor_dataset.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 只使用Easy和Hard难度
    filtered_data = [sample for sample in data['dataset'] if sample['difficulty'] in ['easy', 'hard']]
    print(f"📊 总样本数: {len(filtered_data)} (Easy和Hard难度)")
    
    # 按难度分组
    easy_samples = [s for s in filtered_data if s['difficulty'] == 'easy']
    hard_samples = [s for s in filtered_data if s['difficulty'] == 'hard']
    
    print(f"📊 Easy难度: {len(easy_samples)} 个样本")
    print(f"📊 Hard难度: {len(hard_samples)} 个样本")
    
    # 要测试的模型（除了qwen7b）
    models_to_test = [
        'gpt-4o',
        'gemini-2.5-flash', 
        'gemma3-12b',
        'gemma3-27b'
    ]
    
    print(f"🤖 测试模型: {models_to_test}")
    
    # 创建任务列表
    tasks = []
    for model in models_to_test:
        tasks.append((model, 'easy', easy_samples))
        tasks.append((model, 'hard', hard_samples))
    
    print(f"📋 总任务数: {len(tasks)}")
    
    # 使用多进程执行
    manager = mp.Manager()
    output_queue = manager.Queue()
    
    print("🚀 开始并行执行...")
    start_time = time.time()
    
    with mp.Pool(processes=8) as pool:  # 8个进程
        # 提交所有任务
        for model, difficulty, samples in tasks:
            pool.apply_async(
                run_entailment_scoring_for_model,
                args=(model, difficulty, samples, output_queue)
            )
        
        # 等待所有任务完成
        pool.close()
        pool.join()
    
    # 收集结果
    all_results = {}
    while not output_queue.empty():
        result = output_queue.get()
        key = f"{result['model']}_{result['difficulty']}"
        all_results[key] = result
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"⏱️ 总耗时: {duration:.1f} 秒")
    
    # 生成报告
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"entailment_scoring_report_v2_{timestamp}.json"
    
    report_data = {
        'timestamp': timestamp,
        'duration_seconds': duration,
        'models_tested': models_to_test,
        'total_samples': len(filtered_data),
        'easy_samples': len(easy_samples),
        'hard_samples': len(hard_samples),
        'results': all_results
    }
    
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)
    
    print(f"📄 报告已保存: {report_file}")
    
    # 打印总结
    print("\n📊 测试结果总结:")
    print("=" * 60)
    
    for model in models_to_test:
        print(f"\n🤖 {model}:")
        for difficulty in ['easy', 'hard']:
            key = f"{model}_{difficulty}"
            if key in all_results:
                result = all_results[key]
                print(f"  {difficulty.upper()}: {result['correct_count']}/{result['total_count']} ({result['accuracy']:.1f}%)")
            else:
                print(f"  {difficulty.upper()}: 未完成")
    
    print("\n✅ Entailment Scoring测试完成!")

if __name__ == "__main__":
    main()
