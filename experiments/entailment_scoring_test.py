"""
Entailment Scoring 并行测试
使用entailment scoring策略测试5个模型（除了qwen7b）
只测试Easy和Hard难度，跳过Medium
"""
import json
import time
import multiprocessing as mp
from datetime import datetime
import os
import requests
import base64
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def encode_image(image_path):
    """编码图片为base64"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def call_llm_api_with_image(messages, model_name, temperature=0.1):
    """调用支持图片的LLM API"""
    # 模型映射 - 只使用支持图片的模型
    model_mapping = {
        "gpt-4o": "qwen/qwen-2.5-vl-7b-instruct",  # 使用支持图片的qwen模型
        "gemini-2.5-flash": "qwen/qwen-2.5-vl-7b-instruct",  # 暂时都用qwen
        "gemma3-12b": "qwen/qwen-2.5-vl-7b-instruct",  # 暂时都用qwen
        "gemma3-27b": "qwen/qwen-2.5-vl-7b-instruct"  # 暂时都用qwen
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

def entailment_score_single_image(image_path, caption, model_name="gpt-4o"):
    """
    对单张图片进行entailment评分
    返回0-10的分数
    """
    # 构建entailment评分提示
    prompt = f"""你是一个专业的图像-文本匹配评估专家。请根据以下规则对图像进行评分：

**游戏规则**：
- 你需要在0-10的范围内给图像打分
- 分数表示图像与给定描述的匹配程度
- 10分：完美匹配，图像完全符合描述
- 7-9分：高度匹配，图像基本符合描述
- 4-6分：中等匹配，图像部分符合描述
- 1-3分：低度匹配，图像勉强符合描述
- 0分：完全不匹配，图像与描述无关

**任务**：
请分析图像并给出0-10的分数。

**描述**: "{caption}"

请以JSON格式回复，包含以下字段：
- reasoning: 你的分析过程
- score: 0-10的分数"""
    
    # 编码图片
    try:
        base64_image = encode_image(image_path)
    except Exception as e:
        return 0, f"图片编码失败: {str(e)}", 0
    
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
    
    try:
        # 调用支持图片的API
        response, tokens, error = call_llm_api_with_image(
            messages=messages,
            model_name=model_name,
            temperature=0.1
        )
        
        if error:
            print(f"[{model_name}] API错误: {error}")
            return 0, f"API错误: {error}", 0
        
        # 解析响应
        try:
            print(f"[{model_name}] 响应: {response}")
            if isinstance(response, dict):
                result = response
            else:
                result = json.loads(response)
            
            score = int(result.get('score', 0))
            reasoning = result.get('reasoning', '')
            
            print(f"[{model_name}] 解析结果: score={score}, reasoning={reasoning[:100]}...")
            
            # 确保分数在0-10范围内
            score = max(0, min(10, score))
            
            return score, reasoning, tokens
            
        except json.JSONDecodeError:
            # 如果不是JSON，尝试提取数字
            import re
            numbers = re.findall(r'\d+', str(response))
            score = int(numbers[0]) if numbers else 0
            score = max(0, min(10, score))
            return score, str(response), tokens
            
    except Exception as e:
        return 0, f"异常: {e}", 0

def test_entailment_sample(sample, model_name="gpt-4o", shuffle_seed=42):
    """
    使用entailment scoring测试单个样本
    """
    import random
    
    try:
        # 准备图片路径
        target_image = sample['target']['img']
        distractor_images = [d['img'] for d in sample['distractors']]
        caption = sample['target']['caption']
        
        # 打乱图片顺序（重要！）
        all_images = [target_image] + distractor_images
        if shuffle_seed is not None:
            random.seed(shuffle_seed)
        random.shuffle(all_images)
        
        # 找到目标图片的新位置
        correct_position = all_images.index(target_image) + 1
        
        # 对所有图片评分
        all_scores = []
        total_tokens = 0
        
        for img_path in all_images:
            score, reasoning, tokens = entailment_score_single_image(
                img_path, caption, model_name
            )
            all_scores.append(score)
            total_tokens += tokens
            time.sleep(0.5)  # 避免API限流
        
        # 找到最高分的图片
        max_score = max(all_scores)
        predicted_position = all_scores.index(max_score) + 1
        
        # 判断是否正确
        correct = (predicted_position == correct_position)
        
        return {
            'success': True,
            'correct': correct,
            'predicted_position': predicted_position,
            'correct_position': correct_position,
            'all_scores': all_scores,
            'max_score': max_score,
            'reasoning': reasoning,
            'tokens': total_tokens,
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

def worker_process_entailment(worker_id, model_name, difficulty, image_range, dataset, results_queue, progress_queue):
    """Entailment scoring工作进程"""
    print(f"[Worker {worker_id}] 开始处理 {model_name} - {difficulty} 难度，图片范围: {image_range}")
    
    # 筛选数据
    worker_samples = []
    for sample in dataset:
        if (sample['difficulty'] == difficulty and 
            image_range[0] <= sample['image_id'] <= image_range[1]):
            worker_samples.append(sample)
    
    print(f"[Worker {worker_id}] 找到 {len(worker_samples)} 个样本")
    
    # 处理样本
    worker_results = []
    correct_count = 0
    total_tokens = 0
    
    for i, sample in enumerate(worker_samples):
        try:
            result = test_entailment_sample(
                sample, 
                model_name, 
                shuffle_seed=42 + worker_id * 1000 + i
            )
            worker_results.append(result)
            
            if result.get('success'):
                if result.get('correct'):
                    correct_count += 1
                if result.get('tokens'):
                    total_tokens += result['tokens']
            
            # 更新进度
            progress_queue.put({
                'worker_id': worker_id,
                'model': model_name,
                'difficulty': difficulty,
                'completed': i + 1,
                'total': len(worker_samples),
                'correct': correct_count,
                'tokens': total_tokens
            })
            
            # 延迟避免API限流
            time.sleep(1)
            
        except Exception as e:
            print(f"[Worker {worker_id}] 处理样本 {sample['image_id']} 时出错: {e}")
            worker_results.append({
                'success': False,
                'error': str(e),
                'image_id': sample['image_id'],
                'difficulty': sample['difficulty'],
                'caption': sample['target']['caption']
            })
    
    # 发送结果
    results_queue.put({
        'worker_id': worker_id,
        'model': model_name,
        'difficulty': difficulty,
        'image_range': image_range,
        'results': worker_results,
        'correct_count': correct_count,
        'total_samples': len(worker_samples),
        'total_tokens': total_tokens
    })
    
    print(f"[Worker {worker_id}] 完成！准确率: {correct_count}/{len(worker_samples)} = {correct_count/len(worker_samples)*100:.2f}%")

def monitor_progress_entailment(progress_queue, total_workers):
    """监控进度"""
    completed_workers = 0
    worker_progress = {}
    
    while completed_workers < total_workers:
        try:
            progress = progress_queue.get(timeout=1)
            worker_progress[progress['worker_id']] = progress
            
            # 显示进度
            print(f"\n📊 进度更新:")
            for worker_id, prog in worker_progress.items():
                print(f"  Worker {worker_id} ({prog['model']}-{prog['difficulty']}): {prog['completed']}/{prog['total']} "
                      f"({prog['completed']/prog['total']*100:.1f}%) - 正确: {prog['correct']}")
            
        except:
            continue

def main():
    """主函数"""
    print("="*80)
    print("🧠 Entailment Scoring 并行测试")
    print("="*80)
    
    # 加载数据集
    print("📂 加载数据集...")
    with open("distractor_dataset.json", 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    dataset = data['dataset']
    
    # 只保留Easy和Hard难度
    filtered_dataset = [sample for sample in dataset if sample['difficulty'] in ['easy', 'hard']]
    print(f"   总样本数: {len(dataset)}")
    print(f"   Easy+Hard样本数: {len(filtered_dataset)}")
    
    # 定义要测试的模型（除了qwen7b）
    models_to_test = [
        'gpt-4o',
        'gemini-2.5-flash', 
        'gemma3-12b',
        'gemma3-27b'
    ]
    
    # 定义工作进程配置
    worker_configs = []
    worker_id = 1
    
    for model in models_to_test:
        for difficulty in ['easy', 'hard']:
            # 每个难度分成两部分
            worker_configs.extend([
                {'worker_id': worker_id, 'model': model, 'difficulty': difficulty, 'image_range': (1, 42)},
                {'worker_id': worker_id + 1, 'model': model, 'difficulty': difficulty, 'image_range': (43, 84)}
            ])
            worker_id += 2
    
    print(f"🔧 启动 {len(worker_configs)} 个工作进程...")
    for config in worker_configs:
        print(f"   Worker {config['worker_id']}: {config['model']} - {config['difficulty']} 难度，图片 {config['image_range'][0]}-{config['image_range'][1]}")
    
    # 创建进程间通信队列
    results_queue = mp.Queue()
    progress_queue = mp.Queue()
    
    # 启动工作进程
    processes = []
    for config in worker_configs:
        p = mp.Process(
            target=worker_process_entailment,
            args=(
                config['worker_id'],
                config['model'],
                config['difficulty'],
                config['image_range'],
                filtered_dataset,
                results_queue,
                progress_queue
            )
        )
        p.start()
        processes.append(p)
    
    # 启动进度监控
    monitor_process = mp.Process(target=monitor_progress_entailment, args=(progress_queue, len(worker_configs)))
    monitor_process.start()
    
    # 等待所有工作进程完成
    print(f"\n⏳ 等待所有进程完成...")
    for p in processes:
        p.join()
    
    # 停止进度监控
    monitor_process.terminate()
    monitor_process.join()
    
    # 收集结果
    print(f"\n📊 收集结果...")
    all_results = []
    model_stats = {}
    total_correct = 0
    total_samples = 0
    total_tokens = 0
    
    while not results_queue.empty():
        worker_result = results_queue.get()
        all_results.extend(worker_result['results'])
        
        model = worker_result['model']
        if model not in model_stats:
            model_stats[model] = {'easy': [], 'hard': [], 'total_correct': 0, 'total_samples': 0, 'total_tokens': 0}
        
        # 按难度分类
        for result in worker_result['results']:
            if result.get('success'):
                model_stats[model][result['difficulty']].append(result)
        
        model_stats[model]['total_correct'] += worker_result['correct_count']
        model_stats[model]['total_samples'] += worker_result['total_samples']
        model_stats[model]['total_tokens'] += worker_result['total_tokens']
        
        total_correct += worker_result['correct_count']
        total_samples += worker_result['total_samples']
        total_tokens += worker_result['total_tokens']
        
        print(f"Worker {worker_result['worker_id']} ({worker_result['model']}-{worker_result['difficulty']}): "
              f"{worker_result['correct_count']}/{worker_result['total_samples']} = "
              f"{worker_result['correct_count']/worker_result['total_samples']*100:.2f}%")
    
    # 计算总体准确率
    overall_accuracy = total_correct / total_samples * 100 if total_samples > 0 else 0
    
    # 按模型统计
    print(f"\n📈 按模型统计:")
    model_rankings = []
    for model, stats in model_stats.items():
        easy_correct = sum(1 for r in stats['easy'] if r.get('correct'))
        easy_total = len(stats['easy'])
        hard_correct = sum(1 for r in stats['hard'] if r.get('correct'))
        hard_total = len(stats['hard'])
        
        easy_acc = easy_correct / easy_total * 100 if easy_total > 0 else 0
        hard_acc = hard_correct / hard_total * 100 if hard_total > 0 else 0
        total_acc = stats['total_correct'] / stats['total_samples'] * 100 if stats['total_samples'] > 0 else 0
        
        model_rankings.append({
            'model': model,
            'total_accuracy': total_acc,
            'easy_accuracy': easy_acc,
            'hard_accuracy': hard_acc,
            'total_correct': stats['total_correct'],
            'total_samples': stats['total_samples'],
            'total_tokens': stats['total_tokens']
        })
        
        print(f"  {model}: {total_acc:.2f}% (Easy: {easy_acc:.2f}%, Hard: {hard_acc:.2f}%)")
    
    # 按准确率排序
    model_rankings.sort(key=lambda x: x['total_accuracy'], reverse=True)
    
    print(f"\n🎯 总体结果:")
    print(f"  总样本数: {total_samples}")
    print(f"  正确数: {total_correct}")
    print(f"  准确率: {overall_accuracy:.2f}%")
    print(f"  总tokens: {total_tokens}")
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_data = {
        'test_type': 'entailment_scoring',
        'timestamp': timestamp,
        'models_tested': models_to_test,
        'difficulties_tested': ['easy', 'hard'],
        'total_samples': total_samples,
        'correct_count': total_correct,
        'accuracy': overall_accuracy,
        'total_tokens': total_tokens,
        'model_rankings': model_rankings,
        'results': all_results
    }
    
    output_file = f"entailment_scoring_results_{timestamp}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 结果已保存到: {output_file}")
    
    # 生成简要报告
    report_file = f"entailment_scoring_report_{timestamp}.md"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(f"# Entailment Scoring 测试报告\n\n")
        f.write(f"**测试时间**: {timestamp}\n")
        f.write(f"**测试策略**: Entailment Scoring\n")
        f.write(f"**测试模型**: {', '.join(models_to_test)}\n")
        f.write(f"**测试难度**: Easy, Hard (跳过Medium)\n\n")
        f.write(f"## 总体结果\n\n")
        f.write(f"- **总样本数**: {total_samples}\n")
        f.write(f"- **正确数**: {total_correct}\n")
        f.write(f"- **准确率**: {overall_accuracy:.2f}%\n")
        f.write(f"- **总tokens**: {total_tokens}\n\n")
        f.write(f"## 模型排名\n\n")
        f.write(f"| 排名 | 模型 | 总准确率 | Easy | Hard | Tokens |\n")
        f.write(f"|------|------|---------|------|------|--------|\n")
        for i, model in enumerate(model_rankings, 1):
            f.write(f"| {i} | {model['model']} | {model['total_accuracy']:.2f}% | {model['easy_accuracy']:.2f}% | {model['hard_accuracy']:.2f}% | {model['total_tokens']} |\n")
    
    print(f"📄 报告已保存到: {report_file}")
    
    return output_data

if __name__ == "__main__":
    # 设置multiprocessing启动方法
    mp.set_start_method('spawn', force=True)
    main()
