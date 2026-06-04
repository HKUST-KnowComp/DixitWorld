"""
并行批量测试所有6个VLM模型
使用多进程同时运行，大幅提升效率
"""
import json
import time
import os
from datetime import datetime
from multiprocessing import Pool, Manager
from test_vlm_with_distractors import test_single_sample
from tqdm import tqdm
import threading

# 定义要测试的6个模型
MODELS_TO_TEST = [
    "qwen2.5-vl-7b",
    "qwen2.5-vl-32b", 
    "gemma3-12b",
    "gemma3-27b",
    "gpt-4o",
    "gemini-2.5-flash"
]

def test_model_worker(args):
    """
    工作进程函数：测试单个模型
    
    Args:
        args: (model_name, dataset, start_idx, end_idx, progress_dict, lock)
    
    Returns:
        dict: 测试结果
    """
    model_name, dataset, start_idx, end_idx, progress_dict, lock = args
    
    print(f"\n🚀 进程开始测试模型: {model_name}")
    print(f"   样本范围: {start_idx}-{end_idx} (共{end_idx-start_idx}个)")
    
    # 测试指定范围的样本
    results = []
    correct_count = 0
    total_tokens = 0
    failed_count = 0
    
    # 按难度统计
    difficulty_stats = {
        'easy': {'total': 0, 'correct': 0},
        'medium': {'total': 0, 'correct': 0},
        'hard': {'total': 0, 'correct': 0}
    }
    
    start_time = time.time()
    
    for i in range(start_idx, end_idx):
        sample = dataset[i]
        
        # 测试样本
        result = test_single_sample(
            sample,
            model_name=model_name,
            shuffle_seed=42 + i
        )
        
        # 添加元信息
        result['image_id'] = sample['image_id']
        result['difficulty'] = sample['difficulty']
        result['caption'] = sample['target']['caption']
        
        results.append(result)
        
        # 更新统计
        difficulty = sample['difficulty']
        difficulty_stats[difficulty]['total'] += 1
        
        if result.get('success'):
            if result.get('correct'):
                correct_count += 1
                difficulty_stats[difficulty]['correct'] += 1
        else:
            failed_count += 1
        
        if result.get('tokens'):
            total_tokens += result['tokens']
        
        # 更新进度
        with lock:
            progress_dict[model_name] = {
                'current': i - start_idx + 1,
                'total': end_idx - start_idx,
                'correct': correct_count,
                'accuracy': correct_count / (i - start_idx + 1 - failed_count) * 100 if (i - start_idx + 1 - failed_count) > 0 else 0
            }
        
        # 添加延迟避免API限流
        time.sleep(0.2)
    
    elapsed_time = time.time() - start_time
    
    # 计算最终统计
    successful_samples = (end_idx - start_idx) - failed_count
    overall_accuracy = correct_count / successful_samples * 100 if successful_samples > 0 else 0
    
    # 计算每个难度的准确率
    for diff in difficulty_stats:
        if difficulty_stats[diff]['total'] > 0:
            difficulty_stats[diff]['accuracy'] = \
                difficulty_stats[diff]['correct'] / difficulty_stats[diff]['total'] * 100
        else:
            difficulty_stats[diff]['accuracy'] = 0
    
    print(f"\n✅ {model_name} 测试完成!")
    print(f"   准确率: {overall_accuracy:.2f}% ({correct_count}/{successful_samples})")
    print(f"   耗时: {elapsed_time/60:.1f} 分钟")
    
    return {
        'model': model_name,
        'total_samples': end_idx - start_idx,
        'successful_samples': successful_samples,
        'failed_samples': failed_count,
        'correct_count': correct_count,
        'overall_accuracy': overall_accuracy,
        'total_tokens': total_tokens,
        'elapsed_time': elapsed_time,
        'difficulty_stats': difficulty_stats,
        'results': results
    }

def progress_monitor(progress_dict, lock, total_models):
    """进度监控线程"""
    while True:
        time.sleep(10)  # 每10秒更新一次
        
        with lock:
            if len(progress_dict) == 0:
                continue
            
            print(f"\n📊 实时进度更新 ({datetime.now().strftime('%H:%M:%S')}):")
            print("-" * 60)
            
            for model_name, progress in progress_dict.items():
                current = progress['current']
                total = progress['total']
                accuracy = progress['accuracy']
                correct = progress['correct']
                
                percentage = (current / total) * 100
                bar_length = 20
                filled_length = int(bar_length * current // total)
                bar = '█' * filled_length + '-' * (bar_length - filled_length)
                
                print(f"{model_name:<20} [{bar}] {percentage:5.1f}% ({current}/{total}) "
                      f"准确率: {accuracy:5.1f}% ({correct}正确)")
            
            print("-" * 60)
            
            # 检查是否所有模型都完成
            if len(progress_dict) >= total_models:
                all_complete = all(
                    progress['current'] >= progress['total'] 
                    for progress in progress_dict.values()
                )
                if all_complete:
                    print("🎉 所有模型测试完成！")
                    break

def parallel_test_all_models(output_dir="parallel_batch_test_results", max_workers=6):
    """
    并行测试所有模型
    
    Args:
        output_dir: 输出目录
        max_workers: 最大工作进程数
    """
    print("="*70)
    print("🚀 并行批量测试所有VLM模型")
    print("="*70)
    print(f"模型数量: {len(MODELS_TO_TEST)}")
    print(f"模型列表: {', '.join(MODELS_TO_TEST)}")
    print(f"每个模型样本数: 252 (84张图 × 3个难度)")
    print(f"总测试次数: {len(MODELS_TO_TEST) * 252} = {len(MODELS_TO_TEST) * 252}")
    print(f"并行进程数: {max_workers}")
    print(f"预计耗时: {252 * 0.5 / 60:.1f} 小时 (并行)")
    print()
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载数据集
    print("📥 加载数据集...")
    with open("distractor_dataset.json", 'r', encoding='utf-8') as f:
        data = json.load(f)
    dataset = data['dataset']
    print(f"✅ 加载了 {len(dataset)} 个样本")
    
    # 创建共享对象用于进度监控
    manager = Manager()
    progress_dict = manager.dict()
    lock = manager.Lock()
    
    # 启动进度监控线程
    monitor_thread = threading.Thread(
        target=progress_monitor, 
        args=(progress_dict, lock, len(MODELS_TO_TEST))
    )
    monitor_thread.daemon = True
    monitor_thread.start()
    
    # 准备参数：每个模型测试全部252个样本
    worker_args = []
    for model_name in MODELS_TO_TEST:
        args = (model_name, dataset, 0, len(dataset), progress_dict, lock)
        worker_args.append(args)
    
    overall_start_time = time.time()
    
    print(f"\n🚀 开始并行测试...")
    print(f"⏰ 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # 使用进程池并行执行
    with Pool(processes=max_workers) as pool:
        results = pool.map(test_model_worker, worker_args)
    
    overall_elapsed_time = time.time() - overall_start_time
    
    # 保存每个模型的详细结果
    print(f"\n💾 保存详细结果...")
    for result in results:
        model_name = result['model']
        model_output_file = os.path.join(
            output_dir,
            f"{model_name.replace('/', '-')}_results.json"
        )
        with open(model_output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"  ✅ {model_name}: {model_output_file}")
    
    # 生成汇总统计
    print(f"\n📊 生成汇总报告...")
    summary_stats = []
    for result in results:
        summary_stats.append({
            'model': result['model'],
            'overall_accuracy': result['overall_accuracy'],
            'easy_accuracy': result['difficulty_stats']['easy']['accuracy'],
            'medium_accuracy': result['difficulty_stats']['medium']['accuracy'],
            'hard_accuracy': result['difficulty_stats']['hard']['accuracy'],
            'total_tokens': result['total_tokens'],
            'elapsed_time': result['elapsed_time']
        })
    
    # 按总准确率排序
    summary_stats_sorted = sorted(summary_stats, key=lambda x: x['overall_accuracy'], reverse=True)
    
    # 输出最终结果
    print("\n" + "="*70)
    print("📊 所有模型测试完成 - 汇总报告")
    print("="*70)
    print(f"总耗时: {overall_elapsed_time/3600:.2f} 小时")
    print()
    
    print("🏆 模型排名（按总准确率）:")
    print()
    print(f"{'排名':<4} {'模型':<25} {'总准确率':<10} {'Easy':<8} {'Medium':<8} {'Hard':<8}")
    print("-" * 70)
    
    for rank, stats in enumerate(summary_stats_sorted, 1):
        print(f"{rank:<4} {stats['model']:<25} {stats['overall_accuracy']:>6.2f}%  "
              f"{stats['easy_accuracy']:>6.2f}% {stats['medium_accuracy']:>6.2f}% {stats['hard_accuracy']:>6.2f}%")
    
    print()
    print("详细统计:")
    print(f"{'模型':<25} {'总准确率':<10} {'Easy':<10} {'Medium':<10} {'Hard':<10} {'Tokens':<10}")
    print("-" * 85)
    
    for stats in summary_stats_sorted:
        print(f"{stats['model']:<25} {stats['overall_accuracy']:>6.2f}%   "
              f"{stats['easy_accuracy']:>6.2f}%   {stats['medium_accuracy']:>6.2f}%   "
              f"{stats['hard_accuracy']:>6.2f}%   {stats['total_tokens']:>8}")
    
    # 保存汇总结果
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    summary_file = os.path.join(output_dir, f"summary_{timestamp}.json")
    
    summary_data = {
        'timestamp': timestamp,
        'total_models': len(MODELS_TO_TEST),
        'total_elapsed_time': overall_elapsed_time,
        'parallel_workers': max_workers,
        'summary_stats': summary_stats_sorted,
        'all_results': {r['model']: r for r in results}
    }
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, ensure_ascii=False, indent=2)
    
    # 生成Markdown报告
    md_report = generate_markdown_report(summary_stats_sorted, overall_elapsed_time, timestamp, max_workers)
    md_file = os.path.join(output_dir, f"REPORT_{timestamp}.md")
    with open(md_file, 'w', encoding='utf-8') as f:
        f.write(md_report)
    
    print()
    print(f"💾 汇总结果已保存到:")
    print(f"   - JSON: {summary_file}")
    print(f"   - Markdown: {md_file}")
    print(f"   - 详细结果目录: {output_dir}/")
    
    return summary_data

def generate_markdown_report(summary_stats, elapsed_time, timestamp, max_workers):
    """生成Markdown格式的报告"""
    
    lines = []
    lines.append(f"# 🧪 VLM模型并行批量测试报告")
    lines.append("")
    lines.append(f"**生成时间**: {timestamp}")
    lines.append(f"**总耗时**: {elapsed_time/3600:.2f} 小时")
    lines.append(f"**并行进程数**: {max_workers}")
    lines.append("")
    lines.append("## 📊 测试配置")
    lines.append("")
    lines.append(f"- **测试模型数**: {len(MODELS_TO_TEST)}")
    lines.append(f"- **每个模型样本数**: 252 (84张图 × 3个难度)")
    lines.append(f"- **总测试次数**: {len(MODELS_TO_TEST) * 252}")
    lines.append(f"- **并行特性**: 多进程同时测试，大幅提升效率")
    lines.append("")
    lines.append("## 🏆 模型排名")
    lines.append("")
    lines.append("| 排名 | 模型 | 总准确率 | Easy | Medium | Hard | Tokens |")
    lines.append("|------|------|---------|------|--------|------|--------|")
    
    for rank, stats in enumerate(summary_stats, 1):
        lines.append(f"| {rank} | {stats['model']} | **{stats['overall_accuracy']:.2f}%** | "
                    f"{stats['easy_accuracy']:.2f}% | {stats['medium_accuracy']:.2f}% | "
                    f"{stats['hard_accuracy']:.2f}% | {stats['total_tokens']:,} |")
    
    lines.append("")
    lines.append("## 📈 准确率对比")
    lines.append("")
    lines.append("### 按难度")
    lines.append("")
    
    for difficulty in ['Easy', 'Medium', 'Hard']:
        lines.append(f"#### {difficulty}难度")
        lines.append("")
        diff_key = difficulty.lower() + '_accuracy'
        sorted_by_diff = sorted(summary_stats, key=lambda x: x[diff_key], reverse=True)
        
        for rank, stats in enumerate(sorted_by_diff, 1):
            acc = stats[diff_key]
            bar_length = int(acc / 2)  # 50% = 25个字符
            bar = '█' * bar_length
            lines.append(f"{rank}. **{stats['model']}**: {acc:.2f}% {bar}")
        lines.append("")
    
    lines.append("## 💰 成本分析")
    lines.append("")
    lines.append("| 模型 | Tokens | 预估成本 |")
    lines.append("|------|--------|---------|")
    
    for stats in summary_stats:
        # 简单估算成本（实际成本根据具体API定价）
        tokens = stats['total_tokens']
        if 'gpt' in stats['model']:
            cost = tokens / 1000 * 0.01  # 假设$0.01/1K tokens
        elif 'gemini' in stats['model']:
            cost = tokens / 1000 * 0.005
        else:
            cost = tokens / 1000 * 0.002
        
        lines.append(f"| {stats['model']} | {tokens:,} | ${cost:.2f} |")
    
    lines.append("")
    lines.append("## ⚡ 并行效率")
    lines.append("")
    lines.append(f"- **并行进程数**: {max_workers}")
    lines.append(f"- **总耗时**: {elapsed_time/3600:.2f} 小时")
    lines.append(f"- **串行预估耗时**: {elapsed_time * max_workers / 3600:.2f} 小时")
    lines.append(f"- **效率提升**: {max_workers:.1f}x")
    lines.append("")
    lines.append("## 📝 总结")
    lines.append("")
    
    best_model = summary_stats[0]
    lines.append(f"- **最佳模型**: {best_model['model']} ({best_model['overall_accuracy']:.2f}%)")
    
    best_easy = max(summary_stats, key=lambda x: x['easy_accuracy'])
    lines.append(f"- **Easy最佳**: {best_easy['model']} ({best_easy['easy_accuracy']:.2f}%)")
    
    best_medium = max(summary_stats, key=lambda x: x['medium_accuracy'])
    lines.append(f"- **Medium最佳**: {best_medium['model']} ({best_medium['medium_accuracy']:.2f}%)")
    
    best_hard = max(summary_stats, key=lambda x: x['hard_accuracy'])
    lines.append(f"- **Hard最佳**: {best_hard['model']} ({best_hard['hard_accuracy']:.2f}%)")
    
    avg_accuracy = sum(s['overall_accuracy'] for s in summary_stats) / len(summary_stats)
    lines.append(f"- **平均准确率**: {avg_accuracy:.2f}%")
    
    return '\n'.join(lines)

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='并行批量测试所有VLM模型')
    parser.add_argument(
        '--output-dir',
        type=str,
        default='parallel_batch_test_results',
        help='结果输出目录'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=6,
        help='并行进程数（默认6，与模型数相同）'
    )
    
    args = parser.parse_args()
    
    # 运行并行测试
    parallel_test_all_models(output_dir=args.output_dir, max_workers=args.workers)

if __name__ == "__main__":
    main()
