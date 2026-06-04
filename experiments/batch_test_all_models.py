"""
批量测试所有6个VLM模型
在完整的distractor数据集上测试并生成统计报告
"""
import json
import time
from datetime import datetime
from test_vlm_with_distractors import test_single_sample
from tqdm import tqdm
import os

# 定义要测试的6个模型
MODELS_TO_TEST = [
    "qwen2.5-vl-7b",
    "qwen2.5-vl-32b", 
    "gemma3-12b",
    "gemma3-27b",
    "gpt-4o",
    "gemini-2.5-flash"
]

def test_model_on_full_dataset(model_name, dataset_file="distractor_dataset.json"):
    """
    在完整数据集上测试单个模型
    
    Args:
        model_name: 模型名称
        dataset_file: 数据集文件路径
    
    Returns:
        dict: 测试结果统计
    """
    print("\n" + "="*70)
    print(f"🧪 测试模型: {model_name}")
    print("="*70)
    
    # 加载数据集
    with open(dataset_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    dataset = data['dataset']
    total_samples = len(dataset)
    
    print(f"📊 样本数: {total_samples}")
    print(f"⏰ 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # 测试所有样本
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
    
    for i, sample in enumerate(tqdm(dataset, desc=f"测试{model_name}")):
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
        
        # 每10个样本显示一次进度
        if (i + 1) % 10 == 0:
            current_acc = correct_count / (i + 1 - failed_count) * 100 if (i + 1 - failed_count) > 0 else 0
            print(f"\n  进度: {i+1}/{total_samples}, 当前准确率: {current_acc:.2f}%")
        
        # 添加延迟避免API限流
        time.sleep(0.3)
    
    elapsed_time = time.time() - start_time
    
    # 计算最终统计
    successful_samples = total_samples - failed_count
    overall_accuracy = correct_count / successful_samples * 100 if successful_samples > 0 else 0
    
    # 计算每个难度的准确率
    for diff in difficulty_stats:
        if difficulty_stats[diff]['total'] > 0:
            difficulty_stats[diff]['accuracy'] = \
                difficulty_stats[diff]['correct'] / difficulty_stats[diff]['total'] * 100
        else:
            difficulty_stats[diff]['accuracy'] = 0
    
    # 输出结果
    print("\n" + "="*70)
    print(f"📊 {model_name} 测试结果")
    print("="*70)
    print(f"总样本数: {total_samples}")
    print(f"成功测试: {successful_samples}")
    print(f"失败数: {failed_count}")
    print(f"总正确数: {correct_count}")
    print(f"整体准确率: {overall_accuracy:.2f}%")
    print(f"总tokens: {total_tokens}")
    print(f"耗时: {elapsed_time/60:.1f} 分钟")
    print()
    print("分难度统计:")
    print(f"  Easy:   {difficulty_stats['easy']['accuracy']:.2f}% ({difficulty_stats['easy']['correct']}/{difficulty_stats['easy']['total']})")
    print(f"  Medium: {difficulty_stats['medium']['accuracy']:.2f}% ({difficulty_stats['medium']['correct']}/{difficulty_stats['medium']['total']})")
    print(f"  Hard:   {difficulty_stats['hard']['accuracy']:.2f}% ({difficulty_stats['hard']['correct']}/{difficulty_stats['hard']['total']})")
    
    # 返回结果
    return {
        'model': model_name,
        'total_samples': total_samples,
        'successful_samples': successful_samples,
        'failed_samples': failed_count,
        'correct_count': correct_count,
        'overall_accuracy': overall_accuracy,
        'total_tokens': total_tokens,
        'elapsed_time': elapsed_time,
        'difficulty_stats': difficulty_stats,
        'results': results
    }

def test_all_models(output_dir="batch_test_results"):
    """
    测试所有模型并生成统计报告
    """
    print("="*70)
    print("🚀 批量测试所有VLM模型")
    print("="*70)
    print(f"模型数量: {len(MODELS_TO_TEST)}")
    print(f"模型列表: {', '.join(MODELS_TO_TEST)}")
    print(f"每个模型样本数: 252 (84张图 × 3个难度)")
    print(f"总测试次数: {len(MODELS_TO_TEST) * 252} = {len(MODELS_TO_TEST) * 252}")
    print()
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 存储所有模型的结果
    all_results = {}
    summary_stats = []
    
    overall_start_time = time.time()
    
    # 依次测试每个模型
    for idx, model_name in enumerate(MODELS_TO_TEST, 1):
        print(f"\n{'='*70}")
        print(f"进度: [{idx}/{len(MODELS_TO_TEST)}] 测试模型: {model_name}")
        print(f"{'='*70}")
        
        try:
            # 测试模型
            result = test_model_on_full_dataset(model_name)
            all_results[model_name] = result
            
            # 保存单个模型的详细结果
            model_output_file = os.path.join(
                output_dir,
                f"{model_name.replace('/', '-')}_results.json"
            )
            with open(model_output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            print(f"\n✅ {model_name} 测试完成，结果已保存到: {model_output_file}")
            
            # 添加到汇总统计
            summary_stats.append({
                'model': model_name,
                'overall_accuracy': result['overall_accuracy'],
                'easy_accuracy': result['difficulty_stats']['easy']['accuracy'],
                'medium_accuracy': result['difficulty_stats']['medium']['accuracy'],
                'hard_accuracy': result['difficulty_stats']['hard']['accuracy'],
                'total_tokens': result['total_tokens'],
                'elapsed_time': result['elapsed_time']
            })
            
        except Exception as e:
            print(f"\n❌ 测试 {model_name} 时出错: {e}")
            summary_stats.append({
                'model': model_name,
                'overall_accuracy': 0,
                'easy_accuracy': 0,
                'medium_accuracy': 0,
                'hard_accuracy': 0,
                'total_tokens': 0,
                'elapsed_time': 0,
                'error': str(e)
            })
        
        # 显示已完成的模型数
        print(f"\n✓ 已完成 {idx}/{len(MODELS_TO_TEST)} 个模型")
        if idx < len(MODELS_TO_TEST):
            print(f"⏳ 还剩 {len(MODELS_TO_TEST) - idx} 个模型...")
    
    overall_elapsed_time = time.time() - overall_start_time
    
    # 生成汇总报告
    print("\n" + "="*70)
    print("📊 所有模型测试完成 - 汇总报告")
    print("="*70)
    print(f"总耗时: {overall_elapsed_time/3600:.2f} 小时")
    print()
    
    # 按总准确率排序
    summary_stats_sorted = sorted(summary_stats, key=lambda x: x['overall_accuracy'], reverse=True)
    
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
        'summary_stats': summary_stats_sorted,
        'all_results': all_results
    }
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, ensure_ascii=False, indent=2)
    
    # 生成Markdown报告
    md_report = generate_markdown_report(summary_stats_sorted, overall_elapsed_time, timestamp)
    md_file = os.path.join(output_dir, f"REPORT_{timestamp}.md")
    with open(md_file, 'w', encoding='utf-8') as f:
        f.write(md_report)
    
    print()
    print(f"💾 汇总结果已保存到:")
    print(f"   - JSON: {summary_file}")
    print(f"   - Markdown: {md_file}")
    print(f"   - 详细结果目录: {output_dir}/")
    
    return summary_data

def generate_markdown_report(summary_stats, elapsed_time, timestamp):
    """生成Markdown格式的报告"""
    
    lines = []
    lines.append(f"# 🧪 VLM模型批量测试报告")
    lines.append("")
    lines.append(f"**生成时间**: {timestamp}")
    lines.append(f"**总耗时**: {elapsed_time/3600:.2f} 小时")
    lines.append("")
    lines.append("## 📊 测试配置")
    lines.append("")
    lines.append(f"- **测试模型数**: {len(MODELS_TO_TEST)}")
    lines.append(f"- **每个模型样本数**: 252 (84张图 × 3个难度)")
    lines.append(f"- **总测试次数**: {len(MODELS_TO_TEST) * 252}")
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
    
    global MODELS_TO_TEST
    
    parser = argparse.ArgumentParser(description='批量测试所有VLM模型')
    parser.add_argument(
        '--output-dir',
        type=str,
        default='batch_test_results',
        help='结果输出目录'
    )
    parser.add_argument(
        '--models',
        type=str,
        nargs='+',
        default=MODELS_TO_TEST,
        help='要测试的模型列表'
    )
    
    args = parser.parse_args()
    
    # 更新要测试的模型
    if args.models:
        MODELS_TO_TEST = args.models
    
    # 运行测试
    test_all_models(output_dir=args.output_dir)

if __name__ == "__main__":
    main()
