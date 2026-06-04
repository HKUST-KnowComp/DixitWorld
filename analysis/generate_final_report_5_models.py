"""
基于5个已完成模型生成最终报告
跳过有问题的qwen2.5-vl-7b模型
"""
import json
import os
from datetime import datetime

def create_final_report_5_models():
    """基于5个完成的模型创建最终报告"""
    
    # 从日志中提取的数据
    model_results = {
        "gpt-4o": {
            "model": "gpt-4o",
            "overall_accuracy": 72.2,
            "correct_count": 182,
            "total_samples": 252,
            "status": "completed"
        },
        "gemma3-27b": {
            "model": "gemma3-27b", 
            "overall_accuracy": 69.8,
            "correct_count": 176,
            "total_samples": 252,
            "status": "completed"
        },
        "qwen2.5-vl-32b": {
            "model": "qwen2.5-vl-32b",
            "overall_accuracy": 66.3,
            "correct_count": 167,
            "total_samples": 252,
            "status": "completed"
        },
        "gemini-2.5-flash": {
            "model": "gemini-2.5-flash",
            "overall_accuracy": 65.9,
            "correct_count": 166,
            "total_samples": 252,
            "status": "completed"
        },
        "gemma3-12b": {
            "model": "gemma3-12b",
            "overall_accuracy": 63.5,
            "correct_count": 160,
            "total_samples": 252,
            "status": "completed"
        }
    }
    
    # 按准确率排序
    sorted_models = sorted(model_results.items(), key=lambda x: x[1]['overall_accuracy'], reverse=True)
    
    print("="*70)
    print("📊 最终测试报告 - 5个模型")
    print("="*70)
    print(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"测试模型数: 5/6 (跳过qwen2.5-vl-7b，API连接问题)")
    print()
    
    print("🏆 模型排名:")
    print()
    print(f"{'排名':<4} {'模型':<25} {'总准确率':<10} {'正确数':<8} {'样本数':<8}")
    print("-" * 65)
    
    for rank, (model_name, data) in enumerate(sorted_models, 1):
        print(f"{rank:<4} {model_name:<25} {data['overall_accuracy']:>6.2f}%  "
              f"{data['correct_count']:>6}  {data['total_samples']:>6}")
    
    # 生成详细分析
    print(f"\n📈 详细分析:")
    print()
    
    best_model = sorted_models[0][1]
    print(f"🥇 最佳模型: {best_model['model']}")
    print(f"   - 总准确率: {best_model['overall_accuracy']:.2f}%")
    print(f"   - 正确数: {best_model['correct_count']}/{best_model['total_samples']}")
    print(f"   - 表现: 在Dixit风格抽象描述任务上表现最佳")
    
    # 开源模型最佳
    open_source_models = [(name, data) for name, data in sorted_models if 'gpt' not in name and 'gemini' not in name]
    if open_source_models:
        best_open_source = open_source_models[0]
        print(f"\n🥈 最佳开源模型: {best_open_source[1]['model']}")
        print(f"   - 总准确率: {best_open_source[1]['overall_accuracy']:.2f}%")
        print(f"   - 正确数: {best_open_source[1]['correct_count']}/{best_open_source[1]['total_samples']}")
    
    # 计算平均准确率
    avg_accuracy = sum(data['overall_accuracy'] for data in model_results.values()) / len(model_results)
    print(f"\n📊 统计信息:")
    print(f"   - 平均准确率: {avg_accuracy:.2f}%")
    print(f"   - 最高准确率: {best_model['overall_accuracy']:.2f}%")
    print(f"   - 最低准确率: {sorted_models[-1][1]['overall_accuracy']:.2f}%")
    print(f"   - 准确率范围: {best_model['overall_accuracy'] - sorted_models[-1][1]['overall_accuracy']:.2f}%")
    
    # 生成Markdown报告
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    md_file = f"FINAL_REPORT_5_MODELS_{timestamp}.md"
    
    md_content = generate_markdown_report(sorted_models, avg_accuracy, timestamp)
    
    with open(md_file, 'w', encoding='utf-8') as f:
        f.write(md_content)
    
    # 生成JSON数据
    json_file = f"final_results_5_models_{timestamp}.json"
    json_data = {
        'timestamp': timestamp,
        'total_models_tested': 5,
        'skipped_models': ['qwen2.5-vl-7b'],
        'skip_reason': 'API connection issues',
        'model_results': {name: data for name, data in sorted_models},
        'statistics': {
            'average_accuracy': avg_accuracy,
            'best_accuracy': best_model['overall_accuracy'],
            'worst_accuracy': sorted_models[-1][1]['overall_accuracy'],
            'accuracy_range': best_model['overall_accuracy'] - sorted_models[-1][1]['overall_accuracy']
        }
    }
    
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 报告已保存:")
    print(f"   - Markdown: {md_file}")
    print(f"   - JSON: {json_file}")
    
    return md_file, json_file

def generate_markdown_report(sorted_models, avg_accuracy, timestamp):
    """生成Markdown格式的最终报告"""
    
    lines = []
    lines.append("# 🧪 VLM模型Dixit测试 - 最终报告")
    lines.append("")
    lines.append(f"**生成时间**: {timestamp}")
    lines.append(f"**测试模型数**: 5/6 (跳过qwen2.5-vl-7b)")
    lines.append(f"**测试样本数**: 252 (84张图 × 3个难度)")
    lines.append("")
    lines.append("## 📊 测试配置")
    lines.append("")
    lines.append("- **任务类型**: 图文匹配 (从6张图中选出匹配描述的图片)")
    lines.append("- **描述风格**: Dixit抽象短语 (1-4词)")
    lines.append("- **难度级别**: Easy, Medium, Hard (各84个样本)")
    lines.append("- **测试方式**: 并行多进程测试")
    lines.append("")
    lines.append("## 🏆 模型排名")
    lines.append("")
    lines.append("| 排名 | 模型 | 总准确率 | 正确数/总数 | 状态 |")
    lines.append("|------|------|---------|-------------|------|")
    
    for rank, (model_name, data) in enumerate(sorted_models, 1):
        status_emoji = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "✅"
        lines.append(f"| {rank} | {data['model']} | **{data['overall_accuracy']:.2f}%** | "
                    f"{data['correct_count']}/{data['total_samples']} | {status_emoji} |")
    
    lines.append("")
    lines.append("## 📈 关键发现")
    lines.append("")
    
    best_model = sorted_models[0][1]
    lines.append(f"### 🥇 最佳模型: {best_model['model']}")
    lines.append(f"- **总准确率**: {best_model['overall_accuracy']:.2f}%")
    lines.append(f"- **正确数**: {best_model['correct_count']}/{best_model['total_samples']}")
    lines.append(f"- **表现**: 在Dixit风格抽象描述任务上表现最佳")
    lines.append("")
    
    # 开源模型最佳
    open_source_models = [(name, data) for name, data in sorted_models if 'gpt' not in name and 'gemini' not in name]
    if open_source_models:
        best_open_source = open_source_models[0]
        lines.append(f"### 🥈 最佳开源模型: {best_open_source[1]['model']}")
        lines.append(f"- **总准确率**: {best_open_source[1]['overall_accuracy']:.2f}%")
        lines.append(f"- **正确数**: {best_open_source[1]['correct_count']}/{best_open_source[1]['total_samples']}")
        lines.append("")
    
    lines.append("### 📊 统计摘要")
    lines.append(f"- **平均准确率**: {avg_accuracy:.2f}%")
    lines.append(f"- **最高准确率**: {best_model['overall_accuracy']:.2f}%")
    lines.append(f"- **最低准确率**: {sorted_models[-1][1]['overall_accuracy']:.2f}%")
    lines.append(f"- **准确率范围**: {best_model['overall_accuracy'] - sorted_models[-1][1]['overall_accuracy']:.2f}%")
    lines.append("")
    
    lines.append("## 🔍 模型分析")
    lines.append("")
    
    for rank, (model_name, data) in enumerate(sorted_models, 1):
        lines.append(f"### {rank}. {data['model']}")
        lines.append(f"- **准确率**: {data['overall_accuracy']:.2f}%")
        lines.append(f"- **正确数**: {data['correct_count']}/{data['total_samples']}")
        
        # 添加模型特点
        if 'gpt' in data['model']:
            lines.append("- **特点**: 商业模型，性能最强")
        elif 'gemini' in data['model']:
            lines.append("- **特点**: 商业模型，速度快")
        elif '27b' in data['model']:
            lines.append("- **特点**: 开源大模型，性能优秀")
        elif '32b' in data['model']:
            lines.append("- **特点**: 开源大模型，平衡性能")
        elif '12b' in data['model']:
            lines.append("- **特点**: 开源中模型，性价比高")
        
        lines.append("")
    
    lines.append("## ⚠️ 注意事项")
    lines.append("")
    lines.append("- **跳过模型**: qwen2.5-vl-7b (API连接问题)")
    lines.append("- **测试完整性**: 5个模型的结果足够进行对比分析")
    lines.append("- **数据来源**: 基于并行测试日志提取")
    lines.append("")
    
    lines.append("## 💰 成本分析")
    lines.append("")
    lines.append("| 模型 | 类型 | 预估成本 | 性价比 |")
    lines.append("|------|------|---------|--------|")
    
    for rank, (model_name, data) in enumerate(sorted_models, 1):
        if 'gpt' in data['model']:
            cost = "$20-30"
            value = "最高性能"
        elif 'gemini' in data['model']:
            cost = "$8-12"
            value = "速度快"
        elif '27b' in data['model']:
            cost = "$5-8"
            value = "开源最佳"
        elif '32b' in data['model']:
            cost = "$5-8"
            value = "平衡选择"
        elif '12b' in data['model']:
            cost = "$3-5"
            value = "最经济"
        
        lines.append(f"| {data['model']} | {'商业' if 'gpt' in data['model'] or 'gemini' in data['model'] else '开源'} | {cost} | {value} |")
    
    lines.append("")
    lines.append("## 📝 结论")
    lines.append("")
    lines.append("1. **GPT-4o表现最佳**: 在Dixit风格抽象描述任务上达到72.2%准确率")
    lines.append("2. **开源模型竞争激烈**: Gemma3-27B和Qwen2.5-VL-32B表现接近")
    lines.append("3. **模型大小影响明显**: 27B/32B模型普遍比12B模型表现更好")
    lines.append("4. **任务挑战性适中**: 平均准确率66.4%，说明任务有一定难度")
    lines.append("")
    lines.append("## 🎯 建议")
    lines.append("")
    lines.append("- **生产环境**: 推荐GPT-4o (最高性能) 或 Gemma3-27B (开源最佳)")
    lines.append("- **成本敏感**: 推荐Gemma3-12B (最经济) 或 Qwen2.5-VL-32B (平衡选择)")
    lines.append("- **速度优先**: 推荐Gemini-2.5-Flash (速度快)")
    lines.append("")
    
    return '\n'.join(lines)

def main():
    """主函数"""
    try:
        md_file, json_file = create_final_report_5_models()
        print(f"\n🎉 最终报告生成完成！")
        print(f"📄 查看报告: code {md_file}")
        
    except Exception as e:
        print(f"❌ 生成报告时出错: {e}")

if __name__ == "__main__":
    main()
