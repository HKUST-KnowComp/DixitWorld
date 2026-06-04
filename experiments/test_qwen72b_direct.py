# -*- coding: utf-8 -*-
"""
测试 qwen2.5-vl-72b 模型在三个难度上的直接选择正确率
"""

from batch_test_all_models import test_model_on_full_dataset
import json
import os
from datetime import datetime

def main():
    """主函数 - 测试 qwen2.5-vl-72b 模型"""
    model_name = "qwen2.5-vl-72b"
    
    print("="*70)
    print(f"🧪 测试模型: {model_name}")
    print(f"📋 测试策略: 直接选择")
    print(f"📊 测试数据集: distractor_dataset.json (252个样本: 84张图 × 3个难度)")
    print("="*70)
    print()
    
    # 运行测试
    result = test_model_on_full_dataset(model_name)
    
    # 保存结果
    output_dir = "batch_test_results"
    os.makedirs(output_dir, exist_ok=True)
    
    model_output_file = os.path.join(
        output_dir,
        f"{model_name.replace('/', '-')}_results.json"
    )
    
    with open(model_output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 测试完成，结果已保存到: {model_output_file}")
    
    # 打印详细统计
    print("\n" + "="*70)
    print("📊 详细统计结果")
    print("="*70)
    print(f"模型: {model_name}")
    print(f"总样本数: {result['total_samples']}")
    print(f"成功测试: {result['successful_samples']}")
    print(f"失败数: {result['failed_samples']}")
    print(f"总正确数: {result['correct_count']}")
    print(f"整体准确率: {result['overall_accuracy']:.2f}%")
    print(f"总tokens: {result['total_tokens']}")
    print(f"耗时: {result['elapsed_time']/60:.1f} 分钟")
    print()
    print("分难度统计:")
    print(f"  Easy:   {result['difficulty_stats']['easy']['accuracy']:.2f}% "
          f"({result['difficulty_stats']['easy']['correct']}/{result['difficulty_stats']['easy']['total']})")
    print(f"  Medium: {result['difficulty_stats']['medium']['accuracy']:.2f}% "
          f"({result['difficulty_stats']['medium']['correct']}/{result['difficulty_stats']['medium']['total']})")
    print(f"  Hard:   {result['difficulty_stats']['hard']['accuracy']:.2f}% "
          f"({result['difficulty_stats']['hard']['correct']}/{result['difficulty_stats']['hard']['total']})")
    print("="*70)

if __name__ == "__main__":
    main()

