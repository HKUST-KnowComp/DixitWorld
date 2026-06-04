"""
批量测试前的系统检查
确保所有必要的文件和配置都就绪
"""
import os
import json

def check_files():
    """检查必要文件是否存在"""
    print("📁 检查必要文件...")
    
    required_files = [
        'distractor_dataset.json',
        'distractor_lists.json',
        'image_captions.json',
        'call_api_openrouter.py',
        'test_vlm_with_distractors.py',
        'batch_test_all_models.py'
    ]
    
    missing_files = []
    for file in required_files:
        if os.path.exists(file):
            print(f"  ✅ {file}")
        else:
            print(f"  ❌ {file} - 缺失！")
            missing_files.append(file)
    
    return len(missing_files) == 0

def check_dataset():
    """检查数据集配置"""
    print("\n📊 检查数据集...")
    
    try:
        with open('distractor_dataset.json', 'r') as f:
            data = json.load(f)
        
        total_samples = data['metadata']['total_samples']
        print(f"  ✅ 数据集样本数: {total_samples}")
        
        # 检查是否使用了Dixit风格的短描述
        sample = data['dataset'][0]
        caption = sample['target']['caption']
        word_count = len(caption.split())
        
        if word_count <= 4:
            print(f"  ✅ Caption风格: Dixit抽象短描述 ('{caption}', {word_count}词)")
        else:
            print(f"  ⚠️  Caption较长: '{caption}' ({word_count}词)")
            print(f"     建议使用更短的Dixit风格描述")
        
        return True
    except Exception as e:
        print(f"  ❌ 数据集检查失败: {e}")
        return False

def check_images():
    """检查图片文件"""
    print("\n🖼️  检查图片文件...")
    
    if not os.path.exists('images'):
        print("  ❌ images目录不存在！")
        return False
    
    image_files = [f for f in os.listdir('images') if f.endswith('.png')]
    print(f"  ✅ 找到 {len(image_files)} 张图片")
    
    if len(image_files) < 84:
        print(f"  ⚠️  图片数量不足84张")
        return False
    
    return True

def check_api_config():
    """检查API配置"""
    print("\n🔑 检查API配置...")
    
    try:
        with open('call_api_openrouter.py', 'r') as f:
            content = f.read()
        
        if 'sk-or-v1-' in content:
            print("  ✅ 找到OpenRouter API密钥")
        else:
            print("  ❌ 未找到API密钥")
            return False
        
        # 检查模型映射
        models_to_test = [
            "qwen2.5-vl-7b",
            "qwen2.5-vl-32b",
            "gemma3-12b",
            "gemma3-27b",
            "gpt-4o",
            "gemini-2.5-flash"
        ]
        
        missing_models = []
        for model in models_to_test:
            if model not in content:
                missing_models.append(model)
        
        if missing_models:
            print(f"  ⚠️  以下模型可能未配置: {', '.join(missing_models)}")
        else:
            print(f"  ✅ 所有6个测试模型都已配置")
        
        return True
    except Exception as e:
        print(f"  ❌ API配置检查失败: {e}")
        return False

def estimate_cost_and_time():
    """估算成本和时间"""
    print("\n💰 成本和时间估算...")
    
    print("\n  每个模型测试252个样本:")
    print("  ┌─────────────────────┬──────────┬──────────┬──────────┐")
    print("  │ 模型                │ 预估成本 │ 预估时间 │ 状态     │")
    print("  ├─────────────────────┼──────────┼──────────┼──────────┤")
    print("  │ qwen2.5-vl-7b       │  $3-5    │  30-40分 │ 经济     │")
    print("  │ qwen2.5-vl-32b      │  $5-8    │  35-45分 │ 经济     │")
    print("  │ gemma3-12b          │  $3-5    │  30-40分 │ 经济     │")
    print("  │ gemma3-27b          │  $5-8    │  35-45分 │ 经济     │")
    print("  │ gpt-4o              │ $20-30   │  40-60分 │ 昂贵     │")
    print("  │ gemini-2.5-flash    │  $8-12   │  25-35分 │ 中等     │")
    print("  └─────────────────────┴──────────┴──────────┴──────────┘")
    print("\n  总计:")
    print("    💵 成本: $50-80")
    print("    ⏰ 时间: 3-6小时")
    print()
    print("  建议:")
    print("    - 先测试便宜的模型（qwen, gemma）验证系统")
    print("    - 确认无误后再测试昂贵的模型（gpt-4o）")
    print("    - 在网络稳定、不需要电脑时运行")

def main():
    """主函数"""
    print("="*70)
    print("🔍 批量测试系统检查")
    print("="*70)
    print()
    
    # 执行所有检查
    checks = [
        ("文件检查", check_files()),
        ("数据集检查", check_dataset()),
        ("图片检查", check_images()),
        ("API配置检查", check_api_config())
    ]
    
    # 汇总结果
    print("\n" + "="*70)
    print("📋 检查结果汇总")
    print("="*70)
    
    all_passed = True
    for check_name, passed in checks:
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"  {check_name}: {status}")
        if not passed:
            all_passed = False
    
    print()
    
    # 显示估算
    estimate_cost_and_time()
    
    # 最终建议
    print()
    print("="*70)
    if all_passed:
        print("✅ 所有检查通过！可以开始批量测试")
        print("="*70)
        print()
        print("🚀 运行测试:")
        print()
        print("  # 测试所有6个模型")
        print("  python3 batch_test_all_models.py")
        print()
        print("  # 或先测试便宜的模型")
        print("  python3 batch_test_all_models.py --models qwen2.5-vl-7b gemma3-12b")
        print()
    else:
        print("⚠️  部分检查未通过，请先解决上述问题")
        print("="*70)
        print()
        print("常见问题解决:")
        print("  1. 缺少文件 → 运行对应的生成脚本")
        print("  2. 数据集问题 → 运行 python3 build_distractor_dataset.py")
        print("  3. API配置 → 检查 call_api_openrouter.py 中的密钥")
        print()

if __name__ == "__main__":
    main()
