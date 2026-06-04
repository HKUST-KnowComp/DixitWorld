"""
使用干扰图数据集测试VLM模型的图文匹配能力
"""
import json
import random
import base64
from pathlib import Path
from call_api_openrouter import call_llm_api
from tqdm import tqdm
import time

def encode_image_to_base64(image_path):
    """将图片编码为base64字符串"""
    with open(image_path, 'rb') as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def test_single_sample(sample, model_name="gpt-4o", shuffle_seed=None):
    """
    测试单个样本
    
    Args:
        sample: 数据集样本
        model_name: 使用的VLM模型
        shuffle_seed: 打乱顺序的随机种子
    
    Returns:
        dict: 测试结果
    """
    # 准备所有图片（target + distractors）
    all_images = [sample['target']] + sample['distractors']
    
    # 打乱顺序
    if shuffle_seed is not None:
        random.seed(shuffle_seed)
    random.shuffle(all_images)
    
    # 记录正确答案的位置
    correct_position = None
    for i, img in enumerate(all_images):
        if 'caption' in img:  # target图片有caption
            correct_position = i + 1  # 1-based index
            break
    
    # 构建prompt
    caption = sample['target']['caption']
    
    # 编码所有图片
    image_contents = []
    for i, img in enumerate(all_images):
        img_path = img['img']
        base64_image = encode_image_to_base64(img_path)
        image_contents.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{base64_image}"
            }
        })
    
    # 构建消息
    prompt_text = f"""You are given {len(all_images)} images and a caption. 
Your task is to identify which image best matches the caption.

Caption: "{caption}"

Please analyze each image carefully and choose the number (1-{len(all_images)}) of the image that best matches this caption.

Return your response in JSON format:
{{
    "reasoning": "Brief explanation of why you chose this image",
    "answer": "X"
}}

Where X is the number (1-{len(all_images)}) of the matching image.
"""
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text}
            ] + image_contents
        }
    ]
    
    # 调用VLM
    try:
        response, error, tokens = call_llm_api(
            messages=messages,
            model_name=model_name,
            temperature=0.1,  # 低温度以获得更确定的答案
            trial_info={'trial_id': f"img{sample['image_id']}_{sample['difficulty']}"}
        )
        
        if error:
            return {
                'success': False,
                'error': error,
                'correct_position': correct_position
            }
        
        # 解析答案
        predicted_answer = response.get('answer', '').strip()
        reasoning = response.get('reasoning', '')
        
        # 尝试提取数字
        try:
            predicted_position = int(predicted_answer)
        except ValueError:
            # 尝试从文本中提取数字
            import re
            numbers = re.findall(r'\d+', predicted_answer)
            if numbers:
                predicted_position = int(numbers[0])
            else:
                predicted_position = -1
        
        # 判断是否正确
        is_correct = (predicted_position == correct_position)
        
        return {
            'success': True,
            'correct': is_correct,
            'predicted_position': predicted_position,
            'correct_position': correct_position,
            'reasoning': reasoning,
            'tokens': tokens
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'correct_position': correct_position
        }

def run_evaluation(dataset_file="distractor_dataset.json",
                   model_name="gpt-4o",
                   max_samples=None,
                   difficulties=None,
                   output_file=None):
    """
    运行完整评估
    
    Args:
        dataset_file: 数据集文件
        model_name: VLM模型名称
        max_samples: 最大测试样本数（None表示全部）
        difficulties: 要测试的难度列表（None表示全部）
        output_file: 结果输出文件
    """
    print("="*70)
    print(f"🧪 VLM图文匹配能力测试")
    print("="*70)
    print(f"模型: {model_name}")
    
    # 加载数据集
    with open(dataset_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    dataset = data['dataset']
    
    # 筛选难度
    if difficulties:
        dataset = [s for s in dataset if s['difficulty'] in difficulties]
    
    # 限制样本数
    if max_samples:
        dataset = dataset[:max_samples]
    
    print(f"样本数: {len(dataset)}")
    print(f"难度: {difficulties if difficulties else '全部'}")
    print()
    
    # 测试所有样本
    results = []
    correct_count = 0
    total_tokens = 0
    
    for i, sample in enumerate(tqdm(dataset, desc="测试进度")):
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
        
        if result.get('success') and result.get('correct'):
            correct_count += 1
        
        if result.get('tokens'):
            total_tokens += result['tokens']
        
        # 显示进度
        if (i + 1) % 10 == 0:
            current_acc = correct_count / (i + 1) * 100
            print(f"\n当前准确率: {current_acc:.2f}% ({correct_count}/{i+1})")
        
        # 添加延迟避免API限流
        time.sleep(0.5)
    
    # 计算统计信息
    total_samples = len(results)
    successful_samples = sum(1 for r in results if r.get('success'))
    accuracy = correct_count / successful_samples * 100 if successful_samples > 0 else 0
    
    # 按难度统计
    difficulty_stats = {}
    for diff in ['easy', 'medium', 'hard']:
        diff_results = [r for r in results if r['difficulty'] == diff]
        if diff_results:
            diff_correct = sum(1 for r in diff_results if r.get('correct'))
            diff_total = sum(1 for r in diff_results if r.get('success'))
            diff_acc = diff_correct / diff_total * 100 if diff_total > 0 else 0
            difficulty_stats[diff] = {
                'total': len(diff_results),
                'correct': diff_correct,
                'accuracy': diff_acc
            }
    
    # 输出结果
    print("\n" + "="*70)
    print("📊 测试结果")
    print("="*70)
    print(f"总样本数: {total_samples}")
    print(f"成功测试: {successful_samples}")
    print(f"整体准确率: {accuracy:.2f}% ({correct_count}/{successful_samples})")
    print(f"总tokens: {total_tokens}")
    print()
    
    print("按难度统计:")
    for diff in ['easy', 'medium', 'hard']:
        if diff in difficulty_stats:
            stats = difficulty_stats[diff]
            print(f"  {diff.capitalize()}: {stats['accuracy']:.2f}% ({stats['correct']}/{stats['total']})")
    
    # 保存结果
    if output_file is None:
        output_file = f"vlm_distractor_results_{model_name.replace('/', '-')}_{total_samples}samples.json"
    
    output_data = {
        'model': model_name,
        'dataset_file': dataset_file,
        'total_samples': total_samples,
        'successful_samples': successful_samples,
        'correct_count': correct_count,
        'overall_accuracy': accuracy,
        'total_tokens': total_tokens,
        'difficulty_stats': difficulty_stats,
        'results': results
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 详细结果已保存到: {output_file}")
    
    # 显示一些错误案例
    print("\n" + "="*70)
    print("❌ 错误案例示例（前5个）")
    print("="*70)
    error_cases = [r for r in results if r.get('success') and not r.get('correct')][:5]
    for case in error_cases:
        print(f"\n图片 {case['image_id']} ({case['difficulty']}):")
        print(f"  描述: {case['caption']}")
        print(f"  正确位置: {case['correct_position']}")
        print(f"  预测位置: {case['predicted_position']}")
        print(f"  推理: {case.get('reasoning', 'N/A')[:100]}...")
    
    return output_data

def quick_test(model_name="gpt-4o", n_samples=10):
    """快速测试（少量样本）"""
    print(f"🚀 快速测试模式（{n_samples}个样本）\n")
    return run_evaluation(
        dataset_file="distractor_dataset.json",
        model_name=model_name,
        max_samples=n_samples,
        difficulties=['easy']  # 只测试简单难度
    )

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='VLM图文匹配能力测试')
    parser.add_argument(
        '--model',
        type=str,
        default='gpt-4o',
        help='VLM模型名称'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='最大测试样本数'
    )
    parser.add_argument(
        '--difficulty',
        type=str,
        choices=['easy', 'medium', 'hard'],
        default=None,
        help='只测试指定难度'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='快速测试模式（10个简单样本）'
    )
    
    args = parser.parse_args()
    
    if args.quick:
        quick_test(model_name=args.model)
    else:
        difficulties = [args.difficulty] if args.difficulty else None
        run_evaluation(
            model_name=args.model,
            max_samples=args.max_samples,
            difficulties=difficulties
        )

if __name__ == "__main__":
    main()

