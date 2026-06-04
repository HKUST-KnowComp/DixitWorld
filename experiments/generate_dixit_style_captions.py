"""
生成Dixit游戏风格的抽象短描述
使用Gemma3模型生成简短、抽象、富有想象力的描述
"""
import os
import json
import base64
from call_api_openrouter import call_llm_api
from tqdm import tqdm
import time

def encode_image_to_base64(image_path):
    """将图片编码为base64字符串"""
    with open(image_path, 'rb') as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def generate_dixit_caption(image_path, model_name="gemma3-27b"):
    """
    为单张图片生成Dixit风格的抽象短描述
    
    Args:
        image_path: 图片路径
        model_name: 使用的VLM模型
    
    Returns:
        caption: 简短抽象的描述
    """
    # 读取并编码图片
    base64_image = encode_image_to_base64(image_path)
    
    # 构建符合Dixit游戏规则的prompt
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": """You are playing Dixit, a creative card game. In Dixit, players use abstract, poetic, and imaginative phrases to describe images.

RULES:
- Keep it VERY SHORT: 1-4 words maximum
- Be ABSTRACT and POETIC, not literal
- Use metaphors, emotions, or associations
- Be creative and imaginative
- Avoid direct descriptions

EXAMPLES of good Dixit clues:
- "Lost innocence"
- "Breaking free"
- "Endless hope"
- "Fractured dream"
- "Silent whisper"

Now look at this image and create a Dixit-style clue.

Return ONLY the clue in JSON format:
{
    "reasoning": "Brief thought process",
    "answer": "Your 1-4 word abstract clue"
}"""
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image}"
                    }
                }
            ]
        }
    ]
    
    try:
        # 调用VLM API
        response, error, tokens = call_llm_api(
            messages=messages,
            model_name=model_name,
            temperature=0.7,  # 较高温度以获得更有创意的描述
            trial_info={'trial_id': os.path.basename(image_path)}
        )
        
        if error:
            print(f"⚠️  生成描述时出错: {error}")
            return None
        
        # 提取caption
        caption = response.get('answer', '').strip()
        reasoning = response.get('reasoning', '')
        
        # 清理caption - 移除引号等
        caption = caption.strip('"\'.,!?')
        
        return {
            'caption': caption,
            'reasoning': reasoning,
            'tokens': tokens
        }
        
    except Exception as e:
        print(f"❌ 处理图片 {image_path} 时出错: {e}")
        return None

def update_all_captions(images_dir="images", 
                       old_captions_file="image_captions.json",
                       new_captions_file="image_captions_dixit.json",
                       model_name="gemma3-27b"):
    """
    更新所有图片的描述为Dixit风格
    
    Args:
        images_dir: 图片文件夹路径
        old_captions_file: 旧的描述文件
        new_captions_file: 新的描述文件
        model_name: 使用的VLM模型
    """
    print("="*70)
    print("🎨 生成Dixit风格抽象描述")
    print("="*70)
    print(f"模型: {model_name}")
    print(f"风格: 抽象、简短（1-4词）")
    print()
    
    # 加载旧的描述文件以保持结构
    with open(old_captions_file, 'r', encoding='utf-8') as f:
        old_data = json.load(f)
    
    image_ids = sorted(old_data['captions'].keys(), key=lambda x: int(x))
    
    print(f"找到 {len(image_ids)} 张图片")
    print(f"开始生成Dixit风格描述...\n")
    
    captions_data = {}
    total_tokens = 0
    
    # 使用进度条处理每张图片
    for image_id in tqdm(image_ids, desc="生成描述"):
        image_file = f"{image_id}.png"
        image_path = os.path.join(images_dir, image_file)
        
        # 生成描述
        result = generate_dixit_caption(image_path, model_name)
        
        if result:
            captions_data[image_id] = {
                'filename': image_file,
                'caption': result['caption'],
                'reasoning': result['reasoning'],
                'tokens': result['tokens'],
                'old_caption': old_data['captions'][image_id]['caption']  # 保留旧描述以便对比
            }
            total_tokens += result['tokens']
            
            # 打印进度
            print(f"\n图片 {image_id}:")
            print(f"  旧: {old_data['captions'][image_id]['caption']}")
            print(f"  新: {result['caption']}")
        else:
            print(f"\n⚠️  跳过图片 {image_id}")
        
        # 添加短暂延迟避免API限流
        time.sleep(0.5)
    
    # 保存到新JSON文件
    output_data = {
        'model': model_name,
        'style': 'dixit_abstract',
        'total_images': len(captions_data),
        'total_tokens': total_tokens,
        'captions': captions_data
    }
    
    with open(new_captions_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 完成！")
    print(f"📊 生成了 {len(captions_data)} 个Dixit风格描述")
    print(f"💰 总共使用 {total_tokens} tokens")
    print(f"💾 已保存到: {new_captions_file}")
    
    # 显示一些统计
    word_counts = []
    for data in captions_data.values():
        word_count = len(data['caption'].split())
        word_counts.append(word_count)
    
    import statistics
    print(f"\n📈 描述长度统计:")
    print(f"   平均: {statistics.mean(word_counts):.1f} 词")
    print(f"   最短: {min(word_counts)} 词")
    print(f"   最长: {max(word_counts)} 词")
    
    # 显示一些示例
    print(f"\n📋 随机示例:")
    import random
    examples = random.sample(list(captions_data.items()), min(5, len(captions_data)))
    for img_id, data in examples:
        print(f"   图片{img_id}: '{data['caption']}'")
    
    return output_data

def replace_captions_in_files(new_captions_file="image_captions_dixit.json"):
    """
    用新的Dixit风格描述替换相关文件中的描述
    """
    print("\n" + "="*70)
    print("🔄 更新相关文件中的描述")
    print("="*70)
    
    # 加载新描述
    with open(new_captions_file, 'r', encoding='utf-8') as f:
        new_data = json.load(f)
    new_captions = {k: v['caption'] for k, v in new_data['captions'].items()}
    
    # 1. 替换 image_captions.json（备份旧文件）
    print("\n1. 更新 image_captions.json...")
    if os.path.exists('image_captions.json'):
        # 备份
        import shutil
        shutil.copy('image_captions.json', 'image_captions_old.json')
        print("   ✓ 已备份旧文件到 image_captions_old.json")
        
        # 更新
        with open('image_captions.json', 'r', encoding='utf-8') as f:
            old_data = json.load(f)
        
        # 只更新caption字段
        for img_id in new_captions:
            if img_id in old_data['captions']:
                old_data['captions'][img_id]['caption'] = new_captions[img_id]
        
        old_data['model'] = new_data['model']
        old_data['style'] = 'dixit_abstract'
        
        with open('image_captions.json', 'w', encoding='utf-8') as f:
            json.dump(old_data, f, ensure_ascii=False, indent=2)
        print("   ✓ 已更新 image_captions.json")
    
    # 2. 更新 distractor_lists.json
    print("\n2. 更新 distractor_lists.json...")
    if os.path.exists('distractor_lists.json'):
        with open('distractor_lists.json', 'r', encoding='utf-8') as f:
            dist_data = json.load(f)
        
        for item in dist_data['data']:
            img_id = str(item['id'])
            if img_id in new_captions:
                item['target_caption'] = new_captions[img_id]
        
        with open('distractor_lists.json', 'w', encoding='utf-8') as f:
            json.dump(dist_data, f, ensure_ascii=False, indent=2)
        print("   ✓ 已更新 distractor_lists.json")
    
    # 3. 更新 distractor_dataset.json
    print("\n3. 更新 distractor_dataset.json...")
    if os.path.exists('distractor_dataset.json'):
        with open('distractor_dataset.json', 'r', encoding='utf-8') as f:
            dataset_data = json.load(f)
        
        for item in dataset_data['dataset']:
            img_id = str(item['image_id'])
            if img_id in new_captions:
                item['target']['caption'] = new_captions[img_id]
        
        # 更新元数据
        dataset_data['metadata']['caption_style'] = 'dixit_abstract'
        dataset_data['metadata']['caption_length'] = '1-4 words'
        
        with open('distractor_dataset.json', 'w', encoding='utf-8') as f:
            json.dump(dataset_data, f, ensure_ascii=False, indent=2)
        print("   ✓ 已更新 distractor_dataset.json")
    
    # 4. 更新 similarity_report.json
    print("\n4. 更新 similarity_report.json...")
    if os.path.exists('similarity_report.json'):
        with open('similarity_report.json', 'r', encoding='utf-8') as f:
            report_data = json.load(f)
        
        for img_id in report_data['images']:
            if img_id in new_captions:
                report_data['images'][img_id]['caption'] = new_captions[img_id]
        
        with open('similarity_report.json', 'w', encoding='utf-8') as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)
        print("   ✓ 已更新 similarity_report.json")
    
    print("\n✅ 所有文件已更新！")
    print("\n📝 备份文件:")
    print("   - image_captions_old.json (原始详细描述)")
    print("   - image_captions_dixit.json (新Dixit风格描述)")

def main():
    """主函数"""
    print("\n" + "🎨"*35)
    print("Dixit风格抽象描述生成器")
    print("🎨"*35 + "\n")
    
    # 检查必需文件
    if not os.path.exists("image_captions.json"):
        print("❌ 找不到 image_captions.json")
        print("请先运行: python generate_image_captions.py")
        return
    
    # 生成新的Dixit风格描述
    new_data = update_all_captions(
        images_dir="images",
        old_captions_file="image_captions.json",
        new_captions_file="image_captions_dixit.json",
        model_name="gemma3-27b"  # 使用Gemma3-27B
    )
    
    # 替换所有相关文件中的描述
    replace_captions_in_files("image_captions_dixit.json")
    
    print("\n" + "="*70)
    print("🎉 全部完成！")
    print("="*70)
    print("\n现在所有文件都使用Dixit风格的抽象短描述了！")
    print("相似度矩阵保持不变，可以直接用于测试。")

if __name__ == "__main__":
    main()

