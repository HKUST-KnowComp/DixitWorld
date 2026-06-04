"""
为所有Dixit图片生成简短描述（caption）
使用VLM模型分析图片并生成描述性短语
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

def generate_caption_for_image(image_path, model_name="gpt-4o"):
    """
    为单张图片生成简短描述
    
    Args:
        image_path: 图片路径
        model_name: 使用的VLM模型名称
    
    Returns:
        caption: 图片的简短描述
    """
    # 读取并编码图片
    base64_image = encode_image_to_base64(image_path)
    
    # 构建prompt - 要求模型返回简短的描述性短语
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": """Please provide a short, descriptive caption for this image (10-20 words). 
Focus on the main subject, mood, and key visual elements. 
Be concise and specific.

Return your response in JSON format:
{
    "reasoning": "Brief analysis of the image",
    "answer": "The short caption"
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
            temperature=0.3,  # 较低的温度以获得更一致的描述
            trial_info={'trial_id': os.path.basename(image_path)}
        )
        
        if error:
            print(f"⚠️  生成描述时出错: {error}")
            return None
        
        # 提取caption
        caption = response.get('answer', '').strip()
        reasoning = response.get('reasoning', '')
        
        return {
            'caption': caption,
            'reasoning': reasoning,
            'tokens': tokens
        }
        
    except Exception as e:
        print(f"❌ 处理图片 {image_path} 时出错: {e}")
        return None

def generate_all_captions(images_dir="images", model_name="gpt-4o", output_file="image_captions.json"):
    """
    为所有图片生成描述并保存
    
    Args:
        images_dir: 图片文件夹路径
        model_name: 使用的VLM模型
        output_file: 输出JSON文件路径
    """
    # 获取所有PNG图片
    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.png')],
                        key=lambda x: int(x.split('.')[0]))
    
    print(f"找到 {len(image_files)} 张图片")
    print(f"使用模型: {model_name}")
    print(f"开始生成描述...\n")
    
    captions_data = {}
    total_tokens = 0
    
    # 使用进度条处理每张图片
    for image_file in tqdm(image_files, desc="生成描述"):
        image_path = os.path.join(images_dir, image_file)
        image_id = image_file.split('.')[0]  # 例如: "1", "2", etc.
        
        # 生成描述
        result = generate_caption_for_image(image_path, model_name)
        
        if result:
            captions_data[image_id] = {
                'filename': image_file,
                'caption': result['caption'],
                'reasoning': result['reasoning'],
                'tokens': result['tokens']
            }
            total_tokens += result['tokens']
            
            # 打印进度
            print(f"\n图片 {image_id}: {result['caption']}")
        else:
            print(f"\n⚠️  跳过图片 {image_id}")
        
        # 添加短暂延迟避免API限流
        time.sleep(0.5)
    
    # 保存到JSON文件
    output_data = {
        'model': model_name,
        'total_images': len(captions_data),
        'total_tokens': total_tokens,
        'captions': captions_data
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 完成！")
    print(f"📊 生成了 {len(captions_data)} 个描述")
    print(f"💰 总共使用 {total_tokens} tokens")
    print(f"💾 已保存到: {output_file}")
    
    return output_data

if __name__ == "__main__":
    # 可以选择不同的模型：gpt-4o, gemini-2.5-flash, qwen2.5-vl-7b, qwen2.5-vl-32b
    generate_all_captions(
        images_dir="images",
        model_name="gpt-4o",  # 推荐使用GPT-4o，视觉理解能力强
        output_file="image_captions.json"
    )

