"""
交互式查询图片相似度
可以查询任意图片与其他图片的相似度
"""
import json
import sys
from compute_image_similarity import ImageSimilarityComputer

def interactive_query():
    """交互式查询界面"""
    
    print("="*60)
    print("🔍 图片相似度查询工具")
    print("="*60)
    
    # 初始化
    try:
        computer = ImageSimilarityComputer(
            captions_file="image_captions.json",
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    except FileNotFoundError:
        print("❌ 找不到 image_captions.json 文件")
        print("请先运行 generate_image_captions.py 生成图片描述")
        return
    
    print(f"\n可查询的图片ID: 1 到 {len(computer.image_ids)}")
    print("输入 'quit' 或 'exit' 退出程序\n")
    
    while True:
        # 获取用户输入
        user_input = input("请输入图片ID（例如：1）：").strip()
        
        # 检查退出命令
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("👋 再见！")
            break
        
        # 验证输入
        if user_input not in computer.image_ids:
            print(f"❌ 无效的图片ID。请输入 1 到 {len(computer.image_ids)} 之间的数字\n")
            continue
        
        # 显示该图片的信息
        print("\n" + "-"*60)
        print(f"🖼️  图片 {user_input}")
        print(f"📝 描述: {computer.captions_data[user_input]['caption']}")
        
        # 获取并显示最相似的图片
        print("\n🎯 最相似的5张图片:")
        similar_images = computer.find_most_similar(user_input, top_k=5)
        
        for i, (sim_id, sim_score, sim_caption) in enumerate(similar_images, 1):
            print(f"\n   {i}. 图片 {sim_id} (相似度: {sim_score:.4f})")
            print(f"      描述: {sim_caption}")
        
        print("-"*60 + "\n")

def query_specific_image(image_id, top_k=5):
    """
    查询特定图片的相似图片（用于脚本调用）
    
    Args:
        image_id: 图片ID
        top_k: 返回最相似的前k个图片
    """
    computer = ImageSimilarityComputer(
        captions_file="image_captions.json",
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    print(f"🖼️  图片 {image_id}")
    print(f"📝 描述: {computer.captions_data[image_id]['caption']}")
    print(f"\n🎯 最相似的{top_k}张图片:")
    
    similar_images = computer.find_most_similar(image_id, top_k=top_k)
    
    for i, (sim_id, sim_score, sim_caption) in enumerate(similar_images, 1):
        print(f"\n   {i}. 图片 {sim_id} (相似度: {sim_score:.4f})")
        print(f"      描述: {sim_caption}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # 命令行模式：python query_image_similarity.py 1
        image_id = sys.argv[1]
        top_k = int(sys.argv[2]) if len(sys.argv) > 2 else 5
        query_specific_image(image_id, top_k)
    else:
        # 交互模式
        interactive_query()

