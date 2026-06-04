"""
一键运行完整的图片相似度分析流程
包括生成描述、计算相似度、生成报告
"""
import os
import sys
import argparse

def check_dependencies():
    """检查必要的依赖是否已安装"""
    required_packages = [
        'sentence_transformers',
        'numpy',
        'pandas',
        'sklearn',
        'tqdm'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ 缺少以下依赖包:")
        for pkg in missing_packages:
            print(f"   - {pkg}")
        print("\n请运行以下命令安装依赖:")
        print("   pip install -r requirements.txt")
        return False
    
    return True

def run_complete_analysis(model_name="gpt-4o", skip_caption_generation=False):
    """
    运行完整的分析流程
    
    Args:
        model_name: 用于生成描述的VLM模型
        skip_caption_generation: 如果为True，跳过描述生成（使用已有的描述文件）
    """
    print("="*70)
    print("🚀 图片相似度分析系统 - 完整流程")
    print("="*70)
    
    # 检查依赖
    if not check_dependencies():
        return
    
    # 步骤1: 生成图片描述
    if not skip_caption_generation:
        print("\n📝 步骤 1/2: 生成图片描述")
        print("-"*70)
        
        if os.path.exists("image_captions.json"):
            response = input("⚠️  发现已有的描述文件 image_captions.json，是否重新生成? (y/n): ")
            if response.lower() != 'y':
                print("跳过描述生成，使用已有文件")
                skip_caption_generation = True
        
        if not skip_caption_generation:
            print(f"使用模型: {model_name}")
            print("⏰ 预计耗时: 2-10分钟（取决于图片数量和API速度）")
            print("💰 预计费用: $2-4 USD（GPT-4o）或 $0.5-1 USD（Qwen）\n")
            
            response = input("是否继续? (y/n): ")
            if response.lower() != 'y':
                print("❌ 已取消")
                return
            
            # 导入并运行描述生成
            from generate_image_captions import generate_all_captions
            generate_all_captions(
                images_dir="images",
                model_name=model_name,
                output_file="image_captions.json"
            )
    else:
        print("\n📝 步骤 1/2: 使用已有的图片描述")
        print("-"*70)
        if not os.path.exists("image_captions.json"):
            print("❌ 找不到 image_captions.json 文件")
            print("请先运行描述生成或移除 --skip-captions 参数")
            return
        print("✅ 找到已有的描述文件")
    
    # 步骤2: 计算相似度
    print("\n🧮 步骤 2/2: 计算相似度并生成报告")
    print("-"*70)
    print("⏰ 预计耗时: 10-30秒")
    print("💰 费用: 免费（本地计算）\n")
    
    from compute_image_similarity import ImageSimilarityComputer
    
    # 初始化相似度计算器
    computer = ImageSimilarityComputer(
        captions_file="image_captions.json",
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # 保存相似度矩阵
    similarity_df = computer.save_similarity_matrix("similarity_matrix.csv")
    
    # 保存向量表示
    computer.save_embeddings("image_embeddings.npz")
    
    # 生成详细报告
    report = computer.generate_similarity_report("similarity_report.json")
    
    # 显示统计信息
    print("\n" + "="*70)
    print("📊 分析完成！统计信息")
    print("="*70)
    print(f"✅ 分析了 {len(computer.image_ids)} 张图片")
    print(f"✅ 向量维度: {computer.embeddings.shape[1]}")
    print(f"\n📄 生成的文件：")
    print(f"   1. image_captions.json - 图片描述数据")
    print(f"   2. similarity_matrix.csv - 完整的相似度矩阵")
    print(f"   3. image_embeddings.npz - 向量表示（可复用）")
    print(f"   4. similarity_report.json - 详细的相似度报告")
    
    # 显示一些示例
    print("\n" + "="*70)
    print("📋 相似度分析示例（前3张图片）")
    print("="*70)
    
    for img_id in computer.image_ids[:3]:
        print(f"\n🖼️  图片 {img_id}: {computer.captions_data[img_id]['caption']}")
        print("   最相似的3张图片:")
        similar = computer.find_most_similar(img_id, top_k=3)
        for i, (sim_id, sim_score, sim_caption) in enumerate(similar, 1):
            print(f"      {i}. 图片{sim_id}: {sim_caption} (相似度: {sim_score:.4f})")
    
    # 提示后续操作
    print("\n" + "="*70)
    print("🎯 后续操作")
    print("="*70)
    print("查询特定图片的相似图片:")
    print("   python query_image_similarity.py")
    print("   或")
    print("   python query_image_similarity.py <图片ID>")
    print("="*70)

def main():
    parser = argparse.ArgumentParser(description='图片相似度分析 - 完整流程')
    parser.add_argument(
        '--model',
        type=str,
        default='gpt-4o',
        choices=['gpt-4o', 'gemini-2.5-flash', 'qwen2.5-vl-7b', 'qwen2.5-vl-32b'],
        help='用于生成描述的VLM模型（默认: gpt-4o）'
    )
    parser.add_argument(
        '--skip-captions',
        action='store_true',
        help='跳过描述生成，使用已有的image_captions.json文件'
    )
    
    args = parser.parse_args()
    
    run_complete_analysis(
        model_name=args.model,
        skip_caption_generation=args.skip_captions
    )

if __name__ == "__main__":
    main()

