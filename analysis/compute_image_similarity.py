"""
基于图片描述计算相似度
使用sentence-transformers将描述转化为向量并计算相似度
"""
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import os

class ImageSimilarityComputer:
    def __init__(self, captions_file="image_captions.json", 
                 model_name="sentence-transformers/all-MiniLM-L6-v2"):
        """
        初始化相似度计算器
        
        Args:
            captions_file: 包含图片描述的JSON文件
            model_name: 使用的sentence-transformer模型
        """
        print(f"📥 加载图片描述...")
        with open(captions_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.captions_data = data['captions']
        self.image_ids = sorted(self.captions_data.keys(), key=lambda x: int(x))
        self.captions = [self.captions_data[img_id]['caption'] for img_id in self.image_ids]
        
        print(f"✅ 加载了 {len(self.captions)} 个图片描述")
        print(f"🔧 加载embedding模型: {model_name}")
        
        # 加载sentence-transformer模型
        self.model = SentenceTransformer(model_name)
        
        # 生成所有描述的向量
        print(f"🧮 计算向量表示...")
        self.embeddings = self.model.encode(self.captions, show_progress_bar=True)
        print(f"✅ 向量维度: {self.embeddings.shape}")
        
    def compute_similarity_matrix(self):
        """计算所有图片之间的相似度矩阵"""
        print(f"📊 计算相似度矩阵...")
        similarity_matrix = cosine_similarity(self.embeddings)
        return similarity_matrix
    
    def find_most_similar(self, image_id, top_k=5):
        """
        找到与指定图片最相似的其他图片
        
        Args:
            image_id: 图片ID（字符串，如"1", "2"等）
            top_k: 返回最相似的前k个图片
        
        Returns:
            List of (image_id, similarity_score, caption) tuples
        """
        if image_id not in self.image_ids:
            print(f"❌ 图片ID {image_id} 不存在")
            return []
        
        # 获取该图片的索引
        idx = self.image_ids.index(image_id)
        
        # 计算该图片与所有其他图片的相似度
        image_embedding = self.embeddings[idx:idx+1]
        similarities = cosine_similarity(image_embedding, self.embeddings)[0]
        
        # 获取最相似的图片（排除自己）
        similar_indices = np.argsort(similarities)[::-1][1:top_k+1]
        
        results = []
        for sim_idx in similar_indices:
            sim_image_id = self.image_ids[sim_idx]
            sim_score = similarities[sim_idx]
            sim_caption = self.captions[sim_idx]
            results.append((sim_image_id, sim_score, sim_caption))
        
        return results
    
    def save_similarity_matrix(self, output_file="similarity_matrix.csv"):
        """保存完整的相似度矩阵为CSV文件"""
        similarity_matrix = self.compute_similarity_matrix()
        df = pd.DataFrame(similarity_matrix, 
                         index=self.image_ids, 
                         columns=self.image_ids)
        df.to_csv(output_file)
        print(f"💾 相似度矩阵已保存到: {output_file}")
        return df
    
    def save_embeddings(self, output_file="image_embeddings.npz"):
        """保存图片描述的向量表示"""
        np.savez(output_file, 
                embeddings=self.embeddings,
                image_ids=np.array(self.image_ids),
                captions=np.array(self.captions))
        print(f"💾 向量表示已保存到: {output_file}")
    
    def generate_similarity_report(self, output_file="similarity_report.json"):
        """生成详细的相似度报告"""
        print(f"📝 生成相似度报告...")
        
        report = {
            'summary': {
                'total_images': len(self.image_ids),
                'embedding_model': str(self.model),
                'embedding_dimension': self.embeddings.shape[1]
            },
            'images': {}
        }
        
        for image_id in self.image_ids:
            similar_images = self.find_most_similar(image_id, top_k=5)
            
            report['images'][image_id] = {
                'caption': self.captions_data[image_id]['caption'],
                'most_similar': [
                    {
                        'image_id': sim_id,
                        'similarity': float(sim_score),
                        'caption': sim_caption
                    }
                    for sim_id, sim_score, sim_caption in similar_images
                ]
            }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 相似度报告已保存到: {output_file}")
        return report

def main():
    """主函数：执行完整的相似度分析流程"""
    
    # 检查是否存在描述文件
    if not os.path.exists("image_captions.json"):
        print("❌ 找不到 image_captions.json 文件")
        print("请先运行 generate_image_captions.py 生成图片描述")
        return
    
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
    
    # 打印一些示例
    print("\n" + "="*60)
    print("📋 相似度分析示例")
    print("="*60)
    
    # 随机选择几张图片展示最相似的图片
    example_ids = ['1', '10', '20', '30']
    for img_id in example_ids:
        if img_id in computer.image_ids:
            print(f"\n🖼️  图片 {img_id}: {computer.captions_data[img_id]['caption']}")
            print("   最相似的图片:")
            similar = computer.find_most_similar(img_id, top_k=3)
            for sim_id, sim_score, sim_caption in similar:
                print(f"      {sim_id}. {sim_caption} (相似度: {sim_score:.4f})")
    
    print("\n" + "="*60)
    print("✅ 所有分析完成！")
    print("="*60)
    print(f"📄 生成的文件：")
    print(f"   - similarity_matrix.csv: 完整的相似度矩阵")
    print(f"   - image_embeddings.npz: 图片描述的向量表示")
    print(f"   - similarity_report.json: 详细的相似度报告")

if __name__ == "__main__":
    main()

