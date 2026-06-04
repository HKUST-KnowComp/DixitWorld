"""
基于相似度矩阵构建干扰图数据集
用于测试VLM模型的图文匹配能力
"""
import json
import numpy as np
import pandas as pd
import random
from pathlib import Path

class DistractorDatasetBuilder:
    def __init__(self, similarity_matrix_file="similarity_matrix.csv",
                 captions_file="image_captions.json",
                 images_dir="images"):
        """
        初始化数据集构建器
        
        Args:
            similarity_matrix_file: 相似度矩阵CSV文件
            captions_file: 图片描述JSON文件
            images_dir: 图片文件夹路径
        """
        print("📥 加载数据...")
        
        # 加载相似度矩阵
        self.similarity_df = pd.read_csv(similarity_matrix_file, index_col=0)
        self.similarity_matrix = self.similarity_df.values
        # 将索引转换为字符串以保持一致性
        self.image_ids = [str(x) for x in self.similarity_df.index.tolist()]
        
        # 加载图片描述
        with open(captions_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.captions_data = data['captions']
        
        self.images_dir = images_dir
        
        print(f"✅ 加载了 {len(self.image_ids)} 张图片的相似度数据")
    
    def get_distractors_for_image(self, image_id, difficulty="hard", n=5, seed=None):
        """
        为指定图片获取干扰图
        
        Args:
            image_id: 目标图片ID
            difficulty: 难度级别 ("hard", "medium", "easy")
            n: 返回的干扰图数量
            seed: 随机种子（用于medium和easy）
        
        Returns:
            List of distractor image IDs
        """
        if seed is not None:
            random.seed(seed)
        
        # 获取该图片的相似度排序
        idx = self.image_ids.index(str(image_id))
        similarities = self.similarity_matrix[idx]
        
        # 排序（从高到低），排除自己
        sorted_indices = np.argsort(similarities)[::-1]
        sorted_indices = [i for i in sorted_indices if i != idx]
        
        # 根据难度选择
        if difficulty == "hard":
            # 最相似的前n个（1-n）
            selected_indices = sorted_indices[:n]
        elif difficulty == "medium":
            # 排名10-20中随机选n个
            pool = sorted_indices[9:20]  # Python索引从0开始
            if len(pool) < n:
                pool = sorted_indices[9:9+n*2]  # 扩大范围
            selected_indices = random.sample(pool, min(n, len(pool)))
        elif difficulty == "easy":
            # 排名30-80中随机选n个
            pool = sorted_indices[29:80]
            if len(pool) < n:
                pool = sorted_indices[29:]  # 使用所有剩余的
            selected_indices = random.sample(pool, min(n, len(pool)))
        else:
            raise ValueError(f"Unknown difficulty: {difficulty}")
        
        # 转换为图片ID
        distractor_ids = [self.image_ids[i] for i in selected_indices]
        return distractor_ids
    
    def build_distractor_list(self, output_file="distractor_lists.json", seed=42):
        """
        STEP 1: 为所有图片构建干扰图列表
        
        Args:
            output_file: 输出JSON文件路径
            seed: 随机种子
        """
        print("\n" + "="*70)
        print("📝 STEP 1: 构造干扰图列表")
        print("="*70)
        
        distractor_data = []
        
        for image_id in self.image_ids:
            # 获取三个难度的干扰图
            hard_distractors = self.get_distractors_for_image(
                image_id, difficulty="hard", n=5
            )
            medium_distractors = self.get_distractors_for_image(
                image_id, difficulty="medium", n=5, seed=seed+int(image_id)
            )
            easy_distractors = self.get_distractors_for_image(
                image_id, difficulty="easy", n=5, seed=seed+int(image_id)+1000
            )
            
            # 构建数据项
            item = {
                "id": int(image_id),
                "target_caption": self.captions_data[image_id]['caption'],
                "distractors": {
                    "hard": [int(x) for x in hard_distractors],
                    "medium": [int(x) for x in medium_distractors],
                    "easy": [int(x) for x in easy_distractors]
                }
            }
            distractor_data.append(item)
        
        # 保存到JSON
        output = {
            "metadata": {
                "total_images": len(self.image_ids),
                "distractors_per_difficulty": 5,
                "difficulties": ["hard", "medium", "easy"],
                "seed": seed
            },
            "data": distractor_data
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 生成了 {len(distractor_data)} 个图片的干扰图列表")
        print(f"💾 已保存到: {output_file}")
        
        # 显示示例
        print("\n📋 示例（图片1）:")
        example = distractor_data[0]
        print(f"  目标: 图片{example['id']}")
        print(f"  描述: {example['target_caption']}")
        print(f"  Hard干扰图: {example['distractors']['hard']}")
        print(f"  Medium干扰图: {example['distractors']['medium']}")
        print(f"  Easy干扰图: {example['distractors']['easy']}")
        
        return distractor_data
    
    def build_dataset_items(self, distractor_list_file="distractor_lists.json",
                           output_file="distractor_dataset.json"):
        """
        STEP 2: 构建完整的数据集样本
        
        Args:
            distractor_list_file: 干扰图列表文件
            output_file: 输出数据集文件
        """
        print("\n" + "="*70)
        print("📝 STEP 2: 构建数据集样本")
        print("="*70)
        
        # 加载干扰图列表
        with open(distractor_list_file, 'r', encoding='utf-8') as f:
            distractor_data = json.load(f)
        
        dataset_items = []
        
        # 为每个图片、每个难度生成一个样本
        for item in distractor_data['data']:
            image_id = item['id']
            target_caption = item['target_caption']
            
            for difficulty in ['hard', 'medium', 'easy']:
                distractors = item['distractors'][difficulty]
                
                # 构建样本
                dataset_item = {
                    "image_id": image_id,
                    "caption_type": "phrase",  # 当前只有phrase级别
                    "difficulty": difficulty,
                    "target": {
                        "img": f"{self.images_dir}/{image_id}.png",
                        "caption": target_caption
                    },
                    "distractors": [
                        {"img": f"{self.images_dir}/{dist_id}.png"}
                        for dist_id in distractors
                    ]
                }
                
                dataset_items.append(dataset_item)
        
        # 保存数据集
        output = {
            "metadata": {
                "total_samples": len(dataset_items),
                "total_images": len(distractor_data['data']),
                "difficulties": ["hard", "medium", "easy"],
                "caption_type": "phrase",
                "distractors_per_sample": 5,
                "task": "image-text matching"
            },
            "dataset": dataset_items
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 生成了 {len(dataset_items)} 个数据集样本")
        print(f"   - {len(distractor_data['data'])} 张图片 × 3 个难度")
        print(f"💾 已保存到: {output_file}")
        
        # 显示统计
        print("\n📊 数据集统计:")
        difficulties = {}
        for item in dataset_items:
            diff = item['difficulty']
            difficulties[diff] = difficulties.get(diff, 0) + 1
        
        for diff, count in difficulties.items():
            print(f"   - {diff.capitalize()}: {count} 个样本")
        
        # 显示示例
        print("\n📋 数据集样本示例:")
        example = dataset_items[0]
        print(f"  图片ID: {example['image_id']}")
        print(f"  难度: {example['difficulty']}")
        print(f"  目标图片: {example['target']['img']}")
        print(f"  目标描述: {example['target']['caption']}")
        print(f"  干扰图片: {[d['img'] for d in example['distractors']]}")
        
        return dataset_items
    
    def analyze_distractor_quality(self, distractor_list_file="distractor_lists.json"):
        """
        分析干扰图质量
        """
        print("\n" + "="*70)
        print("📊 干扰图质量分析")
        print("="*70)
        
        with open(distractor_list_file, 'r', encoding='utf-8') as f:
            distractor_data = json.load(f)
        
        # 统计相似度分布
        hard_sims = []
        medium_sims = []
        easy_sims = []
        
        for item in distractor_data['data']:
            image_id = str(item['id'])
            idx = self.image_ids.index(image_id)
            
            for dist_id in item['distractors']['hard']:
                dist_idx = self.image_ids.index(str(dist_id))
                hard_sims.append(self.similarity_matrix[idx][dist_idx])
            
            for dist_id in item['distractors']['medium']:
                dist_idx = self.image_ids.index(str(dist_id))
                medium_sims.append(self.similarity_matrix[idx][dist_idx])
            
            for dist_id in item['distractors']['easy']:
                dist_idx = self.image_ids.index(str(dist_id))
                easy_sims.append(self.similarity_matrix[idx][dist_idx])
        
        # 显示统计结果
        print("\n相似度分布（与目标图片）:")
        print(f"  Hard (最相似):")
        print(f"    平均: {np.mean(hard_sims):.4f}")
        print(f"    范围: {np.min(hard_sims):.4f} - {np.max(hard_sims):.4f}")
        
        print(f"\n  Medium (中等相似):")
        print(f"    平均: {np.mean(medium_sims):.4f}")
        print(f"    范围: {np.min(medium_sims):.4f} - {np.max(medium_sims):.4f}")
        
        print(f"\n  Easy (低相似):")
        print(f"    平均: {np.mean(easy_sims):.4f}")
        print(f"    范围: {np.min(easy_sims):.4f} - {np.max(easy_sims):.4f}")
        
        # 验证难度梯度
        print("\n✅ 难度梯度验证:")
        if np.mean(hard_sims) > np.mean(medium_sims) > np.mean(easy_sims):
            print("   ✓ 难度梯度正确: Hard > Medium > Easy")
        else:
            print("   ⚠ 难度梯度异常")

def main():
    """主函数：运行完整流程"""
    print("="*70)
    print("🏗️  干扰图数据集构建系统")
    print("="*70)
    
    # 检查必需文件
    import os
    if not os.path.exists("similarity_matrix.csv"):
        print("❌ 找不到 similarity_matrix.csv")
        print("请先运行: python compute_image_similarity.py")
        return
    
    if not os.path.exists("image_captions.json"):
        print("❌ 找不到 image_captions.json")
        print("请先运行: python generate_image_captions.py")
        return
    
    # 初始化构建器
    builder = DistractorDatasetBuilder(
        similarity_matrix_file="similarity_matrix.csv",
        captions_file="image_captions.json",
        images_dir="images"
    )
    
    # STEP 1: 构造干扰图列表
    distractor_list = builder.build_distractor_list(
        output_file="distractor_lists.json",
        seed=42
    )
    
    # STEP 2: 构建数据集样本
    dataset_items = builder.build_dataset_items(
        distractor_list_file="distractor_lists.json",
        output_file="distractor_dataset.json"
    )
    
    # 分析干扰图质量
    builder.analyze_distractor_quality("distractor_lists.json")
    
    print("\n" + "="*70)
    print("✅ 数据集构建完成！")
    print("="*70)
    print("生成的文件:")
    print("  1. distractor_lists.json - 每张图的干扰图列表")
    print("  2. distractor_dataset.json - 完整的数据集样本")
    print("\n使用方法:")
    print("  python test_vlm_with_distractors.py")

if __name__ == "__main__":
    main()

