"""
将distractor JSON文件转化为可视化的Markdown文档
包含图片展示，方便人工检查
"""
import json
import os

def generate_distractor_lists_markdown(json_file="distractor_lists.json", 
                                       output_file="DISTRACTOR_LISTS_VISUAL.md"):
    """
    生成distractor_lists的可视化markdown
    """
    print(f"📖 生成 {json_file} 的可视化文档...")
    
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    md_lines = []
    
    # 标题和说明
    md_lines.append("# 🎨 Dixit干扰图列表 - 可视化检查")
    md_lines.append("")
    md_lines.append("本文档展示每张目标图片及其对应的三个难度级别的干扰图。")
    md_lines.append("")
    md_lines.append("## 📊 数据集信息")
    md_lines.append("")
    md_lines.append(f"- **总图片数**: {data['metadata']['total_images']}")
    md_lines.append(f"- **每个难度的干扰图数**: {data['metadata']['distractors_per_difficulty']}")
    md_lines.append(f"- **难度级别**: {', '.join(data['metadata']['difficulties'])}")
    md_lines.append(f"- **随机种子**: {data['metadata']['seed']}")
    md_lines.append("")
    md_lines.append("---")
    md_lines.append("")
    
    # 为每张图片生成一个section
    for item in data['data']:
        img_id = item['id']
        caption = item['target_caption']
        
        md_lines.append(f"## 图片 {img_id}: {caption}")
        md_lines.append("")
        
        # 显示目标图片
        md_lines.append("### 🎯 目标图片")
        md_lines.append("")
        md_lines.append(f"**描述**: {caption}")
        md_lines.append("")
        md_lines.append(f"![图片{img_id}](images/{img_id}.png)")
        md_lines.append("")
        
        # Hard难度干扰图
        md_lines.append("### 🔴 Hard难度干扰图（最相似）")
        md_lines.append("")
        md_lines.append("| 图片1 | 图片2 | 图片3 | 图片4 | 图片5 |")
        md_lines.append("|-------|-------|-------|-------|-------|")
        
        # 第一行：图片
        img_row = "| "
        for dist_id in item['distractors']['hard']:
            img_row += f"![{dist_id}](images/{dist_id}.png) | "
        md_lines.append(img_row)
        
        # 第二行：ID
        id_row = "| "
        for dist_id in item['distractors']['hard']:
            id_row += f"**图片{dist_id}** | "
        md_lines.append(id_row)
        md_lines.append("")
        
        # Medium难度干扰图
        md_lines.append("### 🟡 Medium难度干扰图（中等相似）")
        md_lines.append("")
        md_lines.append("| 图片1 | 图片2 | 图片3 | 图片4 | 图片5 |")
        md_lines.append("|-------|-------|-------|-------|-------|")
        
        img_row = "| "
        for dist_id in item['distractors']['medium']:
            img_row += f"![{dist_id}](images/{dist_id}.png) | "
        md_lines.append(img_row)
        
        id_row = "| "
        for dist_id in item['distractors']['medium']:
            id_row += f"**图片{dist_id}** | "
        md_lines.append(id_row)
        md_lines.append("")
        
        # Easy难度干扰图
        md_lines.append("### 🟢 Easy难度干扰图（低相似）")
        md_lines.append("")
        md_lines.append("| 图片1 | 图片2 | 图片3 | 图片4 | 图片5 |")
        md_lines.append("|-------|-------|-------|-------|-------|")
        
        img_row = "| "
        for dist_id in item['distractors']['easy']:
            img_row += f"![{dist_id}](images/{dist_id}.png) | "
        md_lines.append(img_row)
        
        id_row = "| "
        for dist_id in item['distractors']['easy']:
            id_row += f"**图片{dist_id}** | "
        md_lines.append(id_row)
        md_lines.append("")
        
        md_lines.append("---")
        md_lines.append("")
    
    # 写入文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md_lines))
    
    print(f"✅ 已生成: {output_file}")
    return output_file

def generate_dataset_samples_markdown(json_file="distractor_dataset.json",
                                     output_file="DISTRACTOR_DATASET_SAMPLES.md",
                                     max_samples_per_difficulty=10):
    """
    生成distractor_dataset的示例样本可视化markdown
    """
    print(f"📖 生成 {json_file} 的可视化文档...")
    
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    md_lines = []
    
    # 标题和说明
    md_lines.append("# 🎯 Dixit测试数据集样本 - 可视化检查")
    md_lines.append("")
    md_lines.append("本文档展示测试数据集的样本，每个样本包含1个目标图片+5个干扰图。")
    md_lines.append("")
    md_lines.append("## 📊 数据集信息")
    md_lines.append("")
    md_lines.append(f"- **总样本数**: {data['metadata']['total_samples']}")
    md_lines.append(f"- **总图片数**: {data['metadata']['total_images']}")
    md_lines.append(f"- **难度级别**: {', '.join(data['metadata']['difficulties'])}")
    md_lines.append(f"- **Caption类型**: {data['metadata']['caption_type']}")
    md_lines.append(f"- **每个样本的干扰图数**: {data['metadata']['distractors_per_sample']}")
    md_lines.append(f"- **任务类型**: {data['metadata']['task']}")
    md_lines.append("")
    md_lines.append(f"**说明**: 本文档每个难度展示前{max_samples_per_difficulty}个样本")
    md_lines.append("")
    md_lines.append("---")
    md_lines.append("")
    
    # 按难度分组
    for difficulty in ['hard', 'medium', 'easy']:
        md_lines.append(f"# {'🔴' if difficulty == 'hard' else '🟡' if difficulty == 'medium' else '🟢'} {difficulty.upper()}难度样本")
        md_lines.append("")
        
        # 筛选该难度的样本
        samples = [s for s in data['dataset'] if s['difficulty'] == difficulty]
        
        # 只显示前N个
        samples_to_show = samples[:max_samples_per_difficulty]
        
        for idx, sample in enumerate(samples_to_show, 1):
            img_id = sample['image_id']
            caption = sample['target']['caption']
            target_img = sample['target']['img']
            
            md_lines.append(f"## 样本 {idx}: 图片{img_id} - {caption}")
            md_lines.append("")
            md_lines.append(f"**任务**: 从以下6张图片中选出与描述「**{caption}**」匹配的图片")
            md_lines.append("")
            
            # 创建一个包含目标+干扰的表格
            md_lines.append("### 🎲 所有图片（1个目标 + 5个干扰）")
            md_lines.append("")
            
            # 分两行显示（3张 + 3张）
            # 第一行：前3张
            md_lines.append("| 选项A | 选项B | 选项C |")
            md_lines.append("|-------|-------|-------|")
            
            all_imgs = [{'img': target_img, 'is_target': True}] + \
                      [{'img': d['img'], 'is_target': False} for d in sample['distractors']]
            
            # 前3张
            img_row = "| "
            for i in range(3):
                if i < len(all_imgs):
                    img_path = all_imgs[i]['img']
                    img_row += f"![{os.path.basename(img_path)}]({img_path}) | "
            md_lines.append(img_row)
            
            label_row = "| "
            for i in range(3):
                if i < len(all_imgs):
                    img_num = os.path.basename(all_imgs[i]['img']).replace('.png', '')
                    if all_imgs[i]['is_target']:
                        label_row += f"**图片{img_num}** ✅ | "
                    else:
                        label_row += f"图片{img_num} | "
            md_lines.append(label_row)
            md_lines.append("")
            
            # 后3张
            md_lines.append("| 选项D | 选项E | 选项F |")
            md_lines.append("|-------|-------|-------|")
            
            img_row = "| "
            for i in range(3, 6):
                if i < len(all_imgs):
                    img_path = all_imgs[i]['img']
                    img_row += f"![{os.path.basename(img_path)}]({img_path}) | "
            md_lines.append(img_row)
            
            label_row = "| "
            for i in range(3, 6):
                if i < len(all_imgs):
                    img_num = os.path.basename(all_imgs[i]['img']).replace('.png', '')
                    if all_imgs[i]['is_target']:
                        label_row += f"**图片{img_num}** ✅ | "
                    else:
                        label_row += f"图片{img_num} | "
            md_lines.append(label_row)
            md_lines.append("")
            
            # 显示目标图片
            md_lines.append("### 🎯 正确答案")
            md_lines.append("")
            target_num = os.path.basename(target_img).replace('.png', '')
            md_lines.append(f"**图片{target_num}**: {caption}")
            md_lines.append("")
            md_lines.append(f"![正确答案]({target_img})")
            md_lines.append("")
            
            md_lines.append("---")
            md_lines.append("")
    
    # 写入文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md_lines))
    
    print(f"✅ 已生成: {output_file}")
    return output_file

def main():
    """主函数"""
    print("="*70)
    print("🎨 Distractor数据集可视化生成器")
    print("="*70)
    print()
    
    # 检查文件是否存在
    if not os.path.exists("distractor_lists.json"):
        print("❌ 找不到 distractor_lists.json")
        return
    
    if not os.path.exists("distractor_dataset.json"):
        print("❌ 找不到 distractor_dataset.json")
        return
    
    # 生成两个可视化文档
    print("📝 生成可视化文档...\n")
    
    # 1. 干扰图列表的完整可视化（84张图片，较大）
    file1 = generate_distractor_lists_markdown(
        json_file="distractor_lists.json",
        output_file="DISTRACTOR_LISTS_VISUAL.md"
    )
    
    print()
    
    # 2. 测试数据集样本的可视化（每个难度前10个样本）
    file2 = generate_dataset_samples_markdown(
        json_file="distractor_dataset.json",
        output_file="DISTRACTOR_DATASET_SAMPLES.md",
        max_samples_per_difficulty=10
    )
    
    print()
    print("="*70)
    print("✅ 可视化文档生成完成！")
    print("="*70)
    print()
    print("生成的文件:")
    print(f"  1. {file1}")
    print(f"     - 完整的84张图片干扰图列表")
    print(f"     - 每张图展示3个难度的干扰图")
    print(f"     - 适合全面检查干扰图选择")
    print()
    print(f"  2. {file2}")
    print(f"     - 测试数据集样本展示")
    print(f"     - 每个难度展示前10个样本")
    print(f"     - 适合检查测试任务格式")
    print()
    print("💡 使用方法:")
    print("  - 在VS Code中打开markdown文件")
    print("  - 按 Cmd+Shift+V (Mac) 或 Ctrl+Shift+V (Windows) 预览")
    print("  - 或使用任何支持图片的markdown阅读器")
    print()
    print("⚠️  注意:")
    print("  - DISTRACTOR_LISTS_VISUAL.md 文件较大（包含所有84张图）")
    print("  - 预览可能需要一些时间加载图片")
    print("  - 确保images/文件夹在同一目录下")

if __name__ == "__main__":
    main()
