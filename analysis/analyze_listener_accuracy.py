# -*- coding: utf-8 -*-
"""
分析每个模型作为听众的选择正确率
"""

import json
from collections import defaultdict

def analyze_listener_accuracy(json_file):
    """分析听众准确率"""
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 存储每个模型作为听众的统计
    listener_stats = defaultdict(lambda: {
        'total_guesses': 0,
        'correct_guesses': 0,
        'wrong_guesses': 0,
        'accuracy': 0.0,
        'guess_details': []
    })
    
    # 遍历所有比赛
    for match in data['results']:
        if not match.get('success'):
            continue
        
        # 获取模型映射（哪些玩家是哪个模型）
        # 根据你的代码，P1和P2是model_a，P3和P4是model_b
        model_a = match.get('model_a', '')
        model_b = match.get('model_b', '')
        
        # 处理两个阶段
        for phase_name in ['phase1', 'phase2']:
            if phase_name not in match['results']:
                continue
            
            phase = match['results'][phase_name]
            
            # 在phase2，手牌交换
            if phase_name == 'phase2':
                # P1和P2换成model_b，P3和P4换成model_a
                player_model_map = {
                    'P1': model_b,
                    'P2': model_b,
                    'P3': model_a,
                    'P4': model_a
                }
            else:
                player_model_map = {
                    'P1': model_a,
                    'P2': model_a,
                    'P3': model_b,
                    'P4': model_b
                }
            
            # 遍历每一轮
            for round_data in phase['rounds']:
                storyteller = round_data['storyteller']
                storyteller_model = round_data['storyteller_model']
                target_image = round_data['target_image_name']
                guess_names = round_data['guess_names']
                
                # 获取其他玩家（听众）
                all_players = ['P1', 'P2', 'P3', 'P4']
                listeners = [p for p in all_players if p != storyteller]
                
                # 分析每个听众的猜测
                for i, listener in enumerate(listeners):
                    if i >= len(guess_names):
                        continue
                    
                    listener_model = player_model_map[listener]
                    guess = guess_names[i]
                    is_correct = (guess == target_image)
                    
                    # 记录统计
                    listener_stats[listener_model]['total_guesses'] += 1
                    if is_correct:
                        listener_stats[listener_model]['correct_guesses'] += 1
                    else:
                        listener_stats[listener_model]['wrong_guesses'] += 1
                    
                    listener_stats[listener_model]['guess_details'].append({
                        'match': match.get('match_number', 0),
                        'phase': phase_name,
                        'round': round_data.get('round_number', 0),
                        'storyteller_model': storyteller_model,
                        'target': target_image,
                        'guess': guess,
                        'correct': is_correct
                    })
    
    # 计算准确率
    for model, stats in listener_stats.items():
        if stats['total_guesses'] > 0:
            stats['accuracy'] = (stats['correct_guesses'] / stats['total_guesses']) * 100
    
    return listener_stats

def generate_listener_accuracy_report(listener_stats, output_file):
    """生成听众准确率报告（英文）"""
    lines = []
    
    lines.append("# Listener Accuracy Analysis")
    lines.append("")
    lines.append("This document analyzes each model's performance as a **listener** (guesser) in the Dixit game.")
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # 总览表格
    lines.append("## Overall Listener Accuracy")
    lines.append("")
    lines.append("| Rank | Model | Total Guesses | Correct | Wrong | Accuracy |")
    lines.append("|------|-------|---------------|---------|-------|----------|")
    
    # 按准确率排序
    sorted_models = sorted(listener_stats.items(), key=lambda x: x[1]['accuracy'], reverse=True)
    
    for rank, (model, stats) in enumerate(sorted_models, 1):
        lines.append(f"| {rank} | {model} | {stats['total_guesses']} | "
                    f"{stats['correct_guesses']} | {stats['wrong_guesses']} | "
                    f"**{stats['accuracy']:.2f}%** |")
    
    lines.append("")
    
    # 详细分析
    lines.append("---")
    lines.append("")
    lines.append("## Key Insights")
    lines.append("")
    
    # 找出最高和最低准确率
    best_model = sorted_models[0]
    worst_model = sorted_models[-1]
    avg_accuracy = sum(s['accuracy'] for _, s in sorted_models) / len(sorted_models)
    
    lines.append(f"### Summary Statistics")
    lines.append("")
    lines.append(f"- **Best Listener**: {best_model[0]} ({best_model[1]['accuracy']:.2f}%)")
    lines.append(f"- **Worst Listener**: {worst_model[0]} ({worst_model[1]['accuracy']:.2f}%)")
    lines.append(f"- **Average Accuracy**: {avg_accuracy:.2f}%")
    lines.append(f"- **Accuracy Range**: {best_model[1]['accuracy'] - worst_model[1]['accuracy']:.2f}% spread")
    lines.append("")
    
    # 对比说书人和听众表现
    lines.append("### Comparison: Storyteller vs Listener Performance")
    lines.append("")
    lines.append("| Model | Listener Accuracy | Storyteller Score (from LOLO) | Role Balance |")
    lines.append("|-------|-------------------|-------------------------------|--------------|")
    
    # 这里需要手动添加说书人得分数据
    storyteller_scores = {
        'gemini-2.5-flash': 69,
        'gemma3-12b': 60,
        'gemma3-27b': 72,
        'gpt-4o': 51,
        'qwen2.5-vl-32b': 27,
        'qwen2.5-vl-7b': 21
    }
    
    for model, stats in sorted_models:
        st_score = storyteller_scores.get(model, 0)
        # 判断角色平衡：高听众+低说书=偏向理解；低听众+高说书=偏向创造
        if stats['accuracy'] > avg_accuracy and st_score > 50:
            balance = "🔄 Balanced (Good at both)"
        elif stats['accuracy'] > avg_accuracy and st_score <= 50:
            balance = "👂 Understanding-oriented"
        elif stats['accuracy'] <= avg_accuracy and st_score > 50:
            balance = "🎨 Creative-oriented"
        else:
            balance = "⚠️ Weak at both"
        
        lines.append(f"| {model} | {stats['accuracy']:.2f}% | {st_score} | {balance} |")
    
    lines.append("")
    
    # 与不同说书人对战时的准确率
    lines.append("---")
    lines.append("")
    lines.append("## Listener Accuracy by Storyteller Model")
    lines.append("")
    lines.append("How well does each listener model guess when facing different storytellers?")
    lines.append("")
    
    # 计算每个听众面对每个说书人的准确率
    listener_vs_storyteller = defaultdict(lambda: defaultdict(lambda: {'correct': 0, 'total': 0}))
    
    for listener_model, stats in listener_stats.items():
        for detail in stats['guess_details']:
            st_model = detail['storyteller_model']
            listener_vs_storyteller[listener_model][st_model]['total'] += 1
            if detail['correct']:
                listener_vs_storyteller[listener_model][st_model]['correct'] += 1
    
    # 为每个听众生成表格
    for listener_model, stats in sorted_models:
        lines.append(f"### {listener_model} as Listener")
        lines.append("")
        lines.append("| vs Storyteller | Correct | Total | Accuracy |")
        lines.append("|----------------|---------|-------|----------|")
        
        vs_stats = listener_vs_storyteller[listener_model]
        sorted_vs = sorted(vs_stats.items(), 
                          key=lambda x: (x[1]['correct']/max(x[1]['total'],1)), 
                          reverse=True)
        
        for st_model, vs_data in sorted_vs:
            if vs_data['total'] > 0:
                acc = (vs_data['correct'] / vs_data['total']) * 100
                lines.append(f"| {st_model} | {vs_data['correct']} | {vs_data['total']} | {acc:.1f}% |")
        
        lines.append("")
    
    lines.append("---")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("### What Does Listener Accuracy Tell Us?")
    lines.append("")
    lines.append("1. **Image Understanding**: High listener accuracy indicates strong visual-semantic understanding")
    lines.append("2. **Clue Interpretation**: Ability to map textual descriptions to visual content")
    lines.append("3. **Discrimination**: Can distinguish between similar images based on subtle clues")
    lines.append("")
    lines.append("### Listener vs Storyteller Trade-off")
    lines.append("")
    lines.append("- **Balanced models** (high in both): Good general VLM capability")
    lines.append("- **Understanding-oriented** (high listener, low storyteller): Better at comprehension than generation")
    lines.append("- **Creative-oriented** (low listener, high storyteller): Better at generation than comprehension")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("**Generated from**: 21 matches, 1,512 listener guesses total")
    lines.append("")
    
    # 写入文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    print(f"✅ English report generated: {output_file}")

def generate_listener_accuracy_report_cn(listener_stats, output_file):
    """生成听众准确率报告（中文）"""
    lines = []
    
    lines.append("# 听众选择准确率分析")
    lines.append("")
    lines.append("本文档分析每个模型在Dixit游戏中作为**听众**（猜测者）的表现。")
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # 总览表格
    lines.append("## 听众准确率总览")
    lines.append("")
    lines.append("| 排名 | 模型 | 总猜测次数 | 正确 | 错误 | 准确率 |")
    lines.append("|------|------|------------|------|------|--------|")
    
    # 按准确率排序
    sorted_models = sorted(listener_stats.items(), key=lambda x: x[1]['accuracy'], reverse=True)
    
    for rank, (model, stats) in enumerate(sorted_models, 1):
        lines.append(f"| {rank} | {model} | {stats['total_guesses']} | "
                    f"{stats['correct_guesses']} | {stats['wrong_guesses']} | "
                    f"**{stats['accuracy']:.2f}%** |")
    
    lines.append("")
    
    # 详细分析
    lines.append("---")
    lines.append("")
    lines.append("## 关键洞察")
    lines.append("")
    
    # 找出最高和最低准确率
    best_model = sorted_models[0]
    worst_model = sorted_models[-1]
    avg_accuracy = sum(s['accuracy'] for _, s in sorted_models) / len(sorted_models)
    
    lines.append(f"### 统计摘要")
    lines.append("")
    lines.append(f"- **最佳听众**: {best_model[0]} ({best_model[1]['accuracy']:.2f}%)")
    lines.append(f"- **最弱听众**: {worst_model[0]} ({worst_model[1]['accuracy']:.2f}%)")
    lines.append(f"- **平均准确率**: {avg_accuracy:.2f}%")
    lines.append(f"- **准确率范围**: {best_model[1]['accuracy'] - worst_model[1]['accuracy']:.2f}% 差距")
    lines.append("")
    
    # 对比说书人和听众表现
    lines.append("### 对比：说书人 vs 听众表现")
    lines.append("")
    lines.append("| 模型 | 听众准确率 | 说书人得分（来自LOLO） | 角色平衡 |")
    lines.append("|------|------------|------------------------|----------|")
    
    # 这里需要手动添加说书人得分数据
    storyteller_scores = {
        'gemini-2.5-flash': 69,
        'gemma3-12b': 60,
        'gemma3-27b': 72,
        'gpt-4o': 51,
        'qwen2.5-vl-32b': 27,
        'qwen2.5-vl-7b': 21
    }
    
    for model, stats in sorted_models:
        st_score = storyteller_scores.get(model, 0)
        # 判断角色平衡：高听众+低说书=偏向理解；低听众+高说书=偏向创造
        if stats['accuracy'] > avg_accuracy and st_score > 50:
            balance = "🔄 均衡型（两者都强）"
        elif stats['accuracy'] > avg_accuracy and st_score <= 50:
            balance = "👂 理解导向型"
        elif stats['accuracy'] <= avg_accuracy and st_score > 50:
            balance = "🎨 创造导向型"
        else:
            balance = "⚠️ 两者都弱"
        
        lines.append(f"| {model} | {stats['accuracy']:.2f}% | {st_score} | {balance} |")
    
    lines.append("")
    
    # 与不同说书人对战时的准确率
    lines.append("---")
    lines.append("")
    lines.append("## 面对不同说书人的听众准确率")
    lines.append("")
    lines.append("每个听众模型在面对不同说书人时的猜测准确率。")
    lines.append("")
    
    # 计算每个听众面对每个说书人的准确率
    listener_vs_storyteller = defaultdict(lambda: defaultdict(lambda: {'correct': 0, 'total': 0}))
    
    for listener_model, stats in listener_stats.items():
        for detail in stats['guess_details']:
            st_model = detail['storyteller_model']
            listener_vs_storyteller[listener_model][st_model]['total'] += 1
            if detail['correct']:
                listener_vs_storyteller[listener_model][st_model]['correct'] += 1
    
    # 为每个听众生成表格
    for listener_model, stats in sorted_models:
        lines.append(f"### {listener_model} 作为听众")
        lines.append("")
        lines.append("| 对战说书人 | 正确 | 总计 | 准确率 |")
        lines.append("|------------|------|------|--------|")
        
        vs_stats = listener_vs_storyteller[listener_model]
        sorted_vs = sorted(vs_stats.items(), 
                          key=lambda x: (x[1]['correct']/max(x[1]['total'],1)), 
                          reverse=True)
        
        for st_model, vs_data in sorted_vs:
            if vs_data['total'] > 0:
                acc = (vs_data['correct'] / vs_data['total']) * 100
                lines.append(f"| {st_model} | {vs_data['correct']} | {vs_data['total']} | {acc:.1f}% |")
        
        lines.append("")
    
    lines.append("---")
    lines.append("")
    lines.append("## 结果解读")
    lines.append("")
    lines.append("### 听众准确率说明什么？")
    lines.append("")
    lines.append("1. **图像理解能力**：高听众准确率表明强大的视觉-语义理解能力")
    lines.append("2. **线索解读能力**：能够将文本描述映射到视觉内容的能力")
    lines.append("3. **辨别能力**：基于微妙线索区分相似图像的能力")
    lines.append("")
    lines.append("### 听众 vs 说书人权衡")
    lines.append("")
    lines.append("- **均衡型模型**（两者都高）：良好的通用VLM能力")
    lines.append("- **理解导向型**（听众高、说书人低）：理解能力强于生成能力")
    lines.append("- **创造导向型**（听众低、说书人高）：生成能力强于理解能力")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("**生成自**：21场对战，共1,512次听众猜测")
    lines.append("")
    
    # 写入文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    print(f"✅ 中文报告已生成: {output_file}")

def main():
    print("🔍 分析听众准确率...")
    listener_stats = analyze_listener_accuracy('parallel_batch_test_progress.json')
    
    print("\n📊 听众准确率概览:")
    sorted_models = sorted(listener_stats.items(), key=lambda x: x[1]['accuracy'], reverse=True)
    for rank, (model, stats) in enumerate(sorted_models, 1):
        print(f"  {rank}. {model}: {stats['accuracy']:.2f}% "
              f"({stats['correct_guesses']}/{stats['total_guesses']})")
    
    print("\n📝 生成英文报告...")
    generate_listener_accuracy_report(listener_stats, 'LISTENER_ACCURACY_EN.md')
    
    print("📝 生成中文报告...")
    generate_listener_accuracy_report_cn(listener_stats, 'LISTENER_ACCURACY_CN.md')
    
    print("\n🎉 分析完成！")

if __name__ == '__main__':
    main()
