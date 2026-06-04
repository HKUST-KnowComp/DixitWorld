# -*- coding: utf-8 -*-
"""
生成真人评估用的MD文档
每个模型20个有效回合，总共120个案例
"""

import json
import random
from collections import defaultdict

def is_valid_clue(clue):
    """判断clue是否有效（排除API错误如"1"）"""
    if not clue:
        return False
    clue = clue.strip()
    # 排除单个数字或太短的无意义输出
    if clue in ['1', '2', '3', '4', '0']:
        return False
    if len(clue) < 3:
        return False
    # 排除不完整的JSON输出
    if clue.startswith('{') and '"reasoning"' in clue:
        return False
    if clue.startswith('[{') or clue.startswith('```'):
        return False
    return True

def collect_storyteller_rounds(json_file):
    """收集每个模型作为说书人的所有有效回合"""
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    storyteller_rounds = defaultdict(list)
    
    # 遍历所有比赛
    for match in data['results']:
        if not match.get('success'):
            continue
        
        match_number = match.get('match_number', 0)
        
        # 处理两个阶段
        for phase_name in ['phase1', 'phase2']:
            if phase_name not in match['results']:
                continue
            
            phase = match['results'][phase_name]
            
            # 遍历每一轮
            for round_data in phase['rounds']:
                storyteller_model = round_data['storyteller_model']
                clue = round_data.get('description', '')
                
                # 只收集有效的clue
                if not is_valid_clue(clue):
                    continue
                
                round_info = {
                    'match_number': match_number,
                    'phase': phase_name,
                    'round_number': round_data.get('round_number', 0),
                    'storyteller': round_data['storyteller'],
                    'storyteller_model': storyteller_model,
                    'clue': clue,
                    'target_image': round_data['target_image_name'],
                    'candidate_images': round_data['candidate_names'],
                    'target_position': round_data.get('target_position', -1),
                    'guess_names': round_data.get('guess_names', []),
                    'storyteller_score': round_data.get('storyteller_score', 0),
                    'zero_reason': round_data.get('storyteller_zero_reason', '')
                }
                
                storyteller_rounds[storyteller_model].append(round_info)
    
    return storyteller_rounds

def generate_human_evaluation_md(storyteller_rounds, output_file, target_total=120):
    """生成真人评估MD文档"""
    lines = []
    
    # 标题
    lines.append("# VLM-Dixit 真人评估数据集")
    lines.append("")
    lines.append(f"本文档包含{target_total}个Dixit游戏回合，用于真人评估。")
    lines.append("")
    lines.append("## 评估说明")
    lines.append("")
    lines.append("对于每个案例，请根据以下三个维度评估：")
    lines.append("")
    lines.append("1. **Clarity（清晰度）**: 1-5分")
    lines.append("   - 1分：完全无法理解")
    lines.append("   - 5分：非常清晰易懂")
    lines.append("")
    lines.append("2. **Creativity（创意性）**: 1-5分")
    lines.append("   - 1分：毫无创意，直白描述")
    lines.append("   - 5分：非常有想象力和诗意")
    lines.append("")
    lines.append("3. **Ambiguity（模糊度）**: 1-5分")
    lines.append("   - 1分：太明显（所有人都能猜对）")
    lines.append("   - 3分：恰到好处（部分人猜对）")
    lines.append("   - 5分：太模糊（没人能猜对）")
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # 统计信息和采样策略
    total_cases = 0
    models_order = sorted(storyteller_rounds.keys())
    num_models = len(models_order)
    
    # 检查哪些模型回合数不足
    min_rounds = min(len(rounds) for rounds in storyteller_rounds.values())
    
    # 动态调整每个模型的采样数
    samples_per_model = {}
    if min_rounds < 20:
        # 如果有模型不足20个，重新分配
        # 对不足的模型用全部，其余均分剩余quota
        models_with_few = [m for m in models_order if len(storyteller_rounds[m]) < 20]
        models_with_enough = [m for m in models_order if len(storyteller_rounds[m]) >= 20]
        
        used_by_few = sum(len(storyteller_rounds[m]) for m in models_with_few)
        remaining = target_total - used_by_few
        
        if models_with_enough:
            per_enough = remaining // len(models_with_enough)
            for m in models_with_few:
                samples_per_model[m] = len(storyteller_rounds[m])
            for m in models_with_enough:
                samples_per_model[m] = per_enough
        else:
            # 所有模型都不足，按比例分配
            for m in models_order:
                samples_per_model[m] = len(storyteller_rounds[m])
    else:
        # 所有模型都够，平均分配
        per_model = target_total // num_models
        for m in models_order:
            samples_per_model[m] = per_model
    
    lines.append("## 数据集概览")
    lines.append("")
    lines.append("| 模型 | 案例数量 | 可用回合总数 |")
    lines.append("|------|----------|--------------|")
    
    for model in models_order:
        available = len(storyteller_rounds[model])
        selected = samples_per_model[model]
        total_cases += selected
        lines.append(f"| {model} | {selected} | {available} |")
    
    lines.append(f"| **总计** | **{total_cases}** | - |")
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # 为每个模型生成案例
    case_number = 1
    
    for model in models_order:
        rounds = storyteller_rounds[model]
        num_to_sample = samples_per_model[model]
        
        # 如果回合数超过需要的数量，随机采样
        if len(rounds) > num_to_sample:
            selected_rounds = random.sample(rounds, num_to_sample)
        else:
            selected_rounds = rounds
        
        # 按照match和round排序，保持一致性
        selected_rounds.sort(key=lambda x: (x['match_number'], x['round_number']))
        
        lines.append(f"## {model}")
        lines.append("")
        lines.append(f"共 {len(selected_rounds)} 个案例")
        lines.append("")
        
        for round_info in selected_rounds:
            lines.append(f"### 案例 {case_number}")
            lines.append("")
            lines.append(f"**来源**: Match {round_info['match_number']}, {round_info['phase']}, Round {round_info['round_number']}")
            lines.append("")
            lines.append(f"**说书人线索（Clue）**: \"{round_info['clue']}\"")
            lines.append("")
            lines.append("**候选图片（Candidates）**:")
            lines.append("")
            lines.append('<table>')
            lines.append('<tr>')
            
            for i, img in enumerate(round_info['candidate_images']):
                lines.append(f'<td align="center">')
                lines.append(f'<img src="images/{img}" width="200"><br>')
                lines.append(f'<b>{i+1}. {img}</b>')
                lines.append(f'</td>')
            
            lines.append('</tr>')
            lines.append('</table>')
            lines.append("")
            lines.append(f"**✅ 正确图片**: `{round_info['target_image']}` (位置 {round_info['target_position'] + 1})")
            lines.append("")
            
            # 显示游戏结果（供参考）
            correct_count = sum(1 for g in round_info['guess_names'] if g == round_info['target_image'])
            total_guessers = len(round_info['guess_names'])
            
            if round_info['storyteller_score'] == 0:
                if round_info['zero_reason'] == 'all_correct':
                    result_emoji = "🔴"
                    result_text = "失败（全部猜对，太明显）"
                elif round_info['zero_reason'] == 'all_wrong':
                    result_emoji = "🔴"
                    result_text = "失败（全部猜错，太模糊）"
                else:
                    result_emoji = "🔴"
                    result_text = "失败"
            else:
                result_emoji = "✅"
                result_text = f"成功（{correct_count}/{total_guessers}人猜对）"
            
            lines.append(f"**游戏结果**: {result_emoji} {result_text}")
            lines.append("")
            lines.append("**评估区域**:")
            lines.append("- [ ] **您的选择**: _____ (1/2/3/4) | **是否正确**: _____ (✓/✗)")
            lines.append("- [ ] **Clarity（清晰度）**: _____ / 5")
            lines.append("- [ ] **Creativity（创意性）**: _____ / 5")
            lines.append("- [ ] **Ambiguity（模糊度）**: _____ / 5")
            lines.append("- [ ] **备注**: _____________________________")
            lines.append("")
            lines.append("---")
            lines.append("")
            
            case_number += 1
    
    # 附录：图片列表
    lines.append("## 附录：图片参考")
    lines.append("")
    lines.append("所有图片位于 `images/` 目录下，格式为PNG。")
    lines.append("")
    lines.append("评估时可以打开对应图片查看内容。")
    lines.append("")
    
    # 写入文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    print(f"✅ 真人评估文档已生成: {output_file}")
    print(f"📊 总案例数: {total_cases}")
    return total_cases

def generate_human_evaluation_md_en(storyteller_rounds, output_file, target_total=120):
    """生成真人评估MD文档（英文版）"""
    lines = []
    
    # 标题
    lines.append("# VLM-Dixit Human Evaluation Dataset")
    lines.append("")
    lines.append(f"This document contains {target_total} Dixit game rounds for human evaluation.")
    lines.append("")
    lines.append("## Evaluation Instructions")
    lines.append("")
    lines.append("For each case, please evaluate on three dimensions:")
    lines.append("")
    lines.append("1. **Clarity**: 1-5 points")
    lines.append("   - 1: Completely incomprehensible")
    lines.append("   - 5: Very clear and understandable")
    lines.append("")
    lines.append("2. **Creativity**: 1-5 points")
    lines.append("   - 1: No creativity, literal description")
    lines.append("   - 5: Highly imaginative and poetic")
    lines.append("")
    lines.append("3. **Ambiguity**: 1-5 points")
    lines.append("   - 1: Too obvious (everyone guesses correctly)")
    lines.append("   - 3: Just right (some guess correctly)")
    lines.append("   - 5: Too vague (nobody guesses correctly)")
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # 统计信息和采样策略（与中文版相同）
    total_cases = 0
    models_order = sorted(storyteller_rounds.keys())
    num_models = len(models_order)
    
    min_rounds = min(len(rounds) for rounds in storyteller_rounds.values())
    
    samples_per_model = {}
    if min_rounds < 20:
        models_with_few = [m for m in models_order if len(storyteller_rounds[m]) < 20]
        models_with_enough = [m for m in models_order if len(storyteller_rounds[m]) >= 20]
        
        used_by_few = sum(len(storyteller_rounds[m]) for m in models_with_few)
        remaining = target_total - used_by_few
        
        if models_with_enough:
            per_enough = remaining // len(models_with_enough)
            for m in models_with_few:
                samples_per_model[m] = len(storyteller_rounds[m])
            for m in models_with_enough:
                samples_per_model[m] = per_enough
        else:
            for m in models_order:
                samples_per_model[m] = len(storyteller_rounds[m])
    else:
        per_model = target_total // num_models
        for m in models_order:
            samples_per_model[m] = per_model
    
    lines.append("## Dataset Overview")
    lines.append("")
    lines.append("| Model | Cases | Total Available Rounds |")
    lines.append("|-------|-------|------------------------|")
    
    for model in models_order:
        available = len(storyteller_rounds[model])
        selected = samples_per_model[model]
        total_cases += selected
        lines.append(f"| {model} | {selected} | {available} |")
    
    lines.append(f"| **Total** | **{total_cases}** | - |")
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # 为每个模型生成案例
    case_number = 1
    
    for model in models_order:
        rounds = storyteller_rounds[model]
        num_to_sample = samples_per_model[model]
        
        # 如果回合数超过需要的数量，随机采样
        if len(rounds) > num_to_sample:
            selected_rounds = random.sample(rounds, num_to_sample)
        else:
            selected_rounds = rounds
        
        # 按照match和round排序，保持一致性
        selected_rounds.sort(key=lambda x: (x['match_number'], x['round_number']))
        
        lines.append(f"## {model}")
        lines.append("")
        lines.append(f"Total: {len(selected_rounds)} cases")
        lines.append("")
        
        for round_info in selected_rounds:
            lines.append(f"### Case {case_number}")
            lines.append("")
            lines.append(f"**Source**: Match {round_info['match_number']}, {round_info['phase']}, Round {round_info['round_number']}")
            lines.append("")
            lines.append(f"**Storyteller's Clue**: \"{round_info['clue']}\"")
            lines.append("")
            lines.append("**Candidate Images**:")
            lines.append("")
            lines.append('<table>')
            lines.append('<tr>')
            
            for i, img in enumerate(round_info['candidate_images']):
                lines.append(f'<td align="center">')
                lines.append(f'<img src="images/{img}" width="200"><br>')
                lines.append(f'<b>{i+1}. {img}</b>')
                lines.append(f'</td>')
            
            lines.append('</tr>')
            lines.append('</table>')
            lines.append("")
            lines.append(f"**✅ Target Image**: `{round_info['target_image']}` (Position {round_info['target_position'] + 1})")
            lines.append("")
            
            # 显示游戏结果（供参考）
            correct_count = sum(1 for g in round_info['guess_names'] if g == round_info['target_image'])
            total_guessers = len(round_info['guess_names'])
            
            if round_info['storyteller_score'] == 0:
                if round_info['zero_reason'] == 'all_correct':
                    result_emoji = "🔴"
                    result_text = "Failed (All correct - too obvious)"
                elif round_info['zero_reason'] == 'all_wrong':
                    result_emoji = "🔴"
                    result_text = "Failed (All wrong - too vague)"
                else:
                    result_emoji = "🔴"
                    result_text = "Failed"
            else:
                result_emoji = "✅"
                result_text = f"Success ({correct_count}/{total_guessers} guessed correctly)"
            
            lines.append(f"**Game Result**: {result_emoji} {result_text}")
            lines.append("")
            lines.append("**Evaluation Area**:")
            lines.append("- [ ] **Your Choice**: _____ (1/2/3/4) | **Correct?**: _____ (✓/✗)")
            lines.append("- [ ] **Clarity**: _____ / 5")
            lines.append("- [ ] **Creativity**: _____ / 5")
            lines.append("- [ ] **Ambiguity**: _____ / 5")
            lines.append("- [ ] **Notes**: _____________________________")
            lines.append("")
            lines.append("---")
            lines.append("")
            
            case_number += 1
    
    # 附录：图片列表
    lines.append("## Appendix: Image Reference")
    lines.append("")
    lines.append("All images are located in the `images/` directory in PNG format.")
    lines.append("")
    lines.append("Please open the corresponding images during evaluation.")
    lines.append("")
    
    # 写入文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    print(f"✅ English evaluation document generated: {output_file}")
    print(f"📊 Total cases: {total_cases}")
    return total_cases

def main():
    # 设置随机种子以保证可复现
    random.seed(42)
    
    print("🔍 收集说书人回合数据...")
    storyteller_rounds = collect_storyteller_rounds('parallel_batch_test_progress.json')
    
    print("\n📊 每个模型的有效回合数:")
    total_available = 0
    models_with_enough = []
    models_with_few = []
    
    for model, rounds in sorted(storyteller_rounds.items()):
        print(f"  {model}: {len(rounds)} 个有效回合")
        total_available += len(rounds)
        if len(rounds) >= 20:
            models_with_enough.append(model)
        else:
            models_with_few.append(model)
    
    # 如果有模型不足20个，从其他模型补充
    if models_with_few:
        print(f"\n⚠️  注意: {', '.join(models_with_few)} 的有效回合少于20个")
        shortage = sum(20 - len(storyteller_rounds[m]) for m in models_with_few)
        print(f"📌 需要从其他模型补充 {shortage} 个案例以达到120个")
        
        # 从有足够回合的模型中均匀补充
        if models_with_enough:
            extra_per_model = (shortage + len(models_with_enough) - 1) // len(models_with_enough)
            print(f"💡 将从 {', '.join(models_with_enough)} 每个模型额外采样约 {extra_per_model} 个")
    
    print("\n📝 生成中文评估文档...")
    total_cn = generate_human_evaluation_md(storyteller_rounds, 'HUMAN_EVALUATION_CN.md', target_total=120)
    
    print("\n📝 生成英文评估文档...")
    total_en = generate_human_evaluation_md_en(storyteller_rounds, 'HUMAN_EVALUATION_EN.md', target_total=120)
    
    print(f"\n🎉 完成！共生成 {total_cn} 个评估案例")
    print("📄 中文版: HUMAN_EVALUATION_CN.md")
    print("📄 英文版: HUMAN_EVALUATION_EN.md")

if __name__ == '__main__':
    main()
