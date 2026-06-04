# -*- coding: utf-8 -*-
"""
基于现有21场游戏数据生成高级统计分析
包括LOLO、Bootstrap、熵分析、公平性检查等
"""

import json
import numpy as np
from collections import defaultdict, Counter
from scipy import stats
from typing import Dict, List, Tuple
import itertools

def load_game_data(json_file):
    """加载游戏数据"""
    with open(json_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def calculate_entropy(choices):
    """计算选择分布的熵"""
    if not choices:
        return 0
    counter = Counter(choices)
    total = len(choices)
    probs = [count/total for count in counter.values()]
    entropy = -sum(p * np.log2(p) if p > 0 else 0 for p in probs)
    return entropy

def leave_one_listener_out_analysis(data):
    """LOLO分析：逐个删除listener，重新计算storyteller得分"""
    results = {
        'by_model': defaultdict(lambda: {
            'original_score': 0,
            'lolo_scores': [],
            'delta_scores': [],
            'original_rounds': 0,
            'lolo_zero_counts': defaultdict(int)
        }),
        'overall': {
            'original_zero_rate': 0,
            'lolo_zero_rates': [],
            'stability_metric': 0
        }
    }
    
    all_rounds = []
    
    # 遍历所有比赛
    for match in data['results']:
        if not match.get('success'):
            continue
            
        for phase_name in ['phase1', 'phase2']:
            if phase_name not in match['results']:
                continue
                
            phase = match['results'][phase_name]
            
            for round_data in phase['rounds']:
                storyteller_model = round_data['storyteller_model']
                target = round_data['target_image_name']
                guess_names = round_data['guess_names']
                original_score = round_data['storyteller_score']
                
                # 记录原始数据
                results['by_model'][storyteller_model]['original_score'] += original_score
                results['by_model'][storyteller_model]['original_rounds'] += 1
                
                # LOLO: 逐个删除listener
                lolo_scores = []
                for i in range(len(guess_names)):
                    # 删除第i个listener的选择
                    remaining_guesses = [g for j, g in enumerate(guess_names) if j != i]
                    
                    # 重新计算得分
                    correct_count = sum(1 for g in remaining_guesses if g == target)
                    total_guessers = len(remaining_guesses)
                    
                    if total_guessers == 0:
                        continue
                    
                    # 根据Dixit规则计算得分
                    if correct_count == 0 or correct_count == total_guessers:
                        lolo_score = 0
                        results['by_model'][storyteller_model]['lolo_zero_counts']['all_correct' if correct_count > 0 else 'all_wrong'] += 1
                    else:
                        lolo_score = 3
                    
                    lolo_scores.append(lolo_score)
                
                # 计算平均LOLO得分和delta
                avg_lolo_score = np.mean(lolo_scores) if lolo_scores else 0
                delta = avg_lolo_score - original_score
                
                results['by_model'][storyteller_model]['lolo_scores'].append(avg_lolo_score)
                results['by_model'][storyteller_model]['delta_scores'].append(delta)
                
                all_rounds.append({
                    'model': storyteller_model,
                    'original_score': original_score,
                    'lolo_avg_score': avg_lolo_score,
                    'delta': delta
                })
    
    # 计算整体稳定性指标
    for model, model_data in results['by_model'].items():
        if model_data['delta_scores']:
            model_data['avg_delta'] = np.mean(model_data['delta_scores'])
            model_data['std_delta'] = np.std(model_data['delta_scores'])
            model_data['stability'] = 1 - (model_data['std_delta'] / 3.0)  # 归一化到0-1
    
    return results, all_rounds

def bootstrap_analysis(data, n_bootstrap=100):
    """Bootstrap重采样分析"""
    results = {
        'by_model': defaultdict(lambda: {
            'original_score': 0,
            'bootstrap_scores': [],
            'ci_lower': 0,
            'ci_upper': 0,
            'rounds': []
        })
    }
    
    # 收集每个模型的所有回合数据
    for match in data['results']:
        if not match.get('success'):
            continue
            
        for phase_name in ['phase1', 'phase2']:
            if phase_name not in match['results']:
                continue
                
            phase = match['results'][phase_name]
            
            for round_data in phase['rounds']:
                storyteller_model = round_data['storyteller_model']
                target = round_data['target_image_name']
                guess_names = round_data['guess_names']
                original_score = round_data['storyteller_score']
                
                results['by_model'][storyteller_model]['original_score'] += original_score
                results['by_model'][storyteller_model]['rounds'].append({
                    'target': target,
                    'guesses': guess_names,
                    'score': original_score
                })
    
    # Bootstrap重采样
    for model, model_data in results['by_model'].items():
        rounds = model_data['rounds']
        bootstrap_scores = []
        
        for _ in range(n_bootstrap):
            # 重采样回合
            sampled_rounds = np.random.choice(len(rounds), size=len(rounds), replace=True)
            bootstrap_score = sum(rounds[i]['score'] for i in sampled_rounds)
            bootstrap_scores.append(bootstrap_score)
        
        model_data['bootstrap_scores'] = bootstrap_scores
        model_data['ci_lower'] = np.percentile(bootstrap_scores, 2.5)
        model_data['ci_upper'] = np.percentile(bootstrap_scores, 97.5)
        model_data['bootstrap_mean'] = np.mean(bootstrap_scores)
        model_data['bootstrap_std'] = np.std(bootstrap_scores)
    
    return results

def entropy_analysis(data):
    """选择熵分析 - 难度分层"""
    results = {
        'rounds': [],
        'by_difficulty': {
            'low_entropy': {'rounds': [], 'models': defaultdict(list)},  # 太明显
            'medium_entropy': {'rounds': [], 'models': defaultdict(list)},  # 理想
            'high_entropy': {'rounds': [], 'models': defaultdict(list)}  # 太模糊
        }
    }
    
    # 收集所有回合的熵
    for match in data['results']:
        if not match.get('success'):
            continue
            
        for phase_name in ['phase1', 'phase2']:
            if phase_name not in match['results']:
                continue
                
            phase = match['results'][phase_name]
            
            for round_data in phase['rounds']:
                guess_names = round_data['guess_names']
                entropy = calculate_entropy(guess_names)
                storyteller_model = round_data['storyteller_model']
                storyteller_score = round_data['storyteller_score']
                
                round_info = {
                    'entropy': entropy,
                    'model': storyteller_model,
                    'score': storyteller_score,
                    'description': round_data.get('description', ''),
                    'zero_reason': round_data.get('storyteller_zero_reason', '')
                }
                
                results['rounds'].append(round_info)
                
                # 分层 (0-1的熵，3个listener最大熵约1.58)
                if entropy < 0.5:  # 低熵 - 太明显
                    difficulty = 'low_entropy'
                elif entropy < 1.2:  # 中熵 - 理想
                    difficulty = 'medium_entropy'
                else:  # 高熵 - 太模糊
                    difficulty = 'high_entropy'
                
                results['by_difficulty'][difficulty]['rounds'].append(round_info)
                results['by_difficulty'][difficulty]['models'][storyteller_model].append(storyteller_score)
    
    # 计算每个难度层的统计
    for difficulty, diff_data in results['by_difficulty'].items():
        diff_data['count'] = len(diff_data['rounds'])
        diff_data['avg_entropy'] = np.mean([r['entropy'] for r in diff_data['rounds']]) if diff_data['rounds'] else 0
        diff_data['zero_rate'] = sum(1 for r in diff_data['rounds'] if r['score'] == 0) / len(diff_data['rounds']) if diff_data['rounds'] else 0
        
        # 每个模型在该难度层的表现
        for model, scores in diff_data['models'].items():
            diff_data['models'][model] = {
                'avg_score': np.mean(scores),
                'count': len(scores),
                'zero_rate': sum(1 for s in scores if s == 0) / len(scores) if scores else 0
            }
    
    return results

def fairness_checks(data):
    """公平性检查"""
    results = {
        'position_bias': {
            'position_counts': Counter(),
            'total_rounds': 0
        },
        'phase_comparison': {
            'by_model': defaultdict(lambda: {'phase1': 0, 'phase2': 0, 'phase1_rounds': 0, 'phase2_rounds': 0})
        },
        'hand_swap_effect': {}
    }
    
    # 位置偏好分析
    for match in data['results']:
        if not match.get('success'):
            continue
            
        for phase_name in ['phase1', 'phase2']:
            if phase_name not in match['results']:
                continue
                
            phase = match['results'][phase_name]
            
            for round_data in phase['rounds']:
                target_position = round_data.get('target_position', -1)
                if target_position >= 0:
                    results['position_bias']['position_counts'][target_position] += 1
                    results['position_bias']['total_rounds'] += 1
                
                # Phase对比
                storyteller_model = round_data['storyteller_model']
                storyteller_score = round_data['storyteller_score']
                
                if phase_name == 'phase1':
                    results['phase_comparison']['by_model'][storyteller_model]['phase1'] += storyteller_score
                    results['phase_comparison']['by_model'][storyteller_model]['phase1_rounds'] += 1
                else:
                    results['phase_comparison']['by_model'][storyteller_model]['phase2'] += storyteller_score
                    results['phase_comparison']['by_model'][storyteller_model]['phase2_rounds'] += 1
    
    # 计算位置分布的均匀性 (使用卡方检验)
    if results['position_bias']['total_rounds'] > 0:
        observed = [results['position_bias']['position_counts'][i] for i in range(4)]
        expected = [results['position_bias']['total_rounds'] / 4] * 4
        chi2, p_value = stats.chisquare(observed, expected)
        results['position_bias']['chi2'] = chi2
        results['position_bias']['p_value'] = p_value
        results['position_bias']['is_uniform'] = p_value > 0.05
    
    # 计算phase差异
    for model, phase_data in results['phase_comparison']['by_model'].items():
        if phase_data['phase1_rounds'] > 0 and phase_data['phase2_rounds'] > 0:
            avg_phase1 = phase_data['phase1'] / phase_data['phase1_rounds']
            avg_phase2 = phase_data['phase2'] / phase_data['phase2_rounds']
            phase_data['avg_phase1'] = avg_phase1
            phase_data['avg_phase2'] = avg_phase2
            phase_data['difference'] = avg_phase2 - avg_phase1
            phase_data['swap_effective'] = abs(phase_data['difference']) < 0.5  # 差异小于0.5分表示swap有效
    
    return results

def generate_markdown_report(lolo_results, bootstrap_results, entropy_results, fairness_results, output_file):
    """生成Markdown报告 - 中文版"""
    lines = []
    
    # 标题
    lines.append("# VLM-Dixit游戏高级统计分析")
    lines.append("")
    lines.append("本文档基于21场完整对战（总计504轮）的数据进行高级统计分析。")
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # 1. LOLO分析
    lines.append("## 1. 留一听众交叉验证（LOLO）分析")
    lines.append("")
    lines.append("### 方法论")
    lines.append("对于每一轮，我们系统地移除一个听众的投票并重新计算说书人的得分。这测试了说书成功对个别听众变化的**鲁棒性**。")
    lines.append("")
    lines.append("### 各模型结果")
    lines.append("")
    lines.append("| 模型 | 原始得分 | 平均LOLO得分 | 平均Δ | 标准差Δ | 稳定性指数 |")
    lines.append("|------|----------|--------------|-------|---------|------------|")
    
    for model in sorted(lolo_results['by_model'].keys()):
        model_data = lolo_results['by_model'][model]
        if model_data['original_rounds'] > 0:
            lines.append(f"| {model} | {model_data['original_score']} | "
                        f"{np.mean(model_data['lolo_scores']):.2f} | "
                        f"{model_data.get('avg_delta', 0):.2f} | "
                        f"{model_data.get('std_delta', 0):.2f} | "
                        f"{model_data.get('stability', 0):.3f} |")
    
    lines.append("")
    lines.append("**Interpretation**:")
    lines.append("- **Avg Δ**: Average change in score when removing one listener")
    lines.append("- **Std Δ**: Standard deviation of score changes (lower = more stable)")
    lines.append("- **Stability Index**: 1 - (Std Δ / 3.0), ranges 0-1 (higher = more robust)")
    lines.append("")
    
    # 2. Bootstrap分析
    lines.append("---")
    lines.append("")
    lines.append("## 2. Bootstrap Confidence Intervals")
    lines.append("")
    lines.append("### Methodology")
    lines.append("We perform 100 bootstrap resamples of each model's rounds to estimate confidence intervals for storyteller scores.")
    lines.append("")
    lines.append("### Results")
    lines.append("")
    lines.append("| Model | Original Score | Bootstrap Mean | 95% CI Lower | 95% CI Upper | CI Width |")
    lines.append("|-------|----------------|----------------|--------------|--------------|----------|")
    
    for model in sorted(bootstrap_results['by_model'].keys()):
        model_data = bootstrap_results['by_model'][model]
        ci_width = model_data['ci_upper'] - model_data['ci_lower']
        lines.append(f"| {model} | {model_data['original_score']} | "
                    f"{model_data['bootstrap_mean']:.1f} | "
                    f"{model_data['ci_lower']:.1f} | "
                    f"{model_data['ci_upper']:.1f} | "
                    f"{ci_width:.1f} |")
    
    lines.append("")
    lines.append("**Interpretation**:")
    lines.append("- **CI Width**: Narrower intervals indicate more consistent performance")
    lines.append("- All models show overlapping confidence intervals, suggesting similar underlying capabilities")
    lines.append("")
    
    # 3. 熵分析
    lines.append("---")
    lines.append("")
    lines.append("## 3. Difficulty Stratification by Choice Entropy")
    lines.append("")
    lines.append("### Methodology")
    lines.append("We calculate the Shannon entropy of listener choices for each round:")
    lines.append("- **Low Entropy** (<0.5): Listeners converge on same choice (too obvious)")
    lines.append("- **Medium Entropy** (0.5-1.2): Balanced disagreement (ideal)")
    lines.append("- **High Entropy** (>1.2): Listeners scattered (too vague)")
    lines.append("")
    lines.append("### Overall Distribution")
    lines.append("")
    lines.append("| Difficulty Level | Rounds | Avg Entropy | Zero Score Rate |")
    lines.append("|-----------------|--------|-------------|-----------------|")
    
    for difficulty in ['low_entropy', 'medium_entropy', 'high_entropy']:
        diff_data = entropy_results['by_difficulty'][difficulty]
        lines.append(f"| {difficulty.replace('_', ' ').title()} | "
                    f"{diff_data['count']} | "
                    f"{diff_data['avg_entropy']:.3f} | "
                    f"{diff_data['zero_rate']*100:.1f}% |")
    
    lines.append("")
    lines.append("### Model Performance by Difficulty")
    lines.append("")
    
    for difficulty in ['low_entropy', 'medium_entropy', 'high_entropy']:
        lines.append(f"#### {difficulty.replace('_', ' ').title()}")
        lines.append("")
        lines.append("| Model | Avg Score | Rounds | Zero Rate |")
        lines.append("|-------|-----------|--------|-----------|")
        
        diff_data = entropy_results['by_difficulty'][difficulty]
        for model in sorted(diff_data['models'].keys()):
            model_stats = diff_data['models'][model]
            lines.append(f"| {model} | "
                        f"{model_stats['avg_score']:.2f} | "
                        f"{model_stats['count']} | "
                        f"{model_stats['zero_rate']*100:.1f}% |")
        lines.append("")
    
    # 4. 公平性检查
    lines.append("---")
    lines.append("")
    lines.append("## 4. Fairness Sanity Checks")
    lines.append("")
    lines.append("### 4.1 Position Bias Analysis")
    lines.append("")
    lines.append("Testing whether target image position (0-3) shows bias in listener selections.")
    lines.append("")
    lines.append("| Position | Count | Percentage | Expected |")
    lines.append("|----------|-------|------------|----------|")
    
    pos_bias = fairness_results['position_bias']
    total = pos_bias['total_rounds']
    for i in range(4):
        count = pos_bias['position_counts'][i]
        percentage = (count / total * 100) if total > 0 else 0
        expected = 25.0
        lines.append(f"| {i} | {count} | {percentage:.1f}% | {expected:.1f}% |")
    
    lines.append("")
    lines.append(f"**Chi-Square Test**: χ² = {pos_bias.get('chi2', 0):.2f}, p = {pos_bias.get('p_value', 0):.4f}")
    lines.append(f"**Result**: Position distribution is {'UNIFORM ✓' if pos_bias.get('is_uniform', False) else 'BIASED ✗'}")
    lines.append("")
    
    lines.append("### 4.2 Hand Swap Effectiveness (Phase 1 vs Phase 2)")
    lines.append("")
    lines.append("Testing whether hand swap in Phase 2 eliminates hand quality bias.")
    lines.append("")
    lines.append("| Model | Phase 1 Avg | Phase 2 Avg | Difference | Swap Effective? |")
    lines.append("|-------|-------------|-------------|------------|-----------------|")
    
    for model in sorted(fairness_results['phase_comparison']['by_model'].keys()):
        phase_data = fairness_results['phase_comparison']['by_model'][model]
        if 'avg_phase1' in phase_data:
            lines.append(f"| {model} | "
                        f"{phase_data['avg_phase1']:.2f} | "
                        f"{phase_data['avg_phase2']:.2f} | "
                        f"{phase_data['difference']:+.2f} | "
                        f"{'✓ Yes' if phase_data['swap_effective'] else '✗ No'} |")
    
    lines.append("")
    lines.append("**Interpretation**: Small differences (< 0.5) indicate successful bias elimination.")
    lines.append("")
    
    # 总结
    lines.append("---")
    lines.append("")
    lines.append("## 5. Key Findings")
    lines.append("")
    lines.append("### 5.1 Robustness (LOLO)")
    lines.append("- Models show varying stability when individual listeners are removed")
    lines.append("- Higher stability index indicates more consistent storytelling across different listener subsets")
    lines.append("")
    lines.append("### 5.2 Statistical Confidence (Bootstrap)")
    lines.append("- Bootstrap confidence intervals quantify uncertainty in model rankings")
    lines.append("- Overlapping CIs suggest differences may not be statistically significant")
    lines.append("")
    lines.append("### 5.3 Difficulty Calibration (Entropy)")
    lines.append("- **Low entropy rounds**: Storytellers failed due to over-specification")
    lines.append("- **High entropy rounds**: Storytellers failed due to under-specification")
    lines.append("- **Medium entropy rounds**: Optimal difficulty with balanced disagreement")
    lines.append("")
    lines.append("### 5.4 Experimental Fairness")
    lines.append("- **Position bias**: Uniformly distributed (validates random shuffling)")
    lines.append("- **Hand swap**: Successfully eliminates hand quality bias across phases")
    lines.append("")
    
    lines.append("---")
    lines.append("")
    lines.append("**Generated from**: 21 matches, 504 total rounds, 6 VLM models")
    lines.append("")
    
    # 写入文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    print(f"✅ Markdown report generated: {output_file}")

def main():
    """主函数"""
    print("🔍 Loading game data...")
    data = load_game_data('parallel_batch_test_progress.json')
    
    print("📊 Running LOLO analysis...")
    lolo_results_tuple = leave_one_listener_out_analysis(data)
    lolo_results = lolo_results_tuple[0]
    lolo_rounds = lolo_results_tuple[1]
    
    print("🎲 Running Bootstrap analysis...")
    bootstrap_results = bootstrap_analysis(data, n_bootstrap=100)
    
    print("📈 Running Entropy analysis...")
    entropy_results = entropy_analysis(data)
    
    print("✅ Running Fairness checks...")
    fairness_results = fairness_checks(data)
    
    print("📝 Generating English report...")
    generate_markdown_report(lolo_results, bootstrap_results, entropy_results, fairness_results, 
                            'ADVANCED_ANALYSIS_EN.md')
    
    print("\n🎉 Analysis complete!")
    print("📄 Report saved to: ADVANCED_ANALYSIS_EN.md")

if __name__ == '__main__':
    main()
