# -*- coding: utf-8 -*-
"""
并行批量模型测试脚本
支持多轮同时进行的21次测试（15次两两对战+6次自己vs自己）
"""

import os
import time
import json
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from parallel_model_comparison import ParallelFairDixitExperiment

class ParallelBatchModelTester:
    def __init__(self, listener_strategy: str = 'direct'):
        """初始化并行批量测试器
        
        Args:
            listener_strategy: 听书者策略 ('direct' 或 'entailment')
        """
        self.models = [
            "qwen2.5-vl-7b",
            "qwen2.5-vl-32b", 
            "gemma3-12b",
            "gemma3-27b",
            "gemini-2.5-flash",
            "gpt-4o"
        ]
        
        self.results = []
        self.current_match = 0
        self.listener_strategy = listener_strategy
        # 15次两两对战 + 6次自己vs自己 = 21次
        self.total_matches = len(self.models) * (len(self.models) - 1) // 2 + len(self.models)
        
    def run_all_matches_parallel(self, rounds_per_phase: int = 6, max_concurrent_matches: int = 3):
        """
        并行运行所有模型对战
        
        Args:
            rounds_per_phase: 每个阶段的轮次数量
            max_concurrent_matches: 最大并发比赛数量
        """
        print("🚀 并行六模型批量测试开始!")
        print(f"📊 参与模型: {', '.join(self.models)}")
        print(f"🎮 每阶段轮次: {rounds_per_phase}")
        print(f"🏆 总对战数: {self.total_matches}")
        print(f"⚡ 最大并发比赛: {max_concurrent_matches}")
        print(f"⏱️  预计总耗时: 约 {self.total_matches * rounds_per_phase * 2 * 2 / 60 / max_concurrent_matches:.1f} 小时")
        
        # 生成所有对战组合
        matchups = []
        
        # 1. 两两对战 (15次)
        for i in range(len(self.models)):
            for j in range(i + 1, len(self.models)):
                matchups.append((self.models[i], self.models[j], "vs"))
        
        # 2. 自己vs自己 (6次)
        for model in self.models:
            matchups.append((model, model, "self"))
        
        print(f"\n📋 对战安排:")
        for i, (model_a, model_b, match_type) in enumerate(matchups, 1):
            if match_type == "self":
                print(f"  {i:2d}. {model_a} vs {model_b} (自己vs自己)")
            else:
                print(f"  {i:2d}. {model_a} vs {model_b}")
        
        # 并行运行对战
        successful = 0
        failed = 0
        
        with ThreadPoolExecutor(max_workers=max_concurrent_matches) as executor:
            # 提交所有对战任务
            future_to_match = {}
            for i, (model_a, model_b, match_type) in enumerate(matchups, 1):
                future = executor.submit(self._run_single_match_parallel, model_a, model_b, rounds_per_phase, match_type, i)
                future_to_match[future] = (model_a, model_b, match_type, i)
            
            # 处理完成的对战
            for future in as_completed(future_to_match):
                model_a, model_b, match_type, match_num = future_to_match[future]
                try:
                    match_result = future.result()
                    if match_result['success']:
                        successful += 1
                        print(f"✅ 对战成功: {model_a} vs {model_b}")
                    else:
                        failed += 1
                        print(f"❌ 对战失败: {model_a} vs {model_b}")
                    
                    self.results.append(match_result)
                    self._save_progress()
                    
                except Exception as e:
                    failed += 1
                    print(f"❌ 对战异常: {model_a} vs {model_b} - {e}")
                    
                    error_result = {
                        'match_number': match_num,
                        'model_a': model_a,
                        'model_b': model_b,
                        'match_type': match_type,
                        'start_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'duration_minutes': 0,
                        'rounds_per_phase': rounds_per_phase,
                        'success': False,
                        'results': None,
                        'winner': None,
                        'percentage_scores': None,
                        'error': str(e)
                    }
                    self.results.append(error_result)
        
        # 按比赛编号排序结果
        self.results.sort(key=lambda x: x['match_number'])
        
        # 测试完成
        self._generate_final_report(successful, failed)
    
    def _run_single_match_parallel(self, model_a: str, model_b: str, rounds: int, match_type: str, match_num: int) -> dict:
        """并行运行单场对战"""
        start_time = time.time()
        
        try:
            print(f"🎮 开始对战 {match_num}/{self.total_matches}: {model_a} vs {model_b}")
            
            # 创建并行实验实例
            experiment = ParallelFairDixitExperiment(model_a, model_b, self.listener_strategy)
            
            # 运行并行实验
            results = experiment.run_experiment(rounds)
            
            end_time = time.time()
            duration = round((end_time - start_time) / 60, 2)
            
            # 计算百分比得分
            percentage_scores = self._calculate_percentage_scores(results, match_type)
            
            # 收集说书者零得分统计
            storyteller_zero_stats = self._collect_storyteller_zero_stats(results)
            
            # 记录结果
            match_result = {
                'match_number': match_num,
                'model_a': model_a,
                'model_b': model_b,
                'match_type': match_type,
                'start_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'duration_minutes': duration,
                'rounds_per_phase': rounds,
                'success': True,
                'results': results,
                'winner': self._determine_winner(results),
                'percentage_scores': percentage_scores,
                'storyteller_zero_stats': storyteller_zero_stats,
                'error': None
            }
            
            print(f"✅ 对战完成 {match_num}/{self.total_matches}: {model_a} vs {model_b} (耗时: {duration}分钟)")
            return match_result
            
        except Exception as e:
            end_time = time.time()
            duration = round((end_time - start_time) / 60, 2)
            
            print(f"❌ 对战失败 {match_num}/{self.total_matches}: {model_a} vs {model_b} - {e}")
            
            error_result = {
                'match_number': match_num,
                'model_a': model_a,
                'model_b': model_b,
                'match_type': match_type,
                'start_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'duration_minutes': duration,
                'rounds_per_phase': rounds,
                'success': False,
                'results': None,
                'winner': None,
                'percentage_scores': None,
                'error': str(e)
            }
            return error_result
    
    def _collect_storyteller_zero_stats(self, results: dict) -> dict:
        """收集说书者零得分统计"""
        zero_stats = {
            'total_storyteller_rounds': 0,
            'zero_score_rounds': 0,
            'all_correct_rounds': 0,
            'all_wrong_rounds': 0,
            'partial_correct_rounds': 0,
            'by_model': {}
        }
        
        # 遍历所有轮次
        for phase in ['phase1', 'phase2']:
            if phase in results and 'rounds' in results[phase]:
                for round_data in results[phase]['rounds']:
                    storyteller_model = round_data.get('storyteller_model')
                    storyteller_score = round_data.get('storyteller_score', 0)
                    storyteller_zero_reason = round_data.get('storyteller_zero_reason')
                    correct_guesses = round_data.get('correct_guesses', 0)
                    total_guessers = round_data.get('total_guessers', 0)
                    
                    zero_stats['total_storyteller_rounds'] += 1
                    
                    # 按模型统计
                    if storyteller_model not in zero_stats['by_model']:
                        zero_stats['by_model'][storyteller_model] = {
                            'total_rounds': 0,
                            'zero_score_rounds': 0,
                            'all_correct_rounds': 0,
                            'all_wrong_rounds': 0,
                            'partial_correct_rounds': 0
                        }
                    
                    zero_stats['by_model'][storyteller_model]['total_rounds'] += 1
                    
                    if storyteller_score == 0:
                        zero_stats['zero_score_rounds'] += 1
                        zero_stats['by_model'][storyteller_model]['zero_score_rounds'] += 1
                        
                        if storyteller_zero_reason == 'all_correct':
                            zero_stats['all_correct_rounds'] += 1
                            zero_stats['by_model'][storyteller_model]['all_correct_rounds'] += 1
                        elif storyteller_zero_reason == 'all_wrong':
                            zero_stats['all_wrong_rounds'] += 1
                            zero_stats['by_model'][storyteller_model]['all_wrong_rounds'] += 1
                    else:
                        zero_stats['partial_correct_rounds'] += 1
                        zero_stats['by_model'][storyteller_model]['partial_correct_rounds'] += 1
        
        return zero_stats
    
    def _calculate_percentage_scores(self, results: dict, match_type: str) -> dict:
        """计算百分比得分"""
        try:
            total_scores = results['total_scores']
            
            # 计算最高可能得分
            rounds_per_phase = results.get('rounds_per_phase', 12)
            # 每场比赛24轮（2阶段），每个模型操控2个玩家
            # 每个玩家：6次说书者(18分) + 18次听者(90分) = 108分  
            # 每个模型：2个玩家 × 108分 = 216分
            max_possible_score_per_player = 216  # 每场比赛每个模型最大得分
            
            percentage_scores = {}
            
            if match_type == "self":
                # 自己vs自己：分别计算两个"自己"的得分
                model_name = results.get('model_a', 'Unknown')
                
                # 计算两个"自己"的得分
                # P1和P2是第一个"自己"，P3和P4是第二个"自己"
                self_a_total = total_scores.get('P1', 0) + total_scores.get('P2', 0)
                self_b_total = total_scores.get('P3', 0) + total_scores.get('P4', 0)
                
                # 计算说书者和听者得分
                storyteller_scores = results.get('storyteller_scores', {})
                listener_scores = results.get('listener_scores', {})
                
                # 对于自己vs自己，需要正确计算说书者和听者得分
                # P1和P2是第一个"自己"，P3和P4是第二个"自己"
                self_a_storyteller = 0
                self_a_listener = 0
                self_b_storyteller = 0
                self_b_listener = 0
                
                # 从轮次结果中统计说书者和听者得分
                for phase in ['phase1', 'phase2']:
                    if phase in results and 'rounds' in results[phase]:
                        for round_data in results[phase]['rounds']:
                            storyteller_model = round_data.get('storyteller_model')
                            storyteller_score = round_data.get('storyteller_score', 0)
                            listener_scores = round_data.get('listener_scores', {})
                            
                            # 根据说书者模型分配说书者得分
                            if storyteller_model == model_name:
                                # 需要判断是P1/P2还是P3/P4当说书者
                                storyteller_player = round_data.get('storyteller', '')
                                if storyteller_player in ['P1', 'P2']:
                                    self_a_storyteller += storyteller_score
                                else:
                                    self_b_storyteller += storyteller_score
                            
                            # 统计听者得分
                            for player, score in listener_scores.items():
                                if player in ['P1', 'P2']:
                                    self_a_listener += score
                                else:
                                    self_b_listener += score
                
                percentage_scores[f"{model_name}_A"] = {
                    'total_score': self_a_total,
                    'max_possible': 2 * max_possible_score_per_player,
                    'percentage': round((self_a_total / (2 * max_possible_score_per_player)) * 100, 2),
                    'storyteller_score': self_a_storyteller,
                    'listener_score': self_a_listener
                }
                
                percentage_scores[f"{model_name}_B"] = {
                    'total_score': self_b_total,
                    'max_possible': 2 * max_possible_score_per_player,
                    'percentage': round((self_b_total / (2 * max_possible_score_per_player)) * 100, 2),
                    'storyteller_score': self_b_storyteller,
                    'listener_score': self_b_listener
                }
            else:
                # 两两对战：分别计算两个模型的得分
                model_a_total = sum([total_scores[p] for p in ['P1', 'P2']])
                model_b_total = sum([total_scores[p] for p in ['P3', 'P4']])
                
                model_a_name = results.get('model_a', 'Model A')
                model_b_name = results.get('model_b', 'Model B')
                
                # 计算说书者和听者得分
                storyteller_scores = results.get('storyteller_scores', {})
                listener_scores = results.get('listener_scores', {})
                
                model_a_storyteller = storyteller_scores.get(model_a_name, 0)
                model_a_listener = listener_scores.get(model_a_name, 0)
                model_b_storyteller = storyteller_scores.get(model_b_name, 0)
                model_b_listener = listener_scores.get(model_b_name, 0)
                
                percentage_scores[model_a_name] = {
                    'total_score': model_a_total,
                    'max_possible': 2 * max_possible_score_per_player,
                    'percentage': round((model_a_total / (2 * max_possible_score_per_player)) * 100, 2),
                    'storyteller_score': model_a_storyteller,
                    'listener_score': model_a_listener
                }
                
                percentage_scores[model_b_name] = {
                    'total_score': model_b_total,
                    'max_possible': 2 * max_possible_score_per_player,
                    'percentage': round((model_b_total / (2 * max_possible_score_per_player)) * 100, 2),
                    'storyteller_score': model_b_storyteller,
                    'listener_score': model_b_listener
                }
            
            return percentage_scores
            
        except Exception as e:
            print(f"计算百分比得分失败: {e}")
            return {}
    
    def _determine_winner(self, results: dict) -> str:
        """确定获胜者"""
        try:
            total_scores = results['total_scores']
            model_a_total = sum([total_scores[p] for p in ['P1', 'P2']])
            model_b_total = sum([total_scores[p] for p in ['P3', 'P4']])
            
            if model_a_total > model_b_total:
                return "Model A"
            elif model_b_total > model_a_total:
                return "Model B"
            else:
                return "Tie"
        except:
            return "Unknown"
    
    def _save_progress(self):
        """保存测试进度"""
        try:
            progress_data = {
                'models': self.models,
                'current_match': len(self.results),
                'total_matches': self.total_matches,
                'results': self.results,
                'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            with open('parallel_batch_test_progress.json', 'w', encoding='utf-8') as f:
                json.dump(progress_data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            print(f"❌ 保存进度失败: {e}")
    
    def run_limited_matches(self, num_matches: int, rounds_per_phase: int = 6, max_concurrent_matches: int = 3):
        """
        运行有限数量的对战（用于快速测试）
        
        Args:
            num_matches: 要运行的对战数量
            rounds_per_phase: 每个阶段的轮次数量
            max_concurrent_matches: 最大并发比赛数量
        """
        print(f"🚀 运行前{num_matches}场对战测试 (策略: {self.listener_strategy})")
        
        # 生成前num_matches个对战组合
        matchups = []
        
        # 只生成前num_matches场对战
        count = 0
        for i in range(len(self.models)):
            for j in range(i + 1, len(self.models)):
                if count >= num_matches:
                    break
                matchups.append((self.models[i], self.models[j], "vs"))
                count += 1
            if count >= num_matches:
                break
        
        print(f"📋 将运行 {len(matchups)} 场对战")
        
        start_time = time.time()
        
        # 运行对战
        with ThreadPoolExecutor(max_workers=max_concurrent_matches) as executor:
            future_to_match = {
                executor.submit(self._run_single_match, model_a, model_b, match_type, i+1, rounds_per_phase): (model_a, model_b, match_type)
                for i, (model_a, model_b, match_type) in enumerate(matchups)
            }
            
            successful_results = []
            failed_results = []
            
            for future in as_completed(future_to_match):
                model_a, model_b, match_type = future_to_match[future]
                try:
                    result = future.result()
                    if result and result.get('success', False):
                        successful_results.append(result)
                        self.results.append(result)
                    else:
                        failed_results.append((model_a, model_b, match_type))
                        
                except Exception as e:
                    print(f"❌ 对战失败 {model_a} vs {model_b}: {e}")
                    failed_results.append((model_a, model_b, match_type))
        
        total_time = time.time() - start_time
        
        print(f"🎉 {len(matchups)}场对战完成!")
        print(f"📊 成功: {len(successful_results)}/{len(matchups)}")
        print(f"❌ 失败: {len(failed_results)}/{len(matchups)}")
        print(f"📈 成功率: {len(successful_results)/len(matchups)*100:.1f}%")
        print(f"⏱️ 总耗时: {total_time/60:.1f} 分钟")
        
        return successful_results, failed_results
    
    def _generate_final_report(self, successful: int, failed: int):
        """生成最终测试报告"""
        try:
            with open('parallel_batch_test_final_report.md', 'w', encoding='utf-8') as f:
                f.write("# 🚀 Parallel Six Model Batch Test Final Report\n\n")
                f.write(f"**Test Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                f.write("## 📊 Test Summary\n\n")
                f.write(f"- **Total Matches**: {self.total_matches}\n")
                f.write(f"- **Successful**: {successful}\n")
                f.write(f"- **Failed**: {failed}\n")
                f.write(f"- **Success Rate**: {successful/self.total_matches*100:.1f}%\n")
                f.write(f"- **Processing**: Parallel execution with concurrent matches\n\n")
                
                f.write("## 🎮 Match Results\n\n")
                for result in self.results:
                    if result['success']:
                        match_type = result.get('match_type', 'vs')
                        if match_type == "self":
                            f.write(f"### ✅ Match {result['match_number']}: {result['model_a']} vs {result['model_b']} (自己vs自己)\n\n")
                        else:
                            f.write(f"### ✅ Match {result['match_number']}: {result['model_a']} vs {result['model_b']}\n\n")
                        
                        f.write(f"- **Winner**: {result['winner']}\n")
                        f.write(f"- **Duration**: {result['duration_minutes']} minutes\n")
                        f.write(f"- **Time**: {result['start_time']}\n")
                        
                        # 添加百分比得分信息
                        if result.get('percentage_scores'):
                            f.write(f"- **Model Scores (Percentage)**:\n")
                            for model, score_info in result['percentage_scores'].items():
                                f.write(f"  - **{model}**: {score_info['percentage']}% (得分: {score_info['total_score']}/{score_info['max_possible']})\n")
                        f.write("\n")
                    else:
                        f.write(f"### ❌ Match {result['match_number']}: {result['model_a']} vs {result['model_b']}\n\n")
                        f.write(f"- **Error**: {result['error']}\n")
                        f.write(f"- **Time**: {result['start_time']}\n\n")
                
                # 模型得分汇总表
                f.write("## 📊 Model Scores Summary (All 21 Matches)\n\n")
                f.write("| Match | Model A | Score A (实际/最大=%) | Model B | Score B (实际/最大=%) | Winner |\n")
                f.write("|-------|---------|---------------------|---------|---------------------|--------|\n")
                
                for result in self.results:
                    if result['success'] and result.get('percentage_scores'):
                        match_num = result['match_number']
                        model_a = result['model_a']
                        model_b = result['model_b']
                        winner = result['winner']
                        
                        # 获取百分比得分
                        scores = result['percentage_scores']
                        
                        # 根据键名正确映射得分
                        score_a_info = {}
                        score_b_info = {}
                        
                        for score_key, score_info in scores.items():
                            if score_key == model_a:
                                score_a_info = score_info
                            elif score_key == model_b:
                                score_b_info = score_info
                            elif score_key == f"{model_a}_A":
                                score_a_info = score_info
                            elif score_key == f"{model_a}_B":
                                score_b_info = score_info
                            elif score_key == 'Unknown':
                                # 对于自己vs自己的情况，两个模型使用相同的得分
                                score_a_info = score_info
                                score_b_info = score_info
                        
                        # 格式化得分信息
                        score_a_str = f"{score_a_info.get('total_score', 0)}/{score_a_info.get('max_possible', 0)}={score_a_info.get('percentage', 0):.1f}%"
                        score_b_str = f"{score_b_info.get('total_score', 0)}/{score_b_info.get('max_possible', 0)}={score_b_info.get('percentage', 0):.1f}%"
                        
                        f.write(f"| {match_num} | {model_a} | {score_a_str} | {model_b} | {score_b_str} | {winner} |\n")
                
                f.write("\n")
                
                # 模型排名
                f.write("## 🏅 Model Rankings\n\n")
                model_stats = self._calculate_model_stats()
                
                f.write("| Rank | Model | Wins | Losses | Ties | Win Rate |\n")
                f.write("|------|-------|------|--------|------|----------|\n")
                
                for rank, (model, stats) in enumerate(model_stats, 1):
                    win_rate = f"{stats['win_rate']:.1f}%" if stats['total_games'] > 0 else "N/A"
                    f.write(f"| {rank} | {model} | {stats['wins']} | {stats['losses']} | {stats['ties']} | {win_rate} |\n")
                
                f.write("\n")
                
                # 模型平均得分统计
                f.write("## 📈 Model Average Scores\n\n")
                model_avg_scores = self._calculate_average_scores()
                
                f.write("| Model | Average Score (实际/最大=%) | Total Matches |\n")
                f.write("|-------|---------------------------|---------------|\n")
                
                for model, avg_score in model_avg_scores:
                    total_matches = len([r for r in self.results if r['success'] and (r['model_a'] == model or r['model_b'] == model)])
                    # 计算总实际得分和总最大可能得分
                    total_actual = 0
                    total_max = 216 * 7  # 21场比赛中每个模型参与7场，总最大得分 = 216 × 7 = 1512分 (36说书者+180听书者)
                    for result in self.results:
                        if result['success'] and result.get('percentage_scores'):
                            scores = result['percentage_scores']
                            for score_key, score_info in scores.items():
                                # 确定实际模型名称
                                if score_key == model or score_key.endswith(f'_{model}') or (score_key == 'Unknown' and result.get('model_a') == model):
                                    total_actual += score_info.get('total_score', 0)
                                    break
                    
                    if total_max > 0:
                        f.write(f"| {model} | {total_actual}/{total_max}={avg_score:.2f}% | {total_matches} |\n")
                    else:
                        f.write(f"| {model} | 0/0=0.00% | {total_matches} |\n")
                
                f.write("\n")
                
                # 说书者和听者分别排名
                f.write("## 🎭 Storyteller vs Listener Performance\n\n")
                role_scores = self._calculate_role_average_scores()
                
                f.write("### 🎤 Storyteller Rankings\n\n")
                f.write("| Rank | Model | Average Storyteller Score (实际/最大=%) |\n")
                f.write("|------|-------|----------------------------------------|\n")
                for rank, (model, score, total_raw, total_max) in enumerate(role_scores['storyteller'], 1):
                    f.write(f"| {rank} | {model} | {total_raw}/{total_max}={score:.2f}% |\n")
                
                f.write("\n### 👂 Listener Rankings\n\n")
                f.write("| Rank | Model | Average Listener Score (实际/最大=%) |\n")
                f.write("|------|-------|-------------------------------------|\n")
                for rank, (model, score, total_raw, total_max) in enumerate(role_scores['listener'], 1):
                    f.write(f"| {rank} | {model} | {total_raw}/{total_max}={score:.2f}% |\n")
                
                # 说书者零得分原因分析
                f.write("\n## 🎯 Storyteller Zero Score Analysis\n\n")
                zero_stats_summary = self._calculate_zero_score_summary()
                
                f.write("### 📊 Overall Statistics\n\n")
                f.write(f"- **Total Storyteller Rounds**: {zero_stats_summary['total_storyteller_rounds']}\n")
                f.write(f"- **Zero Score Rounds**: {zero_stats_summary['zero_score_rounds']} ({zero_stats_summary['zero_score_rounds']/zero_stats_summary['total_storyteller_rounds']*100:.1f}%)\n")
                f.write(f"- **All Correct (Everyone Guessed Right)**: {zero_stats_summary['all_correct_rounds']} ({zero_stats_summary['all_correct_rounds']/zero_stats_summary['total_storyteller_rounds']*100:.1f}%)\n")
                f.write(f"- **All Wrong (Everyone Guessed Wrong)**: {zero_stats_summary['all_wrong_rounds']} ({zero_stats_summary['all_wrong_rounds']/zero_stats_summary['total_storyteller_rounds']*100:.1f}%)\n")
                f.write(f"- **Partial Correct (Some Guessed Right)**: {zero_stats_summary['partial_correct_rounds']} ({zero_stats_summary['partial_correct_rounds']/zero_stats_summary['total_storyteller_rounds']*100:.1f}%)\n\n")
                
                f.write("### 🎤 By Model Analysis\n\n")
                f.write("| Model | Total Rounds | Zero Score | All Correct | All Wrong | Partial Correct |\n")
                f.write("|-------|-------------|------------|-------------|-----------|----------------|\n")
                
                for model in self.models:
                    if model in zero_stats_summary['by_model']:
                        stats = zero_stats_summary['by_model'][model]
                        total = stats['total_rounds']
                        zero = stats['zero_score_rounds']
                        all_correct = stats['all_correct_rounds']
                        all_wrong = stats['all_wrong_rounds']
                        partial = stats['partial_correct_rounds']
                        
                        f.write(f"| {model} | {total} | {zero} ({zero/total*100:.1f}%) | {all_correct} ({all_correct/total*100:.1f}%) | {all_wrong} ({all_wrong/total*100:.1f}%) | {partial} ({partial/total*100:.1f}%) |\n")
                
                f.write("\n")
            
            print(f"\n✅ 最终报告已保存到: parallel_batch_test_final_report.md")
            
            # 生成得分总结CSV文件
            self._generate_scores_summary_csv()
            
        except Exception as e:
            print(f"❌ 生成最终报告失败: {e}")
    
    def _generate_scores_summary_csv(self):
        """生成得分总结CSV文件"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            csv_filename = f'model_scores_summary_{timestamp}.csv'
            
            with open(csv_filename, 'w', encoding='utf-8') as f:
                # 写入CSV头部
                f.write("Match_Number,Match_Type,Model_A,Model_A_Total_Score,Model_A_Total_Percentage,Model_A_Storyteller_Score,Model_A_Listener_Score,Model_B,Model_B_Total_Score,Model_B_Total_Percentage,Model_B_Storyteller_Score,Model_B_Listener_Score,Winner,Duration_Minutes,Start_Time\n")
                
                # 写入每场比赛的得分数据
                for result in self.results:
                    if result['success'] and result.get('percentage_scores'):
                        match_num = result['match_number']
                        match_type = result.get('match_type', 'vs')
                        model_a = result['model_a']
                        model_b = result['model_b']
                        winner = result['winner']
                        duration = result['duration_minutes']
                        start_time = result['start_time']
                        
                        # 获取得分信息
                        scores = result['percentage_scores']
                        
                        # 根据模型名称找到对应的得分信息
                        score_a_info = {}
                        score_b_info = {}
                        
                        for score_key, score_info in scores.items():
                            if score_key == model_a:
                                score_a_info = score_info
                            elif score_key == model_b:
                                score_b_info = score_info
                            elif score_key == f"{model_a}_A":
                                score_a_info = score_info
                            elif score_key == f"{model_a}_B":
                                score_b_info = score_info
                            elif score_key == 'Unknown':
                                # 对于自己vs自己的情况，两个模型使用相同的得分
                                score_a_info = score_info
                                score_b_info = score_info
                        
                        score_a = score_a_info.get('total_score', 0)
                        score_a_pct = score_a_info.get('percentage', 0)
                        score_a_storyteller = score_a_info.get('storyteller_score', 0)
                        score_a_listener = score_a_info.get('listener_score', 0)
                        
                        score_b = score_b_info.get('total_score', 0)
                        score_b_pct = score_b_info.get('percentage', 0)
                        score_b_storyteller = score_b_info.get('storyteller_score', 0)
                        score_b_listener = score_b_info.get('listener_score', 0)
                        
                        # 写入CSV行
                        f.write(f"{match_num},{match_type},{model_a},{score_a},{score_a_pct},{score_a_storyteller},{score_a_listener},{model_b},{score_b},{score_b_pct},{score_b_storyteller},{score_b_listener},{winner},{duration},{start_time}\n")
            
            print(f"✅ 得分总结CSV文件已保存到: {csv_filename}")
            
            # 同时生成模型平均得分总结
            self._generate_model_averages_csv(timestamp)
            
        except Exception as e:
            print(f"❌ 生成得分总结CSV失败: {e}")
    
    def _generate_model_averages_csv(self, timestamp):
        """生成模型平均得分CSV文件"""
        try:
            csv_filename = f'model_averages_summary_{timestamp}.csv'
            
            with open(csv_filename, 'w', encoding='utf-8') as f:
                # 写入CSV头部
                f.write("Model,Average_Score_Percentage,Total_Matches,Wins,Losses,Ties,Win_Rate_Percentage\n")
                
                # 计算模型统计
                model_stats = self._calculate_model_stats()
                model_avg_scores = self._calculate_average_scores()
                
                # 创建平均得分字典
                avg_scores_dict = dict(model_avg_scores)
                
                # 写入每个模型的统计数据
                for model, stats in model_stats:
                    avg_score = avg_scores_dict.get(model, 0)
                    win_rate = stats['win_rate'] if stats['total_games'] > 0 else 0
                    
                    f.write(f"{model},{avg_score:.2f},{stats['total_games']},{stats['wins']},{stats['losses']},{stats['ties']},{win_rate:.1f}\n")
            
            print(f"✅ 模型平均得分CSV文件已保存到: {csv_filename}")
            
        except Exception as e:
            print(f"❌ 生成模型平均得分CSV失败: {e}")
    
    def _calculate_model_stats(self) -> list:
        """计算模型统计信息"""
        model_stats = {model: {'wins': 0, 'losses': 0, 'ties': 0, 'total_games': 0} for model in self.models}
        
        for result in self.results:
            if not result['success']:
                continue
                
            model_a, model_b = result['model_a'], result['model_b']
            winner = result['winner']
            
            # 统计游戏次数
            model_stats[model_a]['total_games'] += 1
            model_stats[model_b]['total_games'] += 1
            
            # 统计胜负
            if winner == "Model A":
                model_stats[model_a]['wins'] += 1
                model_stats[model_b]['losses'] += 1
            elif winner == "Model B":
                model_stats[model_b]['wins'] += 1
                model_stats[model_a]['losses'] += 1
            else:  # Tie
                model_stats[model_a]['ties'] += 1
                model_stats[model_b]['ties'] += 1
        
        # 计算胜率并排序
        for model in model_stats:
            stats = model_stats[model]
            if stats['total_games'] > 0:
                stats['win_rate'] = (stats['wins'] / stats['total_games']) * 100
            else:
                stats['win_rate'] = 0
        
        # 按胜率排序
        sorted_stats = sorted(model_stats.items(), key=lambda x: x[1]['win_rate'], reverse=True)
        return sorted_stats
    
    def _calculate_average_scores(self) -> list:
        """计算每个模型的平均得分百分比"""
        model_scores = {model: [] for model in self.models}
        
        for result in self.results:
            if not result['success'] or not result.get('percentage_scores'):
                continue
                
            scores = result['percentage_scores']
            
            # 获取实际的模型名称
            model_a_name = result.get('model_a', 'Model A')
            model_b_name = result.get('model_b', 'Model B')
            
            for score_key, score_info in scores.items():
                # 根据score_key确定实际模型名称
                if score_key == 'Model A':
                    actual_model = model_a_name
                elif score_key == 'Model B':
                    actual_model = model_b_name
                elif score_key == 'Unknown':
                    # 对于自己vs自己的情况，需要从result中获取模型名称
                    actual_model = result.get('model_a', 'Unknown')
                elif score_key.endswith('_A') or score_key.endswith('_B'):
                    # 对于自己vs自己的新格式，提取基础模型名称
                    actual_model = score_key.rsplit('_', 1)[0]
                else:
                    actual_model = score_key
                
                if actual_model in model_scores:
                    if 'percentage' in score_info:
                        model_scores[actual_model].append(score_info['percentage'])
        
        # 计算平均得分并排序
        avg_scores = []
        for model, scores in model_scores.items():
            if scores:
                avg_score = sum(scores) / len(scores)
                avg_scores.append((model, avg_score))
            else:
                avg_scores.append((model, 0.0))
        
        # 按平均得分排序
        avg_scores.sort(key=lambda x: x[1], reverse=True)
        return avg_scores
    
    def _calculate_role_average_scores(self) -> dict:
        """计算每个模型作为说书者和听者的平均得分百分比"""
        model_storyteller_scores = {model: [] for model in self.models}
        model_listener_scores = {model: [] for model in self.models}
        
        for result in self.results:
            if not result['success'] or not result.get('percentage_scores'):
                continue
                
            scores = result['percentage_scores']
            rounds_per_phase = result.get('rounds_per_phase', 4)
            
            # 计算最高可能得分
            # 每场比赛：每个模型操控2个玩家，运行2个阶段共24轮
            # 每个模型当12次说书者，12次听书者
            max_storyteller_score = 2 * 6 * 3  # 每场比赛：2个玩家 × 6次说书者 × 3分 = 36分
            max_listener_score = 2 * 18 * 5  # 每场比赛：2个玩家 × 18次听者 × 5分 = 180分
            
            # 获取实际的模型名称
            model_a_name = result.get('model_a', 'Model A')
            model_b_name = result.get('model_b', 'Model B')
            
            for score_key, score_info in scores.items():
                # 根据score_key确定实际模型名称
                if score_key == 'Model A':
                    actual_model = model_a_name
                elif score_key == 'Model B':
                    actual_model = model_b_name
                elif score_key == 'Unknown':
                    # 对于自己vs自己的情况，需要从result中获取模型名称
                    actual_model = result.get('model_a', 'Unknown')
                elif score_key.endswith('_A') or score_key.endswith('_B'):
                    # 对于自己vs自己的新格式，提取基础模型名称
                    actual_model = score_key.rsplit('_', 1)[0]
                else:
                    actual_model = score_key
                
                if actual_model in model_storyteller_scores:
                    if 'storyteller_score' in score_info:
                        # 转换为百分比
                        storyteller_raw_score = score_info['storyteller_score']
                        storyteller_percentage = (storyteller_raw_score / max_storyteller_score * 100) if max_storyteller_score > 0 else 0
                        model_storyteller_scores[actual_model].append((storyteller_raw_score, max_storyteller_score, storyteller_percentage))
                    
                    if 'listener_score' in score_info:
                        # 转换为百分比
                        listener_raw_score = score_info['listener_score']
                        listener_percentage = (listener_raw_score / max_listener_score * 100) if max_listener_score > 0 else 0
                        model_listener_scores[actual_model].append((listener_raw_score, max_listener_score, listener_percentage))
        
        # 计算平均得分百分比
        storyteller_avg = []
        listener_avg = []
        
        for model in self.models:
            if model_storyteller_scores[model]:
                avg_storyteller = sum([score[2] for score in model_storyteller_scores[model]]) / len(model_storyteller_scores[model])
                total_raw = sum([score[0] for score in model_storyteller_scores[model]])
                total_max = 36 * 7  # 21场比赛中每个模型参与7场，说书者最大得分 = 36 × 7 = 252分
                storyteller_avg.append((model, avg_storyteller, total_raw, total_max))
            else:
                storyteller_avg.append((model, 0.0, 0, 0))
                
            if model_listener_scores[model]:
                avg_listener = sum([score[2] for score in model_listener_scores[model]]) / len(model_listener_scores[model])
                total_raw = sum([score[0] for score in model_listener_scores[model]])
                total_max = 180 * 7  # 21场比赛中每个模型参与7场，听者最大得分 = 180 × 7 = 1260分
                listener_avg.append((model, avg_listener, total_raw, total_max))
            else:
                listener_avg.append((model, 0.0, 0, 0))
        
        # 排序
        storyteller_avg.sort(key=lambda x: x[1], reverse=True)
        listener_avg.sort(key=lambda x: x[1], reverse=True)
        
        return {
            'storyteller': storyteller_avg,
            'listener': listener_avg
        }
    
    def _calculate_zero_score_summary(self) -> dict:
        """计算零得分统计汇总"""
        summary = {
            'total_storyteller_rounds': 0,
            'zero_score_rounds': 0,
            'all_correct_rounds': 0,
            'all_wrong_rounds': 0,
            'partial_correct_rounds': 0,
            'by_model': {}
        }
        
        # 汇总所有比赛的数据
        for result in self.results:
            if result['success'] and result.get('storyteller_zero_stats'):
                stats = result['storyteller_zero_stats']
                
                summary['total_storyteller_rounds'] += stats['total_storyteller_rounds']
                summary['zero_score_rounds'] += stats['zero_score_rounds']
                summary['all_correct_rounds'] += stats['all_correct_rounds']
                summary['all_wrong_rounds'] += stats['all_wrong_rounds']
                summary['partial_correct_rounds'] += stats['partial_correct_rounds']
                
                # 按模型汇总
                for model, model_stats in stats['by_model'].items():
                    if model not in summary['by_model']:
                        summary['by_model'][model] = {
                            'total_rounds': 0,
                            'zero_score_rounds': 0,
                            'all_correct_rounds': 0,
                            'all_wrong_rounds': 0,
                            'partial_correct_rounds': 0
                        }
                    
                    summary['by_model'][model]['total_rounds'] += model_stats['total_rounds']
                    summary['by_model'][model]['zero_score_rounds'] += model_stats['zero_score_rounds']
                    summary['by_model'][model]['all_correct_rounds'] += model_stats['all_correct_rounds']
                    summary['by_model'][model]['all_wrong_rounds'] += model_stats['all_wrong_rounds']
                    summary['by_model'][model]['partial_correct_rounds'] += model_stats['partial_correct_rounds']
        
        return summary

def main():
    """主函数"""
    print("🚀 并行六模型批量测试")
    print("="*60)
    
    # 获取测试参数
    try:
        rounds = int(input("请输入每个阶段的轮次数量 (默认6，推荐6): ").strip() or "6")
    except ValueError:
        rounds = 6
    
    try:
        max_concurrent = int(input("请输入最大并发比赛数量 (默认3，推荐3): ").strip() or "3")
    except ValueError:
        max_concurrent = 3
    
    print(f"\n📋 测试配置:")
    print(f"  每阶段轮次: {rounds}")
    print(f"  最大并发比赛: {max_concurrent}")
    print(f"  总对战数: 21场 (15场两两对战 + 6场自己vs自己)")
    print(f"  预计总耗时: 约 {21 * rounds * 2 * 2 / 60 / max_concurrent:.1f} 小时")
    print(f"  ⚡ 并行特性: 多轮同时进行 + 多比赛并发")
    
    # 确认开始
    confirm = input("\n确认开始并行批量测试? (y/n): ").strip().lower()
    if confirm != 'y':
        print("❌ 测试已取消")
        return
    
    # 创建测试器并开始测试
    tester = ParallelBatchModelTester()
    tester.run_all_matches_parallel(rounds, max_concurrent)
    
    print(f"\n🎉 并行批量测试完成!")

if __name__ == "__main__":
    main()
