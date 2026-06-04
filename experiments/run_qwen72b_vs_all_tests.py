# -*- coding: utf-8 -*-
"""
运行 qwen2.5 VL 72b instruct 对战六个模型和自己
总共7场比赛：
- qwen2.5-vl-72b vs qwen2.5-vl-7b
- qwen2.5-vl-72b vs qwen2.5-vl-32b
- qwen2.5-vl-72b vs gemma3-12b
- qwen2.5-vl-72b vs gemma3-27b
- qwen2.5-vl-72b vs gemini-2.5-flash
- qwen2.5-vl-72b vs gpt-4o
- qwen2.5-vl-72b vs qwen2.5-vl-72b (自己vs自己)
"""

from parallel_batch_model_test import ParallelBatchModelTester

class Qwen72BVsAllTester(ParallelBatchModelTester):
    """qwen2.5-vl-72b 对战所有模型的测试器"""
    
    def __init__(self, listener_strategy: str = 'direct'):
        """初始化测试器"""
        # 调用父类初始化，但我们会覆盖models列表
        super().__init__(listener_strategy)
        
        # 新模型
        self.new_model = "qwen2.5-vl-72b"
        
        # 原来的六个模型
        self.original_models = [
            "qwen2.5-vl-7b",
            "qwen2.5-vl-32b", 
            "gemma3-12b",
            "gemma3-27b",
            "gemini-2.5-flash",
            "gpt-4o"
        ]
        
        # 重新计算总比赛数：6场对战 + 1场自己vs自己 = 7场
        self.total_matches = len(self.original_models) + 1
    
    def run_all_matches_parallel(self, rounds_per_phase: int = 12, max_concurrent_matches: int = 3):
        """
        并行运行所有模型对战
        
        Args:
            rounds_per_phase: 每个阶段的轮次数量
            max_concurrent_matches: 最大并发比赛数量
        """
        print("🚀 qwen2.5-vl-72b 对战测试开始!")
        print(f"📊 新模型: {self.new_model}")
        print(f"📊 对战模型: {', '.join(self.original_models)}")
        print(f"🎮 每阶段轮次: {rounds_per_phase}")
        print(f"🏆 总对战数: {self.total_matches}场")
        print(f"⚡ 最大并发比赛: {max_concurrent_matches}")
        print(f"⏱️  预计总耗时: 约 {self.total_matches * rounds_per_phase * 2 * 2 / 60 / max_concurrent_matches:.1f} 小时")
        
        # 生成所有对战组合
        matchups = []
        
        # 1. 对战原来的六个模型 (6场)
        for model in self.original_models:
            matchups.append((self.new_model, model, "vs"))
        
        # 2. 自己vs自己 (1场)
        matchups.append((self.new_model, self.new_model, "self"))
        
        print(f"\n📋 对战安排:")
        for i, (model_a, model_b, match_type) in enumerate(matchups, 1):
            if match_type == "self":
                print(f"  {i:2d}. {model_a} vs {model_b} (自己vs自己)")
            else:
                print(f"  {i:2d}. {model_a} vs {model_b}")
        
        # 并行运行对战
        successful = 0
        failed = 0
        
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
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
                    
                    from datetime import datetime
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
    
    def _save_progress(self):
        """保存测试进度"""
        try:
            from datetime import datetime
            import json
            
            progress_data = {
                'new_model': self.new_model,
                'original_models': self.original_models,
                'current_match': len(self.results),
                'total_matches': self.total_matches,
                'results': self.results,
                'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            with open('qwen72b_vs_all_progress.json', 'w', encoding='utf-8') as f:
                json.dump(progress_data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            print(f"❌ 保存进度失败: {e}")
    
    def _generate_final_report(self, successful: int, failed: int):
        """生成最终测试报告"""
        try:
            from datetime import datetime
            
            with open('qwen72b_vs_all_final_report.md', 'w', encoding='utf-8') as f:
                f.write("# 🚀 qwen2.5-vl-72b 对战测试最终报告\n\n")
                f.write(f"**Test Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                f.write("## 📊 Test Summary\n\n")
                f.write(f"- **New Model**: {self.new_model}\n")
                f.write(f"- **Opponent Models**: {', '.join(self.original_models)}\n")
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
                f.write("## 📊 Model Scores Summary (All 7 Matches)\n\n")
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
                                score_a_info = score_info
                                score_b_info = score_info
                        
                        # 格式化得分信息
                        score_a_str = f"{score_a_info.get('total_score', 0)}/{score_a_info.get('max_possible', 0)}={score_a_info.get('percentage', 0):.1f}%"
                        score_b_str = f"{score_b_info.get('total_score', 0)}/{score_b_info.get('max_possible', 0)}={score_b_info.get('percentage', 0):.1f}%"
                        
                        f.write(f"| {match_num} | {model_a} | {score_a_str} | {model_b} | {score_b_str} | {winner} |\n")
                
                f.write("\n")
            
            print(f"\n✅ 最终报告已保存到: qwen72b_vs_all_final_report.md")
            
            # 生成得分总结CSV文件
            self._generate_scores_summary_csv()
            
        except Exception as e:
            print(f"❌ 生成最终报告失败: {e}")
    
    def _generate_scores_summary_csv(self):
        """生成得分总结CSV文件"""
        try:
            from datetime import datetime
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            csv_filename = f'qwen72b_vs_all_scores_summary_{timestamp}.csv'
            
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
            
        except Exception as e:
            print(f"❌ 生成得分总结CSV失败: {e}")

def main():
    """主函数 - 直接运行全部7场测试"""
    print("🚀 开始运行 qwen2.5-vl-72b 对战全部7场测试")
    print("="*60)
    
    # 测试配置
    rounds_per_phase = 12  # 每个阶段12轮，确保每个玩家当3次说书者
    max_concurrent_matches = 3  # 最大并发3场比赛
    
    print(f"📋 测试配置:")
    print(f"  每阶段轮次: {rounds_per_phase}")
    print(f"  最大并发比赛: {max_concurrent_matches}")
    print(f"  总对战数: 7场 (6场对战原模型 + 1场自己vs自己)")
    print(f"  预计总耗时: 约 {7 * rounds_per_phase * 2 * 2 / 60 / max_concurrent_matches:.1f} 小时")
    print(f"  ⚡ 并行特性: 多轮同时进行 + 多比赛并发")
    print(f"  📊 得分计算: 每场模型得分 = 当前得分 / 最大可能得分 × 100%")
    
    # 创建测试器并开始测试
    tester = Qwen72BVsAllTester()
    tester.run_all_matches_parallel(rounds_per_phase, max_concurrent_matches)
    
    print(f"\n🎉 全部7场测试完成!")
    print(f"📄 详细报告已保存到: qwen72b_vs_all_final_report.md")
    print(f"📊 报告包含:")
    print(f"  - 每场比赛的模型得分详情 (实际得分/最大可能得分 = 百分比)")
    print(f"  - 7场比赛得分汇总表 (显示完整得分信息)")
    print(f"📈 额外生成CSV文件:")
    print(f"  - qwen72b_vs_all_scores_summary_YYYYMMDD_HHMMSS.csv (每场比赛详细得分)")

if __name__ == "__main__":
    main()

