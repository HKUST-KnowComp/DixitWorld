# -*- coding: utf-8 -*-
"""
并行处理的公平Dixit模型能力对比测试脚本
通过手牌反向分配消除手牌差异对结果的影响
支持多轮并行处理以提高效率
"""

import os
import random
import logging
import time
from typing import List, Dict, Tuple, Any
from multiprocessing import Pool, Manager
from concurrent.futures import ThreadPoolExecutor, as_completed
from game import DixitGame
from agents import PlayerAgent
from config import VLM_CONFIG

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ParallelFairDixitExperiment:
    def __init__(self, model_a_name: str, model_b_name: str, listener_strategy: str = 'direct'):
        """
        初始化并行公平Dixit实验
        
        Args:
            model_a_name: 模型A名称（控制P1和P2）
            model_b_name: 模型B名称（控制P3和P4）
            listener_strategy: 听书者策略 ('direct' 或 'entailment')
        """
        self.model_a_name = model_a_name
        self.model_b_name = model_b_name
        self.listener_strategy = listener_strategy
        self.results = {
            'phase1': {'rounds': [], 'final_scores': {}},
            'phase2': {'rounds': [], 'final_scores': {}},
            'total_scores': {}
        }
        
    def create_agents(self) -> List[PlayerAgent]:
        """
        创建4个玩家代理
        
        Returns:
            玩家代理列表 [P1, P2, P3, P4]
        """
        agents = []
        
        # P1和P2由模型A控制
        p1 = PlayerAgent(0, "P1")
        p1.model_name = self.model_a_name
        agents.append(p1)
        
        p2 = PlayerAgent(1, "P2")
        p2.model_name = self.model_a_name
        agents.append(p2)
        
        # P3和P4由模型B控制
        p3 = PlayerAgent(2, "P3")
        p3.model_name = self.model_b_name
        agents.append(p3)
        
        p4 = PlayerAgent(3, "P4")
        p4.model_name = self.model_b_name
        agents.append(p4)
        
        return agents
    
    def run_parallel_rounds(self, agents: List[PlayerAgent], phase_name: str, rounds_per_phase: int = 12) -> List[Dict]:
        """
        并行运行多个轮次
        
        Args:
            agents: 玩家代理列表
            phase_name: 阶段名称
            rounds_per_phase: 轮次数量
            
        Returns:
            轮次结果列表
        """
        print(f"\n🚀 开始并行运行 {phase_name} - {rounds_per_phase} 轮")
        
        # 创建图像池
        image_pool = [f"images/{i}.png" for i in range(1, 85)]
        
        # 创建游戏实例并分发手牌
        game = DixitGame(agents, image_pool)
        game._deal_initial_hands()
        
        # 显示初始手牌分配
        print(f"📋 {phase_name} 初始手牌分配:")
        for agent in agents:
            hand_names = [os.path.basename(img) for img in agent.hand]
            print(f"  {agent.name} ({agent.model_name}): {hand_names}")
        
        # 并行运行所有轮次
        round_results = []
        
        # 使用线程池并行处理轮次
        with ThreadPoolExecutor(max_workers=min(rounds_per_phase, 4)) as executor:
            # 提交所有轮次任务
            future_to_round = {
                executor.submit(self._run_single_round_parallel, agents, round_num, phase_name, image_pool): round_num
                for round_num in range(1, rounds_per_phase + 1)
            }
            
            # 收集结果
            for future in as_completed(future_to_round):
                round_num = future_to_round[future]
                try:
                    round_result = future.result()
                    round_results.append(round_result)
                    print(f"✅ {phase_name} 第 {round_num} 轮完成")
                except Exception as e:
                    print(f"❌ {phase_name} 第 {round_num} 轮失败: {e}")
                    # 创建失败的轮次结果
                    round_results.append({
                        'round_number': round_num,
                        'phase': phase_name,
                        'error': str(e),
                        'scores': {agent.name: 0 for agent in agents},
                        'total_scores': {agent.name: agent.score for agent in agents}
                    })
        
        # 按轮次编号排序
        round_results.sort(key=lambda x: x['round_number'])
        
        # 计算累计分数
        for round_result in round_results:
            if 'error' not in round_result:
                for agent in agents:
                    agent.score += round_result['scores'][agent.name]
                    round_result['total_scores'][agent.name] = agent.score
        
        return round_results
    
    def _run_single_round_parallel(self, agents: List[PlayerAgent], round_num: int, phase_name: str, image_pool: List[str]) -> Dict:
        """
        并行运行单轮游戏（线程安全版本）
        
        Args:
            agents: 玩家代理列表
            round_num: 轮次编号
            phase_name: 阶段名称
            image_pool: 图像池
            
        Returns:
            轮次结果
        """
        # 为每个线程创建独立的代理副本
        thread_agents = []
        for agent in agents:
            thread_agent = PlayerAgent(agent.agent_id, agent.name)
            thread_agent.model_name = agent.model_name
            thread_agent.hand = agent.hand.copy()
            thread_agent.score = agent.score
            thread_agents.append(thread_agent)
        
        print(f"\n🎮 {phase_name} - 第 {round_num} 轮")
        
        # 确定说书人（每轮轮换）
        storyteller_index = (round_num - 1) % len(thread_agents)
        storyteller = thread_agents[storyteller_index]
        print(f"👤 说书人: {storyteller.name} ({storyteller.model_name})")
        
        # 显示所有玩家的手牌
        print(f"📋 玩家手牌:")
        for agent in thread_agents:
            hand_names = [os.path.basename(img) for img in agent.hand]
            print(f"  {agent.name} ({agent.model_name}): {hand_names}")
        
        # 说书人选择目标图像（使用VLM智能选择）
        target_image = storyteller.select_target_image()
        print(f"🎯 说书人选择目标图像: {os.path.basename(target_image)}")
        
        # 生成描述
        description = storyteller.generate_description(target_image)
        print(f"💬 描述: \"{description}\"")
        
        # 其他玩家选择干扰图像（并行处理）
        guessers = [agent for i, agent in enumerate(thread_agents) if i != storyteller_index]
        
        # 并行选择干扰图像
        with Pool(processes=len(guessers)) as pool:
            distractor_results = pool.map(
                self._select_distractor_worker, 
                [(agent, description) for agent in guessers]
            )
        
        distractor_images = [result['distractor'] for result in distractor_results]
        distractor_image_names = [result['name'] for result in distractor_results]
        
        # 打印结果
        for i, result in enumerate(distractor_results):
            print(f"🎭 {result['agent_name']} ({result['model_name']}) 选择干扰图像: {result['name']}")
        
        # 创建候选图像列表并洗牌
        candidate_images = [target_image] + distractor_images
        random.shuffle(candidate_images)
        shuffled_candidate_names = [os.path.basename(img) for img in candidate_images]
        
        # 找到目标图像位置
        target_position = shuffled_candidate_names.index(os.path.basename(target_image))
        
        print(f"📋 候选图像 (洗牌后):")
        for i, candidate in enumerate(shuffled_candidate_names):
            if i == target_position:
                print(f"  {i+1}. {candidate} 🎯 (目标图像)")
            else:
                print(f"  {i+1}. {candidate}")
        
        # 其他玩家进行最终猜测（并行处理）
        with Pool(processes=len(guessers)) as pool:
            guess_results = pool.map(
                self._guess_target_worker,
                [(agent, description, candidate_images, target_image) for agent in guessers]
            )
        
        guess_names = [result['guess_name'] for result in guess_results]
        
        # 打印结果
        for result in guess_results:
            if result['correct']:
                print(f"✅ {result['agent_name']} ({result['model_name']}) 猜对了!")
            else:
                print(f"❌ {result['agent_name']} ({result['model_name']}) 猜错了，选择了 {result['guess_name']}")
        
        # 计算得分
        score_result = self._calculate_scores(storyteller, guessers, target_position, guess_names)
        scores = score_result['scores']
        
        # 更新累计分数
        for agent in thread_agents:
            agent.score += scores[agent.name]
        
        # 返回轮次结果
        round_result = {
            'round_number': round_num,
            'phase': phase_name,
            'storyteller': storyteller.name,
            'storyteller_model': storyteller.model_name,
            'target_image_name': os.path.basename(target_image),
            'description': description,
            'distractor_image_names': distractor_image_names,
            'candidate_names': shuffled_candidate_names,
            'target_position': target_position,
            'guess_names': guess_names,
            'scores': scores,
            'total_scores': {agent.name: agent.score for agent in thread_agents},
            'storyteller_score': scores[storyteller.name],  # 说书人本轮得分
            'listener_scores': {guesser.name: scores[guesser.name] for guesser in guessers},  # 听众本轮得分
            'storyteller_zero_reason': score_result['storyteller_zero_reason'],  # 说书人零得分原因
            'correct_guesses': score_result['correct_guesses'],  # 猜对人数
            'total_guessers': score_result['total_guessers']  # 总猜测人数
        }
        
        return round_result
    
    def _calculate_scores(self, storyteller: PlayerAgent, guessers: List[PlayerAgent], 
                         target_position: int, guess_names: List[str]) -> Dict[str, Any]:
        """
        计算得分
        
        Args:
            storyteller: 说书人
            guessers: 猜测者列表
            target_position: 目标图像位置
            guess_names: 猜测结果列表
            
        Returns:
            包含得分和零得分原因的字典
        """
        scores = {agent.name: 0 for agent in [storyteller] + guessers}
        
        # 统计猜对的人数
        correct_guesses = 0
        for i, guess in enumerate(guess_names):
            if guess == os.path.basename(storyteller.selected_target):
                correct_guesses += 1
                scores[guessers[i].name] += 3  # 猜对得3分
        
        # 说书人得分规则和零得分原因
        storyteller_zero_reason = None
        if correct_guesses == 0:
            # 没人猜对，说书人得0分，其他人得2分
            storyteller_zero_reason = "all_wrong"
            for guesser in guessers:
                scores[guesser.name] += 2
        elif correct_guesses == len(guessers):
            # 所有人都猜对，说书人得0分
            storyteller_zero_reason = "all_correct"
        elif correct_guesses < len(guessers):
            # 部分人猜对，说书人得3分
            scores[storyteller.name] += 3
        
        # 干扰图像被选择的奖励分
        for guesser in guessers:
            if guesser.selected_distractor and os.path.basename(guesser.selected_distractor) in guess_names:
                scores[guesser.name] += 1
        
        return {
            'scores': scores,
            'storyteller_zero_reason': storyteller_zero_reason,
            'correct_guesses': correct_guesses,
            'total_guessers': len(guessers)
        }
    
    def _select_distractor_worker(self, args):
        """
        并行选择干扰图像的工作函数
        
        Args:
            args: (agent, description) 元组
            
        Returns:
            包含选择结果的字典
        """
        agent, description = args
        try:
            distractor = agent.select_distractor_image_with_description(description)
            # 确保设置了selected_distractor
            agent.selected_distractor = distractor
            return {
                'agent_name': agent.name,
                'model_name': agent.model_name,
                'distractor': distractor,
                'name': os.path.basename(distractor)
            }
        except Exception as e:
            logging.error(f"选择干扰图像失败 {agent.name}: {e}")
            # 如果失败，随机选择一张牌
            import random
            distractor = random.choice(agent.hand)
            # 确保设置了selected_distractor
            agent.selected_distractor = distractor
            return {
                'agent_name': agent.name,
                'model_name': agent.model_name,
                'distractor': distractor,
                'name': os.path.basename(distractor)
            }
    
    def _guess_target_worker(self, args):
        """
        并行猜测目标图像的工作函数
        
        Args:
            args: (agent, description, candidate_images, target_image) 元组
            
        Returns:
            包含猜测结果的字典
        """
        agent, description, candidate_images, target_image = args
        try:
            # 玩家不能选择自己的干扰图像
            available_candidates = [img for img in candidate_images if agent.selected_distractor and img != agent.selected_distractor]
            if not available_candidates:  # 如果过滤后没有候选图像，使用所有候选图像
                available_candidates = candidate_images
            print(f"DEBUG: {agent.name} 候选图像数量: {len(candidate_images)}, 可用候选图像数量: {len(available_candidates)}")
            
            # 根据听书者策略选择不同的猜测方法
            if self.listener_strategy == 'entailment':
                guess = agent.guess_target_image_with_entailment(description, available_candidates)
                # 计算蕴含得分用于记录
                entailment_scores = agent.compute_entailment_scores(description, available_candidates)
            else:  # 默认使用直接选择
                guess = agent.guess_target_image(description, available_candidates)
                entailment_scores = None
                
            guess_name = os.path.basename(guess)
            correct = guess_name == os.path.basename(target_image)
            
            result = {
                'agent_name': agent.name,
                'model_name': agent.model_name,
                'guess_name': guess_name,
                'correct': correct,
                'strategy': self.listener_strategy
            }
            
            # 如果使用蕴含策略，记录得分信息
            if entailment_scores is not None:
                result['entailment_scores'] = entailment_scores
                result['selected_score'] = entailment_scores[available_candidates.index(guess)]
            
            return result
        except Exception as e:
            logging.error(f"猜测目标图像失败 {agent.name}: {e}")
            # 如果失败，随机选择一张候选图像
            import random
            guess = random.choice(candidate_images)
            guess_name = os.path.basename(guess)
            correct = guess_name == os.path.basename(target_image)
            
            # 确保设置了selected_distractor
            if not agent.selected_distractor:
                agent.selected_distractor = random.choice(agent.hand)
            
            return {
                'agent_name': agent.name,
                'model_name': agent.model_name,
                'guess_name': guess_name,
                'correct': correct
            }
    
    def reverse_hands(self, agents: List[PlayerAgent]):
        """
        反向分配手牌：P1 ↔ P3，P2 ↔ P4
        
        Args:
            agents: 玩家代理列表
        """
        print(f"\n🔄 手牌反向分配")
        print(f"P1 ↔ P3, P2 ↔ P4")
        
        # 保存当前手牌
        p1_hand = agents[0].hand.copy()
        p2_hand = agents[1].hand.copy()
        p3_hand = agents[2].hand.copy()
        p4_hand = agents[3].hand.copy()
        
        # 交换手牌
        agents[0].hand = p3_hand
        agents[1].hand = p4_hand
        agents[2].hand = p1_hand
        agents[3].hand = p2_hand
        
        # 重置分数
        for agent in agents:
            agent.score = 0
        
        print(f"✅ 手牌反向分配完成")
        print(f"📋 新的手牌分配:")
        for agent in agents:
            hand_names = [os.path.basename(img) for img in agent.hand]
            print(f"  {agent.name} ({agent.model_name}): {hand_names}")
    
    def run_experiment(self, rounds_per_phase: int = 12) -> Dict:
        """
        运行完整的公平Dixit实验
        
        Args:
            rounds_per_phase: 每个阶段的轮次数量
            
        Returns:
            实验结果
        """
        print("🚀 开始并行公平Dixit模型能力对比实验")
        print(f"📊 模型A ({self.model_a_name}): 控制P1和P2")
        print(f"📊 模型B ({self.model_b_name}): 控制P3和P4")
        print(f"🎮 每个阶段 {rounds_per_phase} 轮，总共 {rounds_per_phase * 2} 轮")
        print(f"⚡ 使用并行处理提高效率")
        
        # 创建玩家代理
        agents = self.create_agents()
        
        # 第一阶段：原始手牌分配
        print(f"\n{'='*60}")
        print("📈 第一阶段：原始手牌分配")
        print(f"{'='*60}")
        
        start_time = time.time()
        phase1_results = self.run_parallel_rounds(agents, "Phase 1", rounds_per_phase)
        phase1_time = time.time() - start_time
        
        # 记录第一阶段最终分数
        self.results['phase1']['rounds'] = phase1_results
        self.results['phase1']['final_scores'] = {agent.name: agent.score for agent in agents}
        
        print(f"\n📊 第一阶段完成，耗时: {phase1_time:.2f} 秒")
        print(f"📊 第一阶段最终分数:")
        for agent in agents:
            print(f"  {agent.name} ({agent.model_name}): {agent.score}分")
        
        # 手牌反向分配
        self.reverse_hands(agents)
        
        # 第二阶段：反向手牌分配
        print(f"\n{'='*60}")
        print("📈 第二阶段：反向手牌分配")
        print(f"{'='*60}")
        
        start_time = time.time()
        phase2_results = self.run_parallel_rounds(agents, "Phase 2", rounds_per_phase)
        phase2_time = time.time() - start_time
        
        # 记录第二阶段最终分数
        self.results['phase2']['rounds'] = phase2_results
        self.results['phase2']['final_scores'] = {agent.name: agent.score for agent in agents}
        
        print(f"\n📊 第二阶段完成，耗时: {phase2_time:.2f} 秒")
        print(f"📊 第二阶段最终分数:")
        for agent in agents:
            print(f"  {agent.name} ({agent.model_name}): {agent.score}分")
        
        # 计算总分
        self._calculate_total_scores()
        
        # 分析结果
        self._analyze_results()
        
        # 保存详细结果
        self._save_detailed_results()
        
        total_time = phase1_time + phase2_time
        print(f"\n🎉 实验完成! 总耗时: {total_time:.2f} 秒 ({total_time/60:.2f} 分钟)")
        
        return self.results
    
    def _calculate_total_scores(self):
        """计算每个玩家的总分（两个阶段之和）"""
        total_scores = {}
        for player in ['P1', 'P2', 'P3', 'P4']:
            phase1_score = self.results['phase1']['final_scores'].get(player, 0)
            phase2_score = self.results['phase2']['final_scores'].get(player, 0)
            total_scores[player] = phase1_score + phase2_score
        
        self.results['total_scores'] = total_scores
    
    def _analyze_results(self):
        """分析实验结果"""
        print(f"\n{'='*60}")
        print("📊 实验结果分析")
        print(f"{'='*60}")
        
        # 第一阶段结果
        print(f"\n🏆 第一阶段结果 (原始手牌分配):")
        for player, score in self.results['phase1']['final_scores'].items():
            model = self.model_a_name if player in ['P1', 'P2'] else self.model_b_name
            print(f"  {player} ({model}): {score} 分")
        
        # 第二阶段结果
        print(f"\n🏆 第二阶段结果 (反向手牌分配):")
        for player, score in self.results['phase2']['final_scores'].items():
            model = self.model_a_name if player in ['P1', 'P2'] else self.model_b_name
            print(f"  {player} ({model}): {score} 分")
        
        # 总分结果
        print(f"\n🏆 总分结果 (两个阶段之和):")
        for player, total_score in self.results['total_scores'].items():
            model = self.model_a_name if player in ['P1', 'P2'] else self.model_b_name
            print(f"  {player} ({model}): {total_score} 分")
        
        # 模型对比
        model_a_total = sum([self.results['total_scores'][p] for p in ['P1', 'P2']])
        model_b_total = sum([self.results['total_scores'][p] for p in ['P3', 'P4']])
        
        print(f"\n🏆 模型总分对比:")
        print(f"  {self.model_a_name} (P1+P2): {model_a_total} 分")
        print(f"  {self.model_b_name} (P3+P4): {model_b_total} 分")
        
        # 说书人和听众得分统计
        self._analyze_storyteller_listener_scores()
        
        if model_a_total > model_b_total:
            print(f"🎉 {self.model_a_name} 获胜!")
        elif model_b_total > model_a_total:
            print(f"🎉 {self.model_b_name} 获胜!")
        else:
            print("🤝 平局!")
    
    def _analyze_storyteller_listener_scores(self):
        """分析说书人和听众得分统计"""
        print(f"\n🎭 说书人和听众得分统计:")
        print(f"{'='*60}")
        
        # 统计每个模型作为说书人的得分
        model_a_storyteller_score = 0
        model_b_storyteller_score = 0
        
        # 统计每个模型作为听众的得分
        model_a_listener_score = 0
        model_b_listener_score = 0
        
        # 分析第一阶段
        for round_data in self.results['phase1']['rounds']:
            if 'error' in round_data:
                continue
            storyteller_model = round_data['storyteller_model']
            storyteller_score = round_data['storyteller_score']
            
            if storyteller_model == self.model_a_name:
                model_a_storyteller_score += storyteller_score
            else:
                model_b_storyteller_score += storyteller_score
            
            # 统计听众得分
            for player, score in round_data['listener_scores'].items():
                player_model = self.model_a_name if player in ['P1', 'P2'] else self.model_b_name
                if player_model == self.model_a_name:
                    model_a_listener_score += score
                else:
                    model_b_listener_score += score
        
        # 分析第二阶段
        for round_data in self.results['phase2']['rounds']:
            if 'error' in round_data:
                continue
            storyteller_model = round_data['storyteller_model']
            storyteller_score = round_data['storyteller_score']
            
            if storyteller_model == self.model_a_name:
                model_a_storyteller_score += storyteller_score
            else:
                model_b_storyteller_score += storyteller_score
            
            # 统计听众得分
            for player, score in round_data['listener_scores'].items():
                player_model = self.model_a_name if player in ['P1', 'P2'] else self.model_b_name
                if player_model == self.model_a_name:
                    model_a_listener_score += score
                else:
                    model_b_listener_score += score
        
        # 打印结果
        print(f"\n🎤 说书人得分统计:")
        print(f"  {self.model_a_name}: {model_a_storyteller_score} 分")
        print(f"  {self.model_b_name}: {model_b_storyteller_score} 分")
        
        print(f"\n👂 听众得分统计:")
        print(f"  {self.model_a_name}: {model_a_listener_score} 分")
        print(f"  {self.model_b_name}: {model_b_listener_score} 分")
        
        print(f"\n📊 角色得分对比:")
        print(f"  {self.model_a_name}: 说书人 {model_a_storyteller_score} 分 + 听众 {model_a_listener_score} 分 = {model_a_storyteller_score + model_a_listener_score} 分")
        print(f"  {self.model_b_name}: 说书人 {model_b_storyteller_score} 分 + 听众 {model_b_listener_score} 分 = {model_b_storyteller_score + model_b_listener_score} 分")
        
        # 保存到结果中
        self.results['storyteller_scores'] = {
            self.model_a_name: model_a_storyteller_score,
            self.model_b_name: model_b_storyteller_score
        }
        self.results['listener_scores'] = {
            self.model_a_name: model_a_listener_score,
            self.model_b_name: model_b_listener_score
        }
        
        # 保存模型名称到结果中
        self.results['model_a'] = self.model_a_name
        self.results['model_b'] = self.model_b_name
    
    def _save_detailed_results(self):
        """保存详细实验结果到文件"""
        try:
            with open('parallel_fair_dixit_experiment_results.md', 'w', encoding='utf-8') as f:
                f.write("# 🎮 Parallel Fair Dixit Model Comparison Experiment Report\n\n")
                
                f.write(f"## 📊 Experiment Configuration\n\n")
                f.write(f"- **Model A**: {self.model_a_name} (controls P1 and P2)\n")
                f.write(f"- **Model B**: {self.model_b_name} (controls P3 and P4)\n")
                f.write(f"- **Rounds per phase**: 12\n")
                f.write(f"- **Total rounds**: 24\n")
                f.write(f"- **Fairness design**: Hand cards are reversed between phases\n")
                f.write(f"- **Processing**: Parallel execution for improved efficiency\n\n")
                
                f.write(f"## 🏆 Final Results\n\n")
                
                # Phase 1 results
                f.write("### Phase 1 (Original Hand Distribution)\n\n")
                f.write("| Player | Model | Score |\n")
                f.write("|--------|-------|-------|\n")
                for player, score in self.results['phase1']['final_scores'].items():
                    model = self.model_a_name if player in ['P1', 'P2'] else self.model_b_name
                    f.write(f"| {player} | {model} | {score} |\n")
                f.write("\n")
                
                # Phase 2 results
                f.write("### Phase 2 (Reversed Hand Distribution)\n\n")
                f.write("| Player | Model | Score |\n")
                f.write("|--------|-------|-------|\n")
                for player, score in self.results['phase2']['final_scores'].items():
                    model = self.model_a_name if player in ['P1', 'P2'] else self.model_b_name
                    f.write(f"| {player} | {model} | {score} |\n")
                f.write("\n")
                
                # Total scores
                f.write("### Total Scores (Phase 1 + Phase 2)\n\n")
                f.write("| Player | Model | Total Score |\n")
                f.write("|--------|-------|-------------|\n")
                for player, total_score in self.results['total_scores'].items():
                    model = self.model_a_name if player in ['P1', 'P2'] else self.model_b_name
                    f.write(f"| {player} | {model} | {total_score} |\n")
                f.write("\n")
                
                # Model comparison
                model_a_total = sum([self.results['total_scores'][p] for p in ['P1', 'P2']])
                model_b_total = sum([self.results['total_scores'][p] for p in ['P3', 'P4']])
                
                f.write("### Model Comparison\n\n")
                f.write("| Model | Players | Total Score |\n")
                f.write("|-------|---------|-------------|\n")
                f.write(f"| {self.model_a_name} | P1 + P2 | {model_a_total} |\n")
                f.write(f"| {self.model_b_name} | P3 + P4 | {model_b_total} |\n\n")
                
                # Storyteller and Listener scores
                f.write("### Role-based Score Analysis\n\n")
                f.write("| Model | Storyteller Score | Listener Score | Total Role Score |\n")
                f.write("|-------|-------------------|----------------|------------------|\n")
                
                storyteller_scores = self.results.get('storyteller_scores', {})
                listener_scores = self.results.get('listener_scores', {})
                
                model_a_storyteller = storyteller_scores.get(self.model_a_name, 0)
                model_a_listener = listener_scores.get(self.model_a_name, 0)
                model_b_storyteller = storyteller_scores.get(self.model_b_name, 0)
                model_b_listener = listener_scores.get(self.model_b_name, 0)
                
                f.write(f"| {self.model_a_name} | {model_a_storyteller} | {model_a_listener} | {model_a_storyteller + model_a_listener} |\n")
                f.write(f"| {self.model_b_name} | {model_b_storyteller} | {model_b_listener} | {model_b_storyteller + model_b_listener} |\n\n")
                
                f.write("## 📈 Detailed Round Records\n\n")
                
                # Phase 1 rounds
                f.write("### Phase 1 Rounds\n\n")
                for round_data in self.results['phase1']['rounds']:
                    if 'error' in round_data:
                        f.write(f"#### Round {round_data['round_number']} - ERROR\n\n")
                        f.write(f"- **Error**: {round_data['error']}\n\n")
                        continue
                    
                    f.write(f"#### Round {round_data['round_number']}\n\n")
                    f.write(f"- **Storyteller**: {round_data['storyteller']} ({round_data['storyteller_model']})\n")
                    f.write(f"- **Target Image**: {round_data['target_image_name']}\n")
                    f.write(f"- **Description**: \"{round_data['description']}\"\n")
                    f.write(f"- **Candidate Images**: {', '.join(round_data['candidate_names'])}\n")
                    f.write(f"- **Guess Results**: {', '.join(round_data['guess_names'])}\n")
                    f.write(f"- **Round Scores**: {round_data['scores']}\n")
                    f.write(f"- **Cumulative Scores**: {round_data['total_scores']}\n\n")
                
                # Phase 2 rounds
                f.write("### Phase 2 Rounds\n\n")
                for round_data in self.results['phase2']['rounds']:
                    if 'error' in round_data:
                        f.write(f"#### Round {round_data['round_number']} - ERROR\n\n")
                        f.write(f"- **Error**: {round_data['error']}\n\n")
                        continue
                    
                    f.write(f"#### Round {round_data['round_number']}\n\n")
                    f.write(f"- **Storyteller**: {round_data['storyteller']} ({round_data['storyteller_model']})\n")
                    f.write(f"- **Target Image**: {round_data['target_image_name']}\n")
                    f.write(f"- **Description**: \"{round_data['description']}\"\n")
                    f.write(f"- **Candidate Images**: {', '.join(round_data['candidate_names'])}\n")
                    f.write(f"- **Guess Results**: {', '.join(round_data['guess_names'])}\n")
                    f.write(f"- **Round Scores**: {round_data['scores']}\n")
                    f.write(f"- **Cumulative Scores**: {round_data['total_scores']}\n\n")
            
            print("✅ Detailed results saved to parallel_fair_dixit_experiment_results.md")
            
        except Exception as e:
            print(f"❌ Failed to save results: {e}")

def main():
    """主函数"""
    print("🎮 并行公平Dixit模型能力对比实验")
    print("="*60)
    
    # 获取模型名称
    model_a_name = input("请输入模型A名称 (默认: qwen2.5-vl-7b): ").strip()
    if not model_a_name:
        model_a_name = "qwen2.5-vl-7b"
    
    model_b_name = input("请输入模型B名称 (默认: gpt-4o): ").strip()
    if not model_b_name:
        model_b_name = "gpt-4o"
    
    # 获取轮次数量
    try:
        rounds_per_phase = int(input("请输入每个阶段的轮次数量 (默认: 12): ").strip() or "12")
    except ValueError:
        rounds_per_phase = 12
    
    print(f"\n📋 实验配置:")
    print(f"  模型A ({model_a_name}): 控制P1和P2")
    print(f"  模型B ({model_b_name}): 控制P3和P4")
    print(f"  每阶段轮次: {rounds_per_phase}")
    print(f"  总轮次: {rounds_per_phase * 2}")
    print(f"  公平性设计: 手牌反向分配")
    print(f"  处理方式: 并行执行提高效率")
    
    # 创建实验实例
    experiment = ParallelFairDixitExperiment(model_a_name, model_b_name)
    
    # 运行实验
    results = experiment.run_experiment(rounds_per_phase)
    
    print(f"\n🎉 实验完成!")
    print(f"详细结果已保存到 parallel_fair_dixit_experiment_results.md")

if __name__ == "__main__":
    main()
