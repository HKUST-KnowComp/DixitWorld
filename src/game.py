# -*- coding: utf-8 -*-
"""
Dixit游戏核心逻辑模块
定义了游戏流程、回合管理和得分计算
"""

import random
import logging
from typing import List, Dict, Any
from agents import BaseAgent, PlayerAgent
from config import CARDS_PER_PLAYER, SCORING_RULES

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DixitGame:
    """Dixit游戏类"""

    def __init__(self, agents: List[PlayerAgent], image_pool: List[str]):
        """
        初始化Dixit游戏
        
        Args:
            agents: 玩家代理列表
            image_pool: 图像文件路径列表
        """
        if len(agents) < 3:
            raise ValueError("Dixit game requires at least 3 players.")
        self.agents = agents
        self.image_pool = image_pool
        self.current_round = 0
        self.game_results = [] # 存储每轮游戏结果

    def _deal_initial_hands(self):
        """
        为所有玩家分发初始手牌
        每人4张牌
        """
        random.shuffle(self.image_pool)
        for agent in self.agents:
            agent.receive_hand(self.image_pool[:CARDS_PER_PLAYER])
            self.image_pool = self.image_pool[CARDS_PER_PLAYER:]
            hand_names = [img.split('/')[-1] for img in agent.hand]
            logging.info(f"玩家 {agent.name} ({agent.model_name}) 初始手牌: {hand_names}")

    def play_round(self, storyteller: PlayerAgent, round_num: int) -> Dict[str, Any]:
        """
        进行一轮游戏
        
        Args:
            storyteller: 当前的说书人
            round_num: 当前轮次编号
            
        Returns:
            本轮游戏结果字典
        """
        logging.info(f"\n--- 第 {round_num} 轮开始 ---")
        logging.info(f"说书人: {storyteller.name} ({storyteller.model_name})")

        # 确保说书人手牌中有足够的牌
        if len(storyteller.hand) < 1:
            logging.warning(f"说书人 {storyteller.name} 手牌不足，重新发牌。")
            storyteller.receive_hand(random.sample(self.image_pool, CARDS_PER_PLAYER))
            self.image_pool = [img for img in self.image_pool if img not in storyteller.hand]

        # 说书人选择目标图像
        target_image = storyteller.select_target_image()
        logging.info(f"说书人 {storyteller.name} 选择目标图像: {target_image.split('/')[-1]}")

        # 说书人生成描述
        description = storyteller.generate_description(target_image)
        logging.info(f"说书人 {storyteller.name} 描述: \"{description}\"")

        # 其他玩家（猜测者）选择干扰图像
        guessers = [agent for agent in self.agents if agent != storyteller]
        distractor_images = []
        distractor_image_names = []
        for guesser in guessers:
            # 确保猜测者手牌中有足够的牌
            if len(guesser.hand) < 1:
                logging.warning(f"猜测者 {guesser.name} 手牌不足，重新发牌。")
                guesser.receive_hand(random.sample(self.image_pool, CARDS_PER_PLAYER))
                self.image_pool = [img for img in self.image_pool if img not in guesser.hand]

            distractor = guesser.select_distractor_image_with_description(description)
            guesser.selected_distractor = distractor # 记住选择的干扰图像
            distractor_images.append(distractor)
            distractor_image_names.append(distractor.split('/')[-1])
            logging.info(f"猜测者 {guesser.name} ({guesser.model_name}) 选择干扰图像: {distractor.split('/')[-1]}")

        # 组合所有图像并洗牌作为候选图像
        candidate_images = [target_image] + distractor_images
        random.shuffle(candidate_images)
        candidate_names = [img.split('/')[-1] for img in candidate_images]
        logging.info(f"候选图像 (洗牌后): {candidate_names}")

        # 猜测者进行最终猜测
        guesser_guesses = {}
        guess_names = []
        for guesser in guessers:
            # 玩家不能选择自己的干扰图像
            available_candidates = [img for img in candidate_images if guesser.selected_distractor and img != guesser.selected_distractor]
            if not available_candidates: # 如果只剩下自己的干扰牌，或者没有其他牌了，就从所有牌里选
                available_candidates = candidate_images
            
            guess = guesser.guess_target_image(description, available_candidates)
            guesser_guesses[guesser.name] = guess
            guess_names.append(guess.split('/')[-1])
            logging.info(f"猜测者 {guesser.name} ({guesser.model_name}) 猜测: {guess.split('/')[-1]}")

        # 计算得分
        round_scores = self._calculate_scores(storyteller, guessers, target_image, guesser_guesses)
        
        # 更新玩家总分
        for agent in self.agents:
            agent.add_score(round_scores.get(agent.name, 0))

        # 准备本轮结果
        round_result = {
            'round_number': round_num,
            'storyteller': storyteller.name,
            'storyteller_model': storyteller.model_name,
            'target_image_name': target_image.split('/')[-1],
            'description': description,
            'distractor_images': distractor_images,
            'distractor_image_names': distractor_image_names,
            'shuffled_candidates': candidate_images,
            'candidate_names': candidate_names,
            'guesser_guesses': guesser_guesses,
            'guess_names': guess_names,
            'player_hands': {agent.name: [img.split('/')[-1] for img in agent.hand] for agent in self.agents},
            'scores': round_scores,
            'cumulative_scores': {agent.name: agent.score for agent in self.agents},
            'storyteller_score': round_scores.get(storyteller.name, 0),
            'listener_scores': {guesser.name: round_scores.get(guesser.name, 0) for guesser in guessers}
        }
        self.game_results.append(round_result)

        logging.info(f"--- 第 {round_num} 轮结束 ---")
        logging.info(f"本轮得分: {round_scores}")
        logging.info(f"当前总分: {', '.join([f'{agent.name}: {agent.score}' for agent in self.agents])}")
        return round_result

    def _calculate_scores(self, storyteller: PlayerAgent, guessers: List[PlayerAgent], 
                          target_image: str, guesser_guesses: Dict[str, str]) -> Dict[str, int]:
        """
        根据Dixit规则计算本轮得分
        
        Args:
            storyteller: 说书人
            guessers: 猜测者列表
            target_image: 说书人选择的目标图像
            guesser_guesses: 猜测者选择的图像字典
            
        Returns:
            本轮各玩家得分字典
        """
        scores = {agent.name: 0 for agent in self.agents}
        target_image_name = target_image.split('/')[-1]

        correct_guesses_count = 0
        correct_guessers = []

        # 统计猜对的人数和猜对的玩家
        for guesser in guessers:
            if guesser_guesses.get(guesser.name) == target_image:
                correct_guesses_count += 1
                correct_guessers.append(guesser)
                scores[guesser.name] += SCORING_RULES['guesser_correct'] # 猜对的人得3分

        # 说书人得分
        if correct_guesses_count == 0 or correct_guesses_count == len(guessers):
            # 没人猜对 或 所有人都猜对：说书人得0分
            scores[storyteller.name] += 0
            if correct_guesses_count == 0:
                # 没人猜对，其他所有玩家（包括猜错的）各得2分
                for guesser in guessers:
                    scores[guesser.name] += SCORING_RULES['others_when_storyteller_loses']
        else:
            # 部分人猜对：说书人得3分
            scores[storyteller.name] += SCORING_RULES['storyteller_win']

        # 听众的牌被选择的奖励分
        for guesser in guessers:
            if guesser.selected_distractor and guesser.selected_distractor in guesser_guesses.values():
                # 检查是否有其他玩家选择了这个干扰图像
                for other_guesser_name, other_guesser_guess in guesser_guesses.items():
                    if other_guesser_name != guesser.name and other_guesser_guess == guesser.selected_distractor:
                        scores[guesser.name] += SCORING_RULES['card_selected_bonus'] # 听众的牌被选择：每次得1分
                        logging.info(f"玩家 {guesser.name} 的干扰牌 {guesser.selected_distractor.split('/')[-1]} 被 {other_guesser_name} 选中，获得1分奖励。")
        
        return scores

    def get_game_summary(self) -> Dict[str, Any]:
        """
        获取游戏总结
        
        Returns:
            游戏总结字典
        """
        return {
            'total_rounds': self.current_round,
            'final_scores': {agent.name: agent.score for agent in self.agents},
            'game_results': self.game_results,
            'agents': self.agents # 包含代理列表，方便后续处理
        }
