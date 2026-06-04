# -*- coding: utf-8 -*-
"""
AI代理模块
定义说书人和猜测者类，包含VLM调用的JSON格式支持
"""

import logging
from typing import List, Dict, Any
import config

# 导入VLM API调用模块
try:
    from call_api_openrouter import call_llm_api
    VLM_AVAILABLE = True
except ImportError:
    VLM_AVAILABLE = False
    logging.warning("VLM API模块未找到，将使用占位符实现")

class BaseAgent:
    """AI代理基类"""
    
    def __init__(self, agent_id: int, name: str = None):
        """
        初始化代理
        
        Args:
            agent_id: 代理ID
            name: 代理名称
        """
        self.agent_id = agent_id
        self.name = name or f"Agent_{agent_id}"
        self.hand = []  # 手牌
        self.score = 0  # 分数
        self.vlm_config = config.VLM_CONFIG
        
    def receive_hand(self, hand: List[str]):
        """接收手牌"""
        self.hand = hand.copy()
        logging.info(f"{self.name} 收到 {len(hand)} 张手牌")
    
    def add_score(self, points: int):
        """增加分数"""
        self.score += points
        logging.info(f"{self.name} 获得 {points} 分，总分: {self.score}")
    
    def get_score(self) -> int:
        """获取当前分数"""
        return self.score

class PlayerAgent(BaseAgent):
    """玩家代理 - 可以担任说书人或猜测者"""
    
    def __init__(self, agent_id: int, name: str = None):
        super().__init__(agent_id, name)
        self.role = "player"  # 默认为普通玩家
        self.selected_distractor = None  # 记住选择的干扰图像
        self.selected_target = None  # 记住选择的目标图像
        self.model_name = "default"  # 模型名称
    
    
    def generate_description(self, target_image: str) -> str:
        """
        根据目标图像生成描述
        
        Args:
            target_image: 目标图像路径
            
        Returns:
            生成的描述文本
        """
        # 检查是否启用VLM调用
        if not (config.VLM_CONFIG.get('enable_vlm', False) and 
            VLM_AVAILABLE and 
            config.VLM_CONFIG.get('model_name') != 'placeholder_model'):
            raise RuntimeError("VLM不可用，无法生成描述")
            
        try:
            # 构建VLM提示，让模型完全自己分析图片并生成描述
            prompt = f"""You are a storyteller in a Dixit game. This is a competitive game where your scoring depends on creating the PERFECT level of difficulty.

CRITICAL SCORING RULES:
- ALL players guess correctly = You get 0 points (FAILURE - too obvious)
- NO players guess correctly = You get 0 points (FAILURE - too vague)
- SOME guess correctly, SOME guess wrong = You get 3 points (SUCCESS!)

DIXIT STRATEGY MASTERY:
Your description should be like a riddle - intriguing enough to attract some players, but misleading enough that others choose wrong candidates.

ADVANCED TECHNIQUES:
- Use poetic/metaphorical language that could apply to multiple images
- Focus on emotions, atmosphere, or abstract concepts rather than concrete objects
- Create ambiguity: describe something that could be interpreted multiple ways
- Use misdirection: emphasize elements that might exist in distractor images too
- Think laterally: what feeling or story does this image evoke, not what objects are in it?

EXAMPLES OF GOOD DIXIT DESCRIPTIONS:
- Instead of "a castle in the sky" → "dreams taking flight"
- Instead of "a person reading" → "whispers from forgotten pages"
- Instead of "a red flower" → "passion blooming in solitude"

Your target: Exactly 1-2 out of 3 players should guess correctly. Make it a perfect puzzle!

IMPORTANT: You must respond in the following JSON format:
{{
    "reasoning": "Your strategic analysis of why this description will create the perfect split between correct and incorrect guesses",
    "answer": "Your masterfully crafted Dixit description that will challenge players at exactly the right level"
}}

Focus on creating ambiguity and intrigue, not clarity."""

            # 构建包含图片的消息
            import base64
            try:
                with open(target_image, "rb") as image_file:
                    image_data = base64.b64encode(image_file.read()).decode('utf-8')
                
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_data}"
                                }
                            }
                        ]
                    }
                ]
            except Exception as e:
                logging.warning(f"无法读取图片文件 {target_image}: {e}，使用文本模式")
                messages = [{"role": "user", "content": prompt}]
            
            # 调用VLM API
            logging.info(f"{self.name} 使用模型 {self.model_name} 分析图片 {target_image} 并生成描述")
            content, _, _ = call_llm_api(
                messages, 
                self.model_name,
                temperature=config.VLM_CONFIG['temperature']
            )
                
            # 解析JSON响应
            if isinstance(content, dict):
                reasoning = content.get('reasoning', '')
                answer = content.get('answer', '')
                
                # 限制reasoning长度
                if len(reasoning) > 200:
                    reasoning = reasoning[:200] + "..."
                
                logging.info(f"{self.name} VLM推理过程: {reasoning}")
                logging.info(f"{self.name} VLM最终答案: {answer}")
                
                return answer.strip()
            else:
                # 兼容旧格式（字符串）
                try:
                    import json
                    response_data = json.loads(content.strip())
                    reasoning = response_data.get('reasoning', '')
                    answer = response_data.get('answer', content.strip())
                    
                    # 限制reasoning长度
                    if len(reasoning) > 200:
                        reasoning = reasoning[:200] + "..."
                    
                    logging.info(f"{self.name} VLM推理过程: {reasoning}")
                    logging.info(f"{self.name} VLM最终答案: {answer}")
                    
                    return answer.strip()
                    
                except json.JSONDecodeError:
                    logging.warning(f"{self.name} VLM返回的不是有效JSON，使用原始内容: {content}")
                    return content.strip()
            
        except Exception as e:
            logging.error(f"VLM调用失败: {e}")
            raise RuntimeError(f"VLM调用失败: {e}")
    
    def select_target_image(self) -> str:
        """
        从手牌中选择目标图像（使用VLM分析选择）
        
        Returns:
            选中的目标图像路径
        """
        # 检查是否启用VLM调用
        if not (config.VLM_CONFIG.get('enable_vlm', False) and 
            VLM_AVAILABLE and 
            config.VLM_CONFIG.get('model_name') != 'placeholder_model'):
            # 如果VLM不可用，回退到随机选择
            import random
            target_image = random.choice(self.hand)
            logging.info(f"{self.name} VLM不可用，随机选择目标图像: {target_image}")
            return target_image

        try:
            # 构建VLM提示，让说书人分析自己的手牌并选择目标图像
            hand_info = "\n".join([f"Card {i+1}: {img.split('/')[-1]}" for i, img in enumerate(self.hand)])
            
            prompt = f"""You are a storyteller in a Dixit game. You need to select one card from your 4-card hand as the target image.

Your hand:
{hand_info}

Dixit game rules:
1. Your goal is to generate a description that allows some players to guess correctly while others guess incorrectly
2. If everyone guesses correctly: You get 0 points, correct guessers get 3 points each
3. If some people guess correctly: You get 3 points, correct guessers get 3 points each
4. If no one guesses correctly: You get 0 points, others get 2 points each

Please analyze your hand and select the card most suitable as the target image. Consider:
- Whether the image content is rich enough to provide sufficient descriptive space
- Whether it's easy to generate a description that is both clear and ambiguous
- Whether it can attract some but not all players to choose it

IMPORTANT: You must respond in the following JSON format:
{{
    "reasoning": "Brief analysis of why you chose this card (max 50 words)",
    "answer": "The card number (1-4) you selected"
}}

The reasoning field should be concise (maximum 50 words), and the answer field should contain only the card number."""

            # 构建包含多张图片的消息
            import base64
            content_parts = [
                {
                    "type": "text",
                    "text": prompt
                }
            ]
            
            try:
                # 限制最多5张图片避免API限制
                hand_subset = self.hand[:5] if len(self.hand) > 5 else self.hand
                
                for i, image_path in enumerate(hand_subset, 1):
                    with open(image_path, "rb") as image_file:
                        image_data = base64.b64encode(image_file.read()).decode('utf-8')
                    
                    content_parts.append({
                        "type": "text",
                        "text": f"\nCard {i} ({image_path.split('/')[-1]}):"
                    })
                    content_parts.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_data}"
                        }
                    })
                
                messages = [
                    {
                        "role": "user",
                        "content": content_parts
                    }
                ]
            except Exception as e:
                logging.warning(f"无法读取图片文件: {e}，使用文本模式")
                messages = [{"role": "user", "content": prompt}]
            
            # 调用VLM API
            logging.info(f"{self.name} 使用模型 {self.model_name} 分析手牌选择目标图像")
            content, _, _ = call_llm_api(
                messages, 
                self.model_name,
                temperature=config.VLM_CONFIG['temperature']
            )
            
            # 解析JSON响应
            if isinstance(content, dict):
                reasoning = content.get('reasoning', '')
                answer = content.get('answer', '')
                
                # 限制reasoning长度
                if len(reasoning) > 200:
                    reasoning = reasoning[:200] + "..."
                
                logging.info(f"{self.name} VLM推理过程: {reasoning}")
                logging.info(f"{self.name} VLM最终答案: {answer}")
                
                    # 解析卡片编号（适应实际发送的图片数量）
                try:
                    card_number = int(answer)
                    hand_subset = self.hand[:5] if len(self.hand) > 5 else self.hand
                    if 1 <= card_number <= len(hand_subset):
                        target_image = hand_subset[card_number - 1]
                        self.selected_target = target_image  # 记住选择的目标图像
                        logging.info(f"{self.name} 使用VLM选择目标图像: {target_image}")
                        return target_image
                    else:
                        raise ValueError(f"卡片编号超出范围: {card_number}")
                except ValueError as e:
                    logging.warning(f"VLM返回的卡片编号无效: {answer}, 错误: {e}，使用随机选择")
                    import random
                    target_image = random.choice(self.hand)
                    self.selected_target = target_image  # 记住选择的目标图像
                    logging.info(f"{self.name} VLM选择失败，随机选择目标图像: {target_image}")
                    return target_image
            else:
                # 兼容旧格式（字符串）
                try:
                    import json
                    response_data = json.loads(content.strip())
                    reasoning = response_data.get('reasoning', '')
                    answer = response_data.get('answer', content.strip())
                    
                    # 限制reasoning长度
                    if len(reasoning) > 200:
                        reasoning = reasoning[:200] + "..."
                    
                    logging.info(f"{self.name} VLM推理过程: {reasoning}")
                    logging.info(f"{self.name} VLM最终答案: {answer}")
                    
                    # 解析卡片编号
                    card_number = int(answer.strip())
                    if 1 <= card_number <= 4:
                        target_image = self.hand[card_number - 1]
                        self.selected_target = target_image  # 记住选择的目标图像
                        logging.info(f"{self.name} 使用VLM选择目标图像: {target_image}")
                        return target_image
                    else:
                        raise ValueError(f"卡片编号超出范围: {card_number}")
                        
                except (json.JSONDecodeError, ValueError) as e:
                    logging.warning(f"VLM返回无效: {content}, 错误: {e}，使用随机选择")
                    # 如果VLM选择失败，使用随机选择
                    import random
                    target_image = random.choice(self.hand)
                    self.selected_target = target_image  # 记住选择的目标图像
                    logging.info(f"{self.name} VLM选择失败，随机选择目标图像: {target_image}")
                    return target_image
            
        except Exception as e:
            logging.error(f"VLM调用失败: {e}")
            # 如果VLM调用失败，使用随机选择
            import random
            target_image = random.choice(self.hand)
            self.selected_target = target_image  # 记住选择的目标图像
            logging.info(f"{self.name} VLM调用失败，随机选择目标图像: {target_image}")
            return target_image
    
    def select_distractor_image_with_description(self, description: str) -> str:
        """
        根据说书人的描述选择干扰图像
        
        Args:
            description: 说书人的描述
        
        Returns:
            选中的干扰图像路径
        """
        # 检查是否启用VLM调用
        if not (config.VLM_CONFIG.get('enable_vlm', False) and 
            VLM_AVAILABLE and 
            config.VLM_CONFIG.get('model_name') != 'placeholder_model'):
            # 如果VLM不可用，回退到随机选择
            import random
            distractor_image = random.choice(self.hand)
            logging.info(f"{self.name} VLM不可用，随机选择干扰图像: {distractor_image}")
            return distractor_image
            
        try:
            # 构建VLM提示，让猜测者根据描述选择干扰图像
            hand_info = "\n".join([f"Card {i+1}: {img.split('/')[-1]}" for i, img in enumerate(self.hand)])
            
            prompt = f"""You are a player in a Dixit game. The storyteller gave this description: "{description}"

Your hand:
{hand_info}

Your goal is to select the card from your hand that best matches the storyteller's description. This card will be used as a distractor to confuse other players.

IMPORTANT: You must respond in the following JSON format:
{{
    "reasoning": "Brief analysis of why you chose this card (max 50 words)",
    "answer": "The card number (1-4) you selected"
}}

The reasoning field should be concise (maximum 50 words), and the answer field should contain only the card number."""

            # 构建包含多张图片的消息
            import base64
            content_parts = [
                {
                    "type": "text",
                    "text": prompt
                }
            ]
            
            try:
                # 限制最多5张图片避免API限制
                hand_subset = self.hand[:5] if len(self.hand) > 5 else self.hand
                
                for i, image_path in enumerate(hand_subset, 1):
                    with open(image_path, "rb") as image_file:
                        image_data = base64.b64encode(image_file.read()).decode('utf-8')
                    
                    content_parts.append({
                        "type": "text",
                        "text": f"\nCard {i} ({image_path.split('/')[-1]}):"
                    })
                    content_parts.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_data}"
                        }
                    })
                
                messages = [
                    {
                        "role": "user",
                        "content": content_parts
                    }
                ]
            except Exception as e:
                logging.warning(f"无法读取图片文件: {e}，使用文本模式")
                messages = [{"role": "user", "content": prompt}]
            
            # 调用VLM API
            logging.info(f"{self.name} 使用模型 {self.model_name} 根据描述选择干扰图像")
            content, _, _ = call_llm_api(
                messages, 
                self.model_name,
                temperature=config.VLM_CONFIG['temperature']
            )
            
            # 解析JSON响应
            if isinstance(content, dict):
                reasoning = content.get('reasoning', '')
                answer = content.get('answer', '')
                
                # 限制reasoning长度
                if len(reasoning) > 200:
                    reasoning = reasoning[:200] + "..."
                
                logging.info(f"{self.name} VLM推理过程: {reasoning}")
                logging.info(f"{self.name} VLM最终答案: {answer}")
                
                # 解析卡片编号（适应实际发送的图片数量）
                try:
                    card_number = int(answer)
                    hand_subset = self.hand[:5] if len(self.hand) > 5 else self.hand
                    if 1 <= card_number <= len(hand_subset):
                        distractor_image = hand_subset[card_number - 1]
                        logging.info(f"{self.name} 使用VLM选择干扰图像: {distractor_image}")
                        return distractor_image
                    else:
                        raise ValueError(f"卡片编号超出范围: {card_number}")
                except ValueError as e:
                    logging.warning(f"VLM返回的卡片编号无效: {answer}, 错误: {e}，使用随机选择")
                    import random
                    distractor_image = random.choice(self.hand)
                    logging.info(f"{self.name} VLM选择失败，随机选择干扰图像: {distractor_image}")
                    return distractor_image
            else:
                # 兼容旧格式（字符串）
                try:
                    import json
                    response_data = json.loads(content.strip())
                    reasoning = response_data.get('reasoning', '')
                    answer = response_data.get('answer', content.strip())
                    
                    # 限制reasoning长度
                    if len(reasoning) > 200:
                        reasoning = reasoning[:200] + "..."
                    
                    logging.info(f"{self.name} VLM推理过程: {reasoning}")
                    logging.info(f"{self.name} VLM最终答案: {answer}")
                    
                    # 解析卡片编号
                    card_number = int(answer.strip())
                    hand_subset = self.hand[:5] if len(self.hand) > 5 else self.hand
                    if 1 <= card_number <= len(hand_subset):
                        distractor_image = hand_subset[card_number - 1]
                        logging.info(f"{self.name} 使用VLM选择干扰图像: {distractor_image}")
                        return distractor_image
                    else:
                        raise ValueError(f"卡片编号超出范围: {card_number}")
                        
                except (json.JSONDecodeError, ValueError) as e:
                    logging.warning(f"VLM返回无效: {content}, 错误: {e}，使用随机选择")
                    # 如果VLM选择失败，使用随机选择
                    import random
                    distractor_image = random.choice(self.hand)
                    logging.info(f"{self.name} VLM选择失败，随机选择干扰图像: {distractor_image}")
                    return distractor_image
            
        except Exception as e:
            logging.error(f"VLM调用失败: {e}")
            # 如果VLM调用失败，使用随机选择
            import random
            distractor_image = random.choice(self.hand)
            logging.info(f"{self.name} VLM调用失败，随机选择干扰图像: {distractor_image}")
            return distractor_image
    
    def select_distractor_image(self, description: str) -> str:
        """
        根据描述选择干扰图像（简化版本）
        
        Args:
            description: 说书人的描述
            
        Returns:
            选中的干扰图像路径
        """
        return self.select_distractor_image_with_description(description)
    
    def guess_target_image(self, description: str, candidate_images: List[str]) -> str:
        """
        根据描述从候选图像中猜测目标图像
        
        Args:
            description: 说书人的描述
            candidate_images: 候选图像列表
            
        Returns:
            猜测的目标图像路径
        """
        # 检查是否启用VLM调用
        if not (config.VLM_CONFIG.get('enable_vlm', False) and 
            VLM_AVAILABLE and 
            config.VLM_CONFIG.get('model_name') != 'placeholder_model'):
            # 如果VLM不可用，回退到随机选择
            import random
            guess_image = random.choice(candidate_images)
            logging.info(f"{self.name} VLM不可用，随机猜测目标图像: {guess_image}")
            return guess_image
            
        try:
            # 构建VLM提示，让猜测者根据描述从候选图像中选择
            candidate_info = "\n".join([f"Candidate {i+1}: {img.split('/')[-1]}" for i, img in enumerate(candidate_images)])
            
            prompt = f"""You are a player in a Dixit game. The storyteller gave this description: "{description}"

Available candidate images:
{candidate_info}

Your goal is to select the candidate image that best matches the storyteller's description. This is the final guess for the target image.

IMPORTANT: You must respond in the following JSON format:
{{
    "reasoning": "Brief analysis of why you chose this candidate (max 50 words)",
    "answer": "The candidate number (1-{len(candidate_images)}) you selected"
}}

The reasoning field should be concise (maximum 50 words), and the answer field should contain only the candidate number from the available candidates above."""

            # 构建包含多张图片的消息
            import base64
            content_parts = [
                {
                    "type": "text",
                    "text": prompt
                }
            ]
            
            try:
                for i, image_path in enumerate(candidate_images, 1):
                    with open(image_path, "rb") as image_file:
                        image_data = base64.b64encode(image_file.read()).decode('utf-8')
                    
                    content_parts.append({
                        "type": "text",
                        "text": f"\nCandidate {i} ({image_path.split('/')[-1]}):"
                    })
                    content_parts.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_data}"
                        }
                    })
                
                messages = [
                    {
                        "role": "user",
                        "content": content_parts
                    }
                ]
            except Exception as e:
                logging.warning(f"无法读取图片文件: {e}，使用文本模式")
                messages = [{"role": "user", "content": prompt}]
            
            # 调用VLM API
            logging.info(f"{self.name} 使用模型 {self.model_name} 根据描述猜测目标图像")
            content, _, _ = call_llm_api(
                messages, 
                self.model_name,
                temperature=config.VLM_CONFIG['temperature']
            )
                
            # 解析JSON响应
            if isinstance(content, dict):
                reasoning = content.get('reasoning', '')
                answer = content.get('answer', '')
                
                # 限制reasoning长度
                if len(reasoning) > 200:
                    reasoning = reasoning[:200] + "..."
                
                logging.info(f"{self.name} VLM推理过程: {reasoning}")
                logging.info(f"{self.name} VLM最终答案: {answer}")
                
                # 解析候选编号
                try:
                    # 如果answer为空字符串，使用默认值1
                    if not answer or (isinstance(answer, str) and answer.strip() == ''):
                        answer = "1"
                        logging.info(f"{self.name} VLM返回空答案，使用默认值1")
                    
                    # 确保answer是字符串，然后转换为整数
                    if isinstance(answer, str):
                        candidate_number = int(answer.strip())
                    else:
                        candidate_number = int(answer)
                    
                    # 处理VLM返回0的情况（调整为1）
                    if candidate_number == 0:
                        candidate_number = 1
                        logging.info(f"{self.name} VLM返回编号0，调整为1")
                    elif candidate_number > len(candidate_images):
                        candidate_number = len(candidate_images)
                        logging.info(f"{self.name} VLM返回编号{candidate_number}超出范围，调整为{len(candidate_images)}")
                    
                    if 1 <= candidate_number <= len(candidate_images):
                        guess_image = candidate_images[candidate_number - 1]
                        logging.info(f"{self.name} 使用VLM猜测目标图像: {guess_image}")
                        return guess_image
                    else:
                        raise ValueError(f"候选编号超出范围: {candidate_number}")
                except ValueError as e:
                    logging.warning(f"VLM返回的候选编号无效: {answer}, 错误: {e}，使用随机选择")
                    import random
                    guess_image = random.choice(candidate_images)
                    logging.info(f"{self.name} VLM选择失败，随机猜测目标图像: {guess_image}")
                    return guess_image
            else:
                # 兼容旧格式（字符串）
                try:
                    import json
                    response_data = json.loads(content.strip())
                    reasoning = response_data.get('reasoning', '')
                    answer = response_data.get('answer', content.strip())
                    
                    # 限制reasoning长度
                    if len(reasoning) > 200:
                        reasoning = reasoning[:200] + "..."
                    
                    logging.info(f"{self.name} VLM推理过程: {reasoning}")
                    logging.info(f"{self.name} VLM最终答案: {answer}")
                    
                    # 解析候选编号
                    # 如果answer为空字符串，使用默认值1
                    if not answer or (isinstance(answer, str) and answer.strip() == ''):
                        answer = "1"
                        logging.info(f"{self.name} VLM返回空答案，使用默认值1")
                    
                    # 确保answer是字符串，然后转换为整数
                    if isinstance(answer, str):
                        candidate_number = int(answer.strip())
                    else:
                        candidate_number = int(answer)
                    
                    # 处理VLM返回0的情况（调整为1）
                    if candidate_number == 0:
                        candidate_number = 1
                        logging.info(f"{self.name} VLM返回编号0，调整为1")
                    elif candidate_number > len(candidate_images):
                        candidate_number = len(candidate_images)
                        logging.info(f"{self.name} VLM返回编号{candidate_number}超出范围，调整为{len(candidate_images)}")
                    
                    if 1 <= candidate_number <= len(candidate_images):
                        guess_image = candidate_images[candidate_number - 1]
                        logging.info(f"{self.name} 使用VLM猜测目标图像: {guess_image}")
                        return guess_image
                    else:
                        raise ValueError(f"候选编号超出范围: {candidate_number}")
                        
                except (json.JSONDecodeError, ValueError) as e:
                    logging.warning(f"VLM返回无效: {content}, 错误: {e}，使用随机选择")
                    # 如果VLM选择失败，使用随机选择
                    import random
                    guess_image = random.choice(candidate_images)
                    logging.info(f"{self.name} VLM选择失败，随机猜测目标图像: {guess_image}")
                    return guess_image
            
        except Exception as e:
            logging.error(f"VLM调用失败: {e}")
            # 如果VLM调用失败，使用随机选择
            import random
            guess_image = random.choice(candidate_images)
            logging.info(f"{self.name} VLM调用失败，随机猜测目标图像: {guess_image}")
            return guess_image
    
    def compute_entailment_scores(self, description: str, candidate_images: List[str]) -> List[float]:
        """
        计算每个候选图像对线索的蕴含得分
        
        Args:
            description: 说书人的线索描述
            candidate_images: 候选图像列表
            
        Returns:
            每个候选图像的蕴含得分列表 (0-100)
        """
        scores = []
        
        # 检查是否启用VLM调用
        if not (config.VLM_CONFIG.get('enable_vlm', False) and 
            VLM_AVAILABLE and 
            config.VLM_CONFIG.get('model_name') != 'placeholder_model'):
            # 如果VLM不可用，返回随机分数
            import random
            scores = [random.uniform(20, 80) for _ in candidate_images]
            logging.info(f"{self.name} VLM不可用，使用随机蕴含得分: {[f'{s:.1f}' for s in scores]}")
            return scores
        
        for i, image_path in enumerate(candidate_images):
            try:
                # 为每个候选图像单独计算蕴含得分
                prompt = f"""You are evaluating how well an image matches a given clue in a Dixit game.

Clue: "{description}"

Please evaluate how well this image supports or entails the given clue. Consider:
- Visual elements that match the clue
- Conceptual or metaphorical connections
- Overall thematic alignment
- Symbolic and abstract interpretations

Analyze the image carefully and provide your detailed reasoning steps towards determining a numerical score from 0 to 100, where 0 means completely unrelated and 100 means perfect match.

IMPORTANT: You must respond in the following JSON format:
{{
    "reasoning": "Your detailed reasoning steps towards the numerical score",
    "answer": "Your numerical rating (0-100)"
}}

Provide thorough reasoning that explains your scoring process and the answer field should contain only the numerical rating."""

                # 构建包含单张图片的消息
                import base64
                try:
                    with open(image_path, "rb") as image_file:
                        image_data = base64.b64encode(image_file.read()).decode('utf-8')
                    
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": prompt
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{image_data}"
                                    }
                                }
                            ]
                        }
                    ]
                except Exception as e:
                    logging.warning(f"无法读取图片文件 {image_path}: {e}，使用文本模式")
                    messages = [{"role": "user", "content": prompt}]
                
                # 调用VLM API
                logging.info(f"{self.name} 计算候选图像 {i+1} 的蕴含得分")
                content, _, _ = call_llm_api(
                    messages, 
                    self.model_name,
                    temperature=config.VLM_CONFIG['temperature']  # 使用与其他调用相同的温度
                )
                
                # 解析得分（与其他方法保持一致的解析逻辑）
                try:
                    if isinstance(content, dict):
                        reasoning = content.get('reasoning', '')
                        answer = content.get('answer', '50')
                        
                        # 不限制reasoning长度，显示完整推理过程
                        logging.info(f"{self.name} 蕴含评分推理: {reasoning}")
                        logging.info(f"{self.name} 蕴含评分答案: {answer}")
                        
                        # 解析得分
                        try:
                            score = float(str(answer).strip())
                            # 确保得分在0-100范围内
                            score = max(0, min(100, score))
                        except ValueError:
                            # 如果无法直接转换，尝试提取数字
                            import re
                            numbers = re.findall(r'\d+(?:\.\d+)?', str(answer))
                            if numbers:
                                score = float(numbers[0])
                                score = max(0, min(100, score))
                            else:
                                score = 50.0
                                logging.warning(f"无法解析蕴含得分，使用默认值50: {answer}")
                        
                        scores.append(score)
                        logging.info(f"{self.name} 候选图像 {i+1} 蕴含得分: {score}")
                        
                    else:
                        # 兼容旧格式（字符串）
                        try:
                            import json
                            response_data = json.loads(content.strip())
                            answer = response_data.get('answer', '50')
                            score = float(str(answer).strip())
                            score = max(0, min(100, score))
                            scores.append(score)
                            logging.info(f"{self.name} 候选图像 {i+1} 蕴含得分: {score}")
                        except (json.JSONDecodeError, ValueError) as e:
                            # 如果JSON解析失败，尝试直接提取数字
                            import re
                            numbers = re.findall(r'\d+(?:\.\d+)?', str(content))
                            if numbers:
                                score = float(numbers[0])
                                score = max(0, min(100, score))
                            else:
                                score = 50.0
                                logging.warning(f"无法解析蕴含得分，使用默认值50: {content}")
                            scores.append(score)
                            logging.info(f"{self.name} 候选图像 {i+1} 蕴含得分: {score}")
                    
                except Exception as e:
                    logging.warning(f"解析蕴含得分失败: {e}，使用默认值50")
                    scores.append(50.0)
                    
            except Exception as e:
                logging.error(f"计算候选图像 {i+1} 蕴含得分失败: {e}")
                scores.append(50.0)  # 默认中等得分
        
        logging.info(f"{self.name} 所有候选图像蕴含得分: {[f'{s:.1f}' for s in scores]}")
        return scores
    
    def guess_target_image_with_entailment(self, description: str, candidate_images: List[str]) -> str:
        """
        使用蕴含打分方法猜测目标图像（选择得分最高的候选）
        
        Args:
            description: 说书人的描述
            candidate_images: 候选图像列表
            
        Returns:
            猜测的目标图像路径
        """
        # 计算每个候选图像的蕴含得分
        scores = self.compute_entailment_scores(description, candidate_images)
        
        # 选择得分最高的候选图像
        max_score_index = scores.index(max(scores))
        selected_image = candidate_images[max_score_index]
        
        logging.info(f"{self.name} 蕴含打分选择: 候选{max_score_index+1} (得分{scores[max_score_index]:.1f}) - {selected_image}")
        return selected_image
