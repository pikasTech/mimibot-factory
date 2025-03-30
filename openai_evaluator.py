"""
OpenAI API 评估器
用于评估模型生成的回复质量
"""
import os
import json
import sys
import re
import time
import random
import numpy as np
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from config import OPENAI_API_KEY, OPENAI_BASE_URL, MODEL_NAME, TEMPERATURE, MAX_TOKENS

class OpenAIEvaluator:
    """
    使用OpenAI API评估回复质量的评估器类
    """
    def __init__(self, api_key=None, model=None, batch_size=3):
        """
        初始化OpenAI评估器
        
        Args:
            api_key: OpenAI API密钥，如果为None则使用配置文件中的密钥
            model: 使用的OpenAI模型名称，如果为None则使用配置文件中的模型
            batch_size: 批量评估时的批次大小
        """
        # 优先使用传入参数，其次是配置文件中的变量
        self.api_key = api_key or OPENAI_API_KEY
        self.model = model or MODEL_NAME
        self.base_url = OPENAI_BASE_URL
        self.batch_size = batch_size
        
        # 初始化LangChain的ChatOpenAI客户端
        self.llm = ChatOpenAI(
            openai_api_key=self.api_key,
            base_url=self.base_url,
            model=self.model,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        
    def is_available(self):
        """检查评估器是否可用（API密钥是否存在）"""
        return bool(self.api_key)
    
    def _evaluate_responses_for_prompt(self, prompt, responses, reference_idx=-1, max_retries=2):
        """
        评估同一个prompt的多个回复，其中一个可能是参考答案
        
        Args:
            prompt: 用户提问
            responses: 多个回复的列表，包括可能的参考答案
            reference_idx: 参考答案在responses中的索引，默认为-1（最后一个）
            max_retries: 最大重试次数
            
        Returns:
            list: 相对于参考答案的评分列表（0-100）
            list: 原始评分列表（包含参考答案的分数）
        """
        if not self.is_available():
            raise ValueError("OpenAI API密钥未设置")
        
        if not responses:
            return [], [], None
        
        # 确保reference_idx在有效范围内
        if reference_idx < -1 or reference_idx >= len(responses):
            reference_idx = -1  # 默认使用最后一个作为参考
        
        # 真实的reference_idx（处理负索引）
        ref_idx = reference_idx if reference_idx >= 0 else len(responses) + reference_idx
        
        # 打乱顺序处理
        indices = list(range(len(responses)))
        random.shuffle(indices)
        
        # 记录参考答案在打乱后的位置
        shuffled_ref_idx = indices.index(ref_idx)
        shuffled_responses = [responses[i] for i in indices]
        
        # 所有回复都使用相同的prompt
        prompts = [prompt] * len(responses)
        shuffled_prompts = [prompts[i] for i in indices]
        
        # 开始计时
        start_time = time.time()
        
        retry_count = 0
        last_error = None
        
        while retry_count <= max_retries:
            try:
                # 构建批量评估提示
                evaluation_items = []
                for i, response in enumerate(shuffled_responses):
                    item = f"""
                    回复 #{i+1}:
                    用户提问: {shuffled_prompts[i]}
                    AI回复: {response}
                    """
                    evaluation_items.append(item)
                
                evaluation_items_str = "\n".join(evaluation_items)
                
                # 简化版本 - 仅返回总分数组
                evaluation_prompt = f"""
                请评估以下 {len(shuffled_responses)} 个对同一个问题的AI回复质量，对每个回复给出0-100分的评分。评分标准包括以下，重要性从高到低：
                - 奖励切中最新问题的回答
                - 惩罚胡言乱语，难以理解，存在乱码、不完整的回答
                - 惩罚超过 20 字的回答
                - 鼓励和群友的对话相似的风格
                - 惩罚看起来是AI生成的回答
                - 鼓励言之有物的，具体的，有细节的，有信息量的回答
                - 惩罚宽泛的，模糊的，空洞的回答
                - 惩罚有代码块、表格等格式的回答
                - 鼓励提到了群友名字的回答，但是不要代替群友发言，例如 <群友id>:XXX
                - 惩罚带有口头禅的回答，即在多个回答中出现频率过高的语气词，比如“确实”、“没错”、“好啦~”等
                - 惩罚带有颜文字、表情符号等的回答
                - 惩罚过于客气，过于正式的回答，鼓励更接地气的表达
                - 惩罚重复性回答
                - 惩罚过于简单、无意义的回答
                - 惩罚过于复杂、冗长的回答
                - 鼓励日常口语化的表达，避免过于正式或生硬
                - 鼓励像是可爱少女的回答，避免过于严肃
                - 鼓励像是损友互相调侃的回答，锐评、讽刺、挖苦都可以
                - 鼓励回答中包含一些个人观点和情感色彩
                - 鼓励有想象力的回答
                - 鼓励用隐喻、讥讽等手法回答
                - 鼓励用简单的语言回答，避免过于专业术语
                - 鼓励有创意的回答
                    
                用户提问: {prompt}
                    
                {evaluation_items_str}
                    
                请只返回一个总分数组，格式为JSON数组，如 [95, 78, 65]。
                数组中的每个数字对应每个回复的总分（0-100之间的整数）。
                不要返回任何其他文字或解释，只返回分数数组。
                细致评估每个回复，即使回复内容相似，也要根据质量差异给出不同的分数。
                不要粗略地返回5或者10的倍数，具体到个位数。
                """

                # 使用LangChain调用OpenAI API
                messages = [
                    SystemMessage(content="你是一个专业的AI回复质量评估专家。你的任务是客观、公正地评价AI回复的质量。"),
                    HumanMessage(content=evaluation_prompt)
                ]
                
                # 获取API响应
                response_message = self.llm.invoke(messages)
                response_text = response_message.content
                
                # 解析API响应
                # 尝试解析简化版本的响应（仅总分数组）
                scores = self._parse_simplified_scores(response_text)
                    
                if scores and len(scores) == len(shuffled_responses):
                    # 恢复原始顺序的分数
                    restored_scores = [0] * len(indices)
                    for i, idx in enumerate(indices):
                        restored_scores[idx] = scores[i]
                        
                    # 获取参考答案的分数
                    reference_score = restored_scores[ref_idx]
                    # print("参考答案分数:", reference_score)
                        
                    # 计算相对于参考答案的分数
                    relative_scores = []
                    for i, score in enumerate(restored_scores):
                        if i != ref_idx:  # 跳过参考答案本身
                            rel_score = score - reference_score
                            relative_scores.append(round(rel_score, 2))
                        
                    # 计算并显示耗时
                    elapsed_time = time.time() - start_time
                    # print(f"多回复评估({len(responses)}个回复)耗时: {elapsed_time:.2f}秒, 平均每个回复: {elapsed_time/len(responses):.2f}秒")
                        
                    return relative_scores, restored_scores, None
                
            except Exception as e:
                last_error = e
                retry_count += 1
                if retry_count <= max_retries:
                    print(f"API调用失败: {str(e)}，尝试重试 ({retry_count}/{max_retries})...")
                    time.sleep(2)  # 两秒后重试
                else:
                    break
        
        # 如果所有重试都失败，抛出最后一个错误
        if last_error:
            raise ValueError(f"评估多个回复失败，已达最大重试次数: {str(last_error)}")
        
        # 应该不会到达这里，但以防万一返回空结果
        return [], [], None
    

    def evaluate_responses(self, prompts, responses, reference=None, max_retries=2):
        if reference is not None:
            responses.append(reference)
            referece_idx = len(responses) - 1
        else:
            referece_idx = -1
        return self._evaluate_responses_for_prompt(prompts, responses, referece_idx, max_retries)

    
    def _parse_simplified_scores(self, response_text):
        """解析简化响应中的分数"""
        response_text = response_text.strip()
        
        # 移除可能的代码块标记
        if response_text.startswith("```") and "```" in response_text:
            response_text = re.sub(r"```(?:json)?", "", response_text).strip()
            response_text = response_text.replace("```", "").strip()
        
        # 尝试解析为JSON数组
        if response_text.startswith("[") and response_text.endswith("]"):
            scores = json.loads(response_text)
            scores = [round(score, 2) for score in scores]
            return scores
        
        # 如果不是直接的JSON数组，尝试提取
        array_pattern = r'\[(\s*\d+\s*(?:,\s*\d+\s*)*)\]'
        array_match = re.search(array_pattern, response_text)
        if array_match:
            array_str = "[" + array_match.group(1) + "]"
            scores = json.loads(array_str)
            scores = [round(score, 2) for score in scores]
            return scores
        
        # 如果无法提取数组，尝试直接提取数字
        numbers = re.findall(r'\b\d+(?:\.\d+)?\b', response_text)
        if numbers:
            scores = [float(num) for num in numbers]
            scores = [round(score, 2) for score in scores]
            return scores
        
        return None
    

if __name__ == "__main__":
    """
    简单的测试代码，通过实际调用API来测试评估器功能
    """
    
    print("开始测试OpenAI评估器...\n")
    
    # 添加同一prompt多个回复的测试
    print("\n测试同一prompt多个回复的评估:")
    prompt = "解释量子计算的基本原理"
    responses = [
        "量子计算机使用量子位来计算，跟普通电脑不一样。",
        "量子计算是一种利用量子力学现象进行计算的技术。与经典计算中的位不同，量子计算使用量子位（qubit），它可以同时处于多个状态，这种特性称为量子叠加。多个量子位之间可以形成量子纠缠，使得量子计算机能够同时处理大量信息。量子算法通过量子干涉操纵量子态，有潜力解决经典计算机难以处理的问题，如大数分解和量子模拟。当前量子计算面临的主要挑战是量子退相干和量子纠错。",
        "量子计算是基于量子力学原理的计算方法，它使用量子比特代替传统的二进制位，能够进行并行计算。"
    ]
    reference="量子计算利用量子力学原理，如叠加和纠缠来处理信息。与传统计算机使用位不同，量子计算机使用量子位（qubit）。每个量子位可以同时处于多个状态，这赋予量子计算机处理大量可能性的能力。",
    # 为 evaluate_responses 编写测试 
    evaluator = OpenAIEvaluator()
    relative_scores, total_scores, detailed_results = evaluator.evaluate_responses(prompt, responses, reference=reference)
    print("相对分数:", relative_scores)
    print("总分数:", total_scores)


