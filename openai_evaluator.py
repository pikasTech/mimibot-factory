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
    
    def evaluate(self, response, prompt=None, reference=None):
        """
        评估单个回复的质量
        
        Args:
            response: 要评估的AI回复
            prompt: 用户提问（可选）
            reference: 参考答案（可选）
            
        Returns:
            float: 0-100的评分
            dict: 详细评分数据
        """
        if not self.is_available():
            raise ValueError("OpenAI API密钥未设置")
        
        # 开始计时
        start_time = time.time()
        
        # 构建评估提示
        evaluation_prompt = f"""
        请评估以下AI回复的质量，并给出0-100分的评分。评分标准包括：
        1. 上下文连贯性 (0-33分)
        2. 逻辑性 (0-33分)
        3. 问答对应性 (0-34分)
        
        用户提问: {prompt or "未提供"}
        
        AI回复: {response}
        
        {f"参考答案: {reference}" if reference else ""}
        
        请以JSON格式返回评分结果，格式如下：
        {{
            "连贯性": [分数],
            "逻辑性": [分数],
            "对应性": [分数],
            "总分": [总分],
            "评价": "[简短评价]"
        }}
        仅返回JSON格式，不要有其他文字。
        """
        
        # 使用LangChain调用OpenAI API
        messages = [
            SystemMessage(content="你是一个专业的AI回复质量评估专家。你的任务是客观评价AI回复的质量。"),
            HumanMessage(content=evaluation_prompt)
        ]
        
        # 获取API响应
        response_message = self.llm.invoke(messages)
        response_text = response_message.content
        
        # 尝试提取JSON部分
        json_start = response_text.find("{")
        json_end = response_text.rfind("}") + 1
        
        if json_start >= 0 and json_end > json_start:
            json_str = response_text[json_start:json_end]
            score_data = json.loads(json_str)
            total_score = score_data.get("总分", 0)
            
            # 确保分数在0-100范围内
            total_score = max(0, min(100, total_score))
            
            # 计算耗时
            elapsed_time = time.time() - start_time
            print(f"单个评估耗时: {elapsed_time:.2f}秒")
            
            return total_score, score_data
        else:
            # 计算耗时（即使失败）
            elapsed_time = time.time() - start_time
            print(f"单个评估失败耗时: {elapsed_time:.2f}秒")
            
            raise ValueError(f"无法从回复中提取JSON：{response_text}")
    
    def evaluate_multiple(self, responses, prompts=None, references=None, simplified=True, max_retries=2, include_references=False):
        """
        一次API调用评估多个回复的质量
        
        Args:
            responses: 要评估的AI回复列表
            prompts: 用户提问列表（可选）
            references: 参考答案列表（可选）
            simplified: 是否使用简化格式返回结果（仅总分）
            max_retries: 最大重试次数
            include_references: 是否在返回结果中包含参考答案的评分
            
        Returns:
            list: 评分列表（0-100）
            list: 详细评分数据列表（如果simplified=False）或None（如果simplified=True）
        """
        if not self.is_available():
            raise ValueError("OpenAI API密钥未设置")
        
        if not responses:
            return [], [] if not simplified else None
            
        # 创建包含参考答案的回复列表并打乱顺序
        all_responses = responses.copy()
        all_prompts = prompts.copy() if prompts else None
        ref_indices = []  # 保存参考答案的索引位置
        
        if references:
            for i, ref in enumerate(references):
                if ref:  # 只添加非None的参考答案
                    all_responses.append(ref)
                    if all_prompts:
                        all_prompts.append(prompts[i] if i < len(prompts) else "未提供")
                    ref_indices.append(len(all_responses) - 1)  # 记录参考答案在列表中的位置
        
        # 打乱顺序处理
        indices = list(range(len(all_responses)))
        random.shuffle(indices)
        
        shuffled_responses = [all_responses[i] for i in indices]
        shuffled_prompts = [all_prompts[i] for i in indices] if all_prompts else None
        
        # 记录参考答案在打乱后的位置
        shuffled_ref_indices = [indices.index(idx) for idx in ref_indices]
        
        # 进行评估
        retry_count = 0
        last_error = None
        
        # 开始计时
        start_time = time.time()
        
        while retry_count <= max_retries:
            try:
                # 构建批量评估提示
                evaluation_items = []
                for i, response in enumerate(shuffled_responses):
                    prompt = shuffled_prompts[i] if shuffled_prompts and i < len(shuffled_prompts) else "未提供"
                    
                    item = f"""
                    回复 #{i+1}:
                    用户提问: {prompt}
                    AI回复: {response}
                    """
                    evaluation_items.append(item)
                
                evaluation_items_str = "\n".join(evaluation_items)
                
                if simplified:
                    # 简化版本 - 仅返回总分数组
                    evaluation_prompt = f"""
                    请评估以下 {len(shuffled_responses)} 个AI回复的质量，对每个回复给出0-100分的评分。评分标准包括：
                    1. 上下文连贯性 (0-33分)
                    2. 逻辑性 (0-33分)
                    3. 问答对应性 (0-34分)
                    
                    {evaluation_items_str}
                    
                    请只返回一个总分数组，格式为JSON数组，如 [95, 78, 65]。
                    数组中的每个数字对应每个回复的总分（0-100之间的整数）。
                    不要返回任何其他文字或解释，只返回分数数组。
                    每个回复的分数不能相同。
                    不要粗略地返回5或者10的倍数，具体到个位数。
                    """
                else:
                    evaluation_prompt_part1 = f"""
                    请评估以下 {len(responses)} 个AI回复的质量，对每个回复给出0-100分的评分。评分标准包括：
                    1. 上下文连贯性 (0-33分)
                    2. 逻辑性 (0-33分)
                    3. 问答对应性 (0-34分)
                    
                    {evaluation_items_str}
                    """
                    
                    evaluation_prompt_part2 = """
                    请以JSON格式返回评分结果，格式如下：
                    {
                        "评估结果": [
                            {
                                "回复编号": 1,
                                "连贯性": [分数],
                                "逻辑性": [分数],
                                "对应性": [分数],
                                "总分": [总分],
                                "评价": "[简短评价]"
                            },
                            {
                                "回复编号": 2,
                                ...
                            },
                            ...
                        ]
                    }
                    
                    仅返回JSON格式，不要有其他文字。确保每个回复都有对应的评分结果。
                    """
                    
                    evaluation_prompt = evaluation_prompt_part1 + evaluation_prompt_part2

                # 使用LangChain调用OpenAI API
                messages = [
                    SystemMessage(content="你是一个专业的AI回复质量评估专家。你的任务是客观、公正地评价AI回复的质量。"),
                    HumanMessage(content=evaluation_prompt)
                ]
                
                # 获取API响应
                response_message = self.llm.invoke(messages)
                response_text = response_message.content
                
                if simplified:
                    # 解析简化版本的响应（仅总分数组）
                    try:
                        # 1. 尝试直接解析完整JSON
                        response_text = response_text.strip()
                        # 移除可能的代码块标记
                        if response_text.startswith("```") and "```" in response_text:
                            response_text = re.sub(r"```(?:json)?", "", response_text).strip()
                            response_text = response_text.replace("```", "").strip()
                        
                        # 2. 尝试解析为JSON数组
                        if response_text.startswith("[") and response_text.endswith("]"):
                            scores = json.loads(response_text)
                            # 确保分数在0-100范围内
                            scores = [max(0, min(100, float(score))) for score in scores]
                            scores = [round(score, 2) for score in scores]
                            
                            # 恢复原始顺序的分数
                            restored_scores = [0] * len(indices)
                            for i, idx in enumerate(indices):
                                restored_scores[idx] = scores[i]
                            
                            # 计算参考答案的平均分数
                            ref_scores = [restored_scores[idx] for idx in ref_indices if idx < len(restored_scores)]
                            avg_ref_score = sum(ref_scores) / len(ref_scores) if ref_scores else 0
                            
                            # 为每个原始回复减去参考答案的分数
                            normalized_scores = []
                            for i in range(len(responses)):
                                norm_score = max(0, restored_scores[i] - avg_ref_score)
                                normalized_scores.append(round(norm_score, 2))
                            
                            # 计算并显示耗时
                            elapsed_time = time.time() - start_time
                            print(f"批量评估({len(shuffled_responses)}个回复)耗时: {elapsed_time:.2f}秒, 平均每个回复: {elapsed_time/len(shuffled_responses):.2f}秒")
                            
                            if include_references:
                                return normalized_scores, ref_scores
                            else:
                                return normalized_scores, None
                        
                        # 3. 如果不是直接的JSON数组，尝试提取
                        array_pattern = r'\[(\s*\d+\s*(?:,\s*\d+\s*)*)\]'
                        array_match = re.search(array_pattern, response_text)
                        if array_match:
                            array_str = "[" + array_match.group(1) + "]"
                            scores = json.loads(array_str)
                            scores = [max(0, min(100, float(score))) for score in scores]
                            scores = [round(score, 2) for score in scores]
                            
                            # 恢复原始顺序的分数
                            restored_scores = [0] * len(indices)
                            for i, idx in enumerate(indices):
                                restored_scores[idx] = scores[i]
                            
                            # 计算参考答案的平均分数
                            ref_scores = [restored_scores[idx] for idx in ref_indices if idx < len(restored_scores)]
                            avg_ref_score = sum(ref_scores) / len(ref_scores) if ref_scores else 0
                            
                            # 为每个原始回复减去参考答案的分数
                            normalized_scores = []
                            for i in range(len(responses)):
                                norm_score = max(0, restored_scores[i] - avg_ref_score)
                                normalized_scores.append(round(norm_score, 2))
                            
                            # 计算并显示耗时
                            elapsed_time = time.time() - start_time
                            print(f"批量评估({len(shuffled_responses)}个回复)耗时: {elapsed_time:.2f}秒, 平均每个回复: {elapsed_time/len(shuffled_responses):.2f}秒")
                            
                            if include_references:
                                return normalized_scores, ref_scores
                            else:
                                return normalized_scores, None
                        
                        # 4. 如果无法提取数组，尝试直接提取数字
                        numbers = re.findall(r'\b\d+(?:\.\d+)?\b', response_text)
                        if len(numbers) >= len(responses):
                            scores = [float(numbers[i]) for i in range(len(responses))]
                            scores = [max(0, min(100, score)) for score in scores]
                            scores = [round(score, 2) for score in scores]
                            
                            # 恢复原始顺序的分数
                            restored_scores = [0] * len(indices)
                            for i, idx in enumerate(indices):
                                restored_scores[idx] = scores[i]
                            
                            # 计算参考答案的平均分数
                            ref_scores = [restored_scores[idx] for idx in ref_indices if idx < len(restored_scores)]
                            avg_ref_score = sum(ref_scores) / len(ref_scores) if ref_scores else 0
                            
                            # 为每个原始回复减去参考答案的分数
                            normalized_scores = []
                            for i in range(len(responses)):
                                norm_score = max(0, restored_scores[i] - avg_ref_score)
                                normalized_scores.append(round(norm_score, 2))
                            
                            # 计算并显示耗时
                            elapsed_time = time.time() - start_time
                            print(f"批量评估({len(shuffled_responses)}个回复)耗时: {elapsed_time:.2f}秒, 平均每个回复: {elapsed_time/len(shuffled_responses):.2f}秒")
                            
                            if include_references:
                                return normalized_scores, ref_scores
                            else:
                                return normalized_scores, None
                        
                        # 5. 如果所有方法都失败，打印错误信息并重试
                        print(f"无法解析简化JSON结果，尝试重试 (重试 {retry_count+1}/{max_retries})。")
                        print(f"原始响应: {response_text}")
                        raise ValueError(f"无法解析简化的响应格式: {response_text}")
                        
                    except Exception as e:
                        last_error = e
                        print(f"解析失败: {str(e)}")
                        print(f"原始响应内容: {response_text}")
                        retry_count += 1
                        if retry_count <= max_retries:
                            print(f"重试中 ({retry_count}/{max_retries})...")
                            time.sleep(1)  # 稍微等待再重试
                            continue
                        else:
                            # 尝试转为非简化模式处理
                            print("简化模式失败，切换到详细模式...")
                            return self.evaluate_multiple(responses, prompts, references, simplified=False)
                else:
                    # 解析详细版本的响应
                    try:
                        json_start = response_text.find("{")
                        json_end = response_text.rfind("}") + 1
                        
                        if json_start >= 0 and json_end > json_start:
                            json_str = response_text[json_start:json_end]
                            result_data = json.loads(json_str)
                            evaluation_results = result_data.get("评估结果", [])
                            
                            scores = []
                            details = []
                            
                            for i, eval_item in enumerate(evaluation_results):
                                total_score = eval_item.get("总分", 0)
                                # 确保分数在0-100范围内
                                total_score = max(0, min(100, float(total_score)))
                                scores.append(round(total_score, 2))
                                details.append(eval_item)
                            
                            # 计算并显示耗时
                            elapsed_time = time.time() - start_time
                            print(f"详细批量评估({len(responses)}个回复)耗时: {elapsed_time:.2f}秒, 平均每个回复: {elapsed_time/len(responses):.2f}秒")
                            
                            return scores, details
                        else:
                            # 如果找不到JSON对象，尝试提取数字作为分数
                            print("无法从详细响应中提取JSON对象，尝试提取数字作为分数...")
                            numbers = re.findall(r'\b\d+(?:\.\d+)?\b', response_text)
                            if len(numbers) >= len(responses):
                                scores = [float(numbers[i]) for i in range(len(responses))]
                                scores = [max(0, min(100, score)) for score in scores]
                                scores = [round(score, 2) for score in scores]
                                return scores, [{"总分": score} for score in scores]
                            else:
                                raise ValueError(f"无法从回复中提取足够的分数：{response_text}")
                    except Exception as e:
                        last_error = e
                        print(f"详细模式解析失败: {str(e)}")
                        print(f"原始响应内容: {response_text}")
                        retry_count += 1
                        if retry_count <= max_retries:
                            print(f"重试中 ({retry_count}/{max_retries})...")
                            time.sleep(1)
                            continue
                        else:
                            raise ValueError(f"详细模式解析失败，已达最大重试次数: {str(e)}")
            
            except Exception as e:
                last_error = e
                retry_count += 1
                if retry_count <= max_retries:
                    print(f"API调用失败: {str(e)}，尝试重试 ({retry_count}/{max_retries})...")
                    time.sleep(2)  # 两秒后重试
                else:
                    break
        
        # 计算耗时（即使失败）
        elapsed_time = time.time() - start_time
        print(f"批量评估失败耗时: {elapsed_time:.2f}秒")
        
        # 如果所有重试都失败，抛出最后一个错误
        if last_error:
            raise ValueError(f"评估多个回复失败，已达最大重试次数: {str(last_error)}")
        
        # 应该不会到达这里，但以防万一返回空结果
        return [], [] if not simplified else None
    
    def evaluate_batch(self, responses, prompts=None, references=None, simplified=True, max_retries=2):
        """
        批量评估多个回复的质量，自动分批处理
        
        Args:
            responses: 要评估的AI回复列表
            prompts: 用户提问列表（可选）
            references: 参考答案列表（可选）
            simplified: 是否使用简化格式返回结果
            max_retries: 每批次的最大重试次数
            
        Returns:
            list: 评分列表（0-100）
            list: 详细评分数据列表
        """
        if not responses:
            return [], [] if not simplified else None
        
        # 确保每个回复都有对应的参考答案
        if references:
            assert len(responses) == len(references), "回复数量必须与参考答案数量相同"
        else:
            references = [None] * len(responses)
        
        all_scores = []
        all_details = []
        
        # 开始计时
        total_start_time = time.time()
        
        # 按批次处理
        for i in range(0, len(responses), self.batch_size):
            batch_start_time = time.time()
            
            batch_responses = responses[i:i+self.batch_size]
            batch_prompts = prompts[i:i+self.batch_size] if prompts else None
            batch_references = references[i:i+self.batch_size] if references else None
            
            batch_retry_count = 0
            batch_success = False
            
            # 对每个批次进行重试
            while batch_retry_count <= max_retries and not batch_success:
                try:
                    print(f"评估批次 {i//self.batch_size + 1}, 包含 {len(batch_responses)} 个回复...")
                    batch_scores, batch_details = self.evaluate_multiple(
                        batch_responses, batch_prompts, batch_references, 
                        simplified=simplified, max_retries=1  # 内部已有重试
                    )
                    
                    # 检查结果是否正确
                    if batch_scores and len(batch_scores) == len(batch_responses):
                        batch_success = True
                        
                        # 计算批次耗时
                        batch_elapsed_time = time.time() - batch_start_time
                        print(f"批次 {i//self.batch_size + 1} 评估耗时: {batch_elapsed_time:.2f}秒, 平均每个回复: {batch_elapsed_time/len(batch_responses):.2f}秒")
                        
                        # 打印结果
                        for j, score in enumerate(batch_scores):
                            item_index = i + j
                            print(f"OpenAI评分 - 回复 #{item_index+1}: {score}")
                            if not simplified and batch_details and j < len(batch_details):
                                detail = batch_details[j]
                                if detail:
                                    print(f"评价: {detail.get('评价', '无评价')}")
                        
                        all_scores.extend(batch_scores)
                        if not simplified and batch_details:
                            all_details.extend(batch_details)
                    else:
                        raise ValueError(f"批次评估返回的分数数量({len(batch_scores)})与回复数量({len(batch_responses)})不匹配")
                
                except Exception as e:
                    print(f"评估批次 {i//self.batch_size + 1} 时出错 (尝试 {batch_retry_count+1}/{max_retries+1}): {str(e)}")
                    batch_retry_count += 1
                    
                    if batch_retry_count > max_retries:
                        print("批次评估达到最大重试次数，回退到单个评估...")
                        # 如果批次评估失败，回退到单个评估
                        for j, response in enumerate(batch_responses):
                            item_index = i + j
                            prompt = batch_prompts[j] if batch_prompts else None
                            reference = batch_references[j] if batch_references else None
                            
                            try:
                                # 将reference作为另一个回复一起评估
                                if reference:
                                    response_list = [response, reference]
                                    prompt_list = [prompt, prompt] if prompt else None
                                    scores, _ = self.evaluate_multiple(response_list, prompt_list, simplified=True)
                                    # 减去参考答案的分数
                                    score = max(0, scores[0] - scores[1])
                                else:
                                    score, detail = self.evaluate(response, prompt, reference)
                                    
                                all_scores.append(score)
                                if not simplified:
                                    all_details.append({"总分": score, "评价": "单独评估结果"})
                                print(f"单个评估 - 回复 #{item_index+1}: {score}")
                            except Exception as e:
                                print(f"单个评估回复 #{item_index+1} 时出错: {str(e)}")
                                # 使用默认分数
                                default_score = 50.0
                                all_scores.append(default_score)
                                if not simplified:
                                    all_details.append({"总分": default_score, "评价": "评估失败，使用默认分数"})
                                print(f"使用默认分数 {default_score} 用于回复 #{item_index+1}")
                    else:
                        # 等待后重试
                        print(f"等待 2 秒后重试...")
                        time.sleep(2)
        
        # 计算总耗时
        total_elapsed_time = time.time() - total_start_time
        print(f"\n总评估耗时: {total_elapsed_time:.2f}秒, 共 {len(responses)} 个回复, 平均每个回复: {total_elapsed_time/len(responses):.2f}秒")
        
        return all_scores, all_details if not simplified else None
    
    def test_consistency(self, responses, prompts=None, references=None, num_tests=3, simplified=True):
        """
        测试评分的一致性，通过打乱顺序多次评估
        
        Args:
            responses: 要评估的回复列表
            prompts: 提问列表（可选）
            references: 参考答案列表（可选）
            num_tests: 测试次数
            simplified: 是否使用简化格式
            
        Returns:
            list: 多次评分结果列表
            float: 评分一致性分数(0-1)
        """
        if len(responses) < 2:
            print("需要至少2个回复来测试一致性")
            return [[], []], 1.0
        
        all_results = []
        original_indices = list(range(len(responses)))
        
        print(f"\n开始一致性测试，将进行 {num_tests} 次随机顺序评估...")
        
        # 保存原始顺序的结果
        print("\n测试 #0: 原始顺序")
        original_scores, _ = self.evaluate_batch(responses, prompts, references, simplified=simplified)
        all_results.append(original_scores)
        
        # 进行多次打乱顺序的测试
        for test_num in range(1, num_tests+1):
            # 打乱索引
            shuffled_indices = original_indices.copy()
            random.shuffle(shuffled_indices)
            
            # 按打乱的顺序重排
            shuffled_responses = [responses[i] for i in shuffled_indices]
            shuffled_prompts = [prompts[i] for i in shuffled_indices] if prompts else None
            shuffled_references = [references[i] for i in shuffled_indices] if references else None
            
            print(f"\n测试 #{test_num}: 随机顺序")
            print(f"打乱后的顺序: {shuffled_indices}")
            
            # 评估打乱后的数据
            shuffled_scores, _ = self.evaluate_batch(
                shuffled_responses, shuffled_prompts, shuffled_references, simplified=simplified
            )
            
            # 恢复原始顺序
            restored_scores = [None] * len(shuffled_scores)
            for i, orig_idx in enumerate(shuffled_indices):
                restored_scores[orig_idx] = shuffled_scores[i]
            
            all_results.append(restored_scores)
            
            # 比较恢复后的分数与原始分数
            diffs = [abs(original_scores[i] - restored_scores[i]) for i in range(len(original_scores))]
            avg_diff = sum(diffs) / len(diffs)
            max_diff = max(diffs)
            
            print(f"恢复顺序后的分数: {restored_scores}")
            print(f"与原始分数的平均差异: {avg_diff:.2f}, 最大差异: {max_diff:.2f}")
        
        # 计算总体一致性
        all_scores = np.array(all_results)
        std_per_item = np.std(all_scores, axis=0)
        mean_std = np.mean(std_per_item)
        max_std = np.max(std_per_item)
        
        # 一致性分数：1 - 标准差/满分
        consistency = 1 - (mean_std / 100)

        # Calculate ranking consistency based on rank correlation
        rank_consistency = 0.0

        if len(all_results) > 1:
            # Calculate Spearman rank correlation for each pair of evaluations
            correlations = []
            
            for i in range(len(all_results)):
                for j in range(i+1, len(all_results)):
                    scores_i = np.array(all_results[i])
                    scores_j = np.array(all_results[j])
                    
                    # Convert to ranks (argsort of argsort gives ranks)
                    # Using negative scores so higher scores get lower ranks
                    ranks_i = np.argsort(np.argsort(-scores_i))
                    ranks_j = np.argsort(np.argsort(-scores_j))
                    
                    # Calculate Pearson correlation between ranks
                    mean_i = np.mean(ranks_i)
                    mean_j = np.mean(ranks_j)
                    std_i = np.std(ranks_i)
                    std_j = np.std(ranks_j)
                    
                    if std_i > 0 and std_j > 0:  # Avoid division by zero
                        corr = np.mean((ranks_i - mean_i) * (ranks_j - mean_j)) / (std_i * std_j)
                        correlations.append(corr)
            
            # Calculate average correlation (ranges from -1 to 1)
            if correlations:
                rank_consistency = np.mean(correlations)
                # Scale to 0-1 range for easier interpretation
                rank_consistency = (rank_consistency + 1) / 2
        
        print("\n一致性测试结果:")
        print(f"所有测试的评分: {all_results}")
        print(f"每项的标准差: {std_per_item}")
        print(f"平均标准差: {mean_std:.2f}, 最大标准差: {max_std:.2f}")
        print(f"一致性分数: {consistency:.4f} (0-1，越接近1表示一致性越高)")
        print(f"排名一致性: {rank_consistency:.4f} (0-1，越接近1表示排名越一致)")
        
        return all_results, consistency

if __name__ == "__main__":
    """
    简单的测试代码，通过实际调用API来测试评估器功能
    """
    # 测试数据 - 增加更多示例来测试8个一批
    test_cases = [
        {
            "prompt": "解释一下量子计算的基本原理",
            "response": "量子计算利用量子力学原理，如叠加和纠缠来处理信息。与传统计算机使用位不同，量子计算机使用量子位（qubit）。每个量子位可以同时处于多个状态，这赋予量子计算机处理大量可能性的能力。通过量子门操作，可以实现复杂的计算任务。目前，量子计算仍处于发展阶段，面临着量子退相干等挑战。",
            "reference": None
        },
        {
            "prompt": "如何制作意大利面？",
            "response": "买一包面条煮一下就行。",
            "reference": "准备意大利面条、盐和水。将水烧开，加入盐。放入面条，根据包装说明煮8-12分钟至软硬适中。沥干水分，加入你喜欢的酱料搅拌均匀。可以添加橄榄油、帕马森奶酪、香草等增加风味。"
        },
        {
            "prompt": "Python中如何创建一个函数？",
            "response": "在Python中，使用def关键字创建函数。基本语法是：\n\ndef 函数名(参数1, 参数2, ...):\n    函数体\n    return 返回值\n\n例如：\n\ndef add(a, b):\n    return a + b\n\nresult = add(5, 3)  # 调用函数，结果为8",
            "reference": None
        },
        {
            "prompt": "什么是机器学习？",
            "response": "机器学习是人工智能的一个分支，它使计算机能够从数据中学习，而无需明确编程。",
            "reference": "机器学习是人工智能的一个分支，它使计算机系统能够从数据中学习并改进经验，而无需明确编程。它专注于开发能够访问数据并使用它来自我学习的算法。机器学习算法通过构建模型来识别数据中的模式，然后使用这些模型对新数据进行预测或决策。"
        },
        {
            "prompt": "推荐一部经典电影",
            "response": "《肖申克的救赎》是一部非常经典的电影，讲述了银行家安迪被错误判刑入狱后，在监狱中保持希望并最终获得自由的故事。",
            "reference": None
        },
        {
            "prompt": "解释下什么是区块链技术",
            "response": "区块链就是比特币。",
            "reference": "区块链是一种分布式账本技术，它通过一系列连接的区块存储信息，每个区块包含交易数据和前一个区块的哈希值。这种设计使得区块链具有不可篡改性和去中心化的特点。它不仅用于加密货币如比特币，还可应用于供应链追踪、智能合约等多个领域。"
        },
        {
            "prompt": "如何改善睡眠质量？",
            "response": "改善睡眠质量的方法：1. 保持规律的睡眠时间表 2. 创造舒适的睡眠环境，如适宜的温度和光线 3. 睡前避免使用电子设备 4. 避免睡前摄入咖啡因和酒精 5. 适度运动，但避免睡前剧烈运动 6. 睡前放松活动，如阅读或冥想 7. 舒适的床垫和枕头 8. 如持续失眠，考虑咨询医生",
            "reference": None
        },
        {
            "prompt": "介绍一下太阳系",
            "response": "太阳系是恒星太阳及其周围天体的集合，包括八大行星（按距离顺序：水星、金星、地球、火星、木星、土星、天王星和海王星）、矮行星、卫星、小行星、彗星等。太阳占据太阳系质量的99.86%，其引力将整个系统束缚在一起。地球是目前已知唯一存在生命的行星。",
            "reference": None
        },
        {
            "prompt": "什么是可持续发展？",
            "response": "可持续发展是指能够满足当代人需求而不损害后代人满足其需求能力的发展模式。它强调环境保护、社会公平和经济增长的平衡。可持续发展目标包括减少贫困、促进教育、性别平等、清洁能源、气候行动等多个方面。",
            "reference": None
        },
        {
            "prompt": "介绍一下人工智能的伦理问题",
            "response": "人工智能伦理涉及的问题包括：隐私和数据保护、算法偏见和歧视、自动化导致的就业影响、自主武器系统的潜在危害、透明度和可解释性、责任归属等。随着AI技术的发展，建立适当的伦理框架和监管体系变得越来越重要。",
            "reference": None
        }
    ]
    
    print("开始测试OpenAI评估器...\n")
    
    try:
        # 创建评估器实例 - 改为8个一批
        evaluator = OpenAIEvaluator(batch_size=8)  # 设置批次大小为8进行测试
        
        # 检查API密钥和基础URL设置
        print(f"使用基础URL: {evaluator.base_url}")
        print(f"使用模型: {evaluator.model}")
        print(f"API密钥可用: {evaluator.is_available()}\n")
        
        if not evaluator.is_available():
            print("错误: API密钥未设置，无法进行测试")
            sys.exit(1)
        
        # 只保留单个回复评估的简化测试
        print("测试单个回复评估:")
        case = test_cases[0]
        print(f"提问: {case['prompt']}")
        print(f"回复: {case['response']}")
        
        score, details = evaluator.evaluate(
            case['response'], 
            case['prompt'], 
            case['reference']
        )
        
        print("\n评分结果:")
        print(f"总分: {score}/100")
        print(f"连贯性: {details.get('连贯性', 'N/A')}")
        print(f"逻辑性: {details.get('逻辑性', 'N/A')}")
        print(f"对应性: {details.get('对应性', 'N/A')}")
        print(f"评价: {details.get('评价', 'N/A')}")
        print("\n" + "-"*50 + "\n")
        
        # 准备批量测试数据
        print("批量评估测试:")
        responses = [case['response'] for case in test_cases]
        prompts = [case['prompt'] for case in test_cases]
        references = [case['reference'] for case in test_cases]
        
        # 测试批量评估
        print("\n测试一次评估多个回复:")
        # 只用前8个测试一次性评估
        test_batch_size = min(8, len(responses))
        batch_scores, _ = evaluator.evaluate_multiple(
            responses[:test_batch_size], 
            prompts[:test_batch_size], 
            references[:test_batch_size]
        )
        print(f"批量评分结果: {batch_scores}")
        
        print("\n测试分批评估(每批8个):")
        all_scores, _ = evaluator.evaluate_batch(responses, prompts, references)
        print(f"全部评分结果: {all_scores}\n")
        
        # 测试一致性（打乱顺序测试）
        print("\n开始一致性测试（多次打乱顺序评估）:")
        # 使用一部分数据进行一致性测试，避免过长
        test_size = min(8, len(responses))
        test_responses = responses[:test_size]
        test_prompts = prompts[:test_size]
        test_references = references[:test_size]
        
        all_test_results, consistency = evaluator.test_consistency(
            test_responses, test_prompts, test_references, num_tests=3
        )
        
        print("\n测试完成!")
    
    except Exception as e:
        print(f"测试过程中出错: {str(e)}")
        sys.exit(1)
