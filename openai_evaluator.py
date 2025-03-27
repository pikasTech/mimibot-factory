"""
OpenAI API 评估器
用于评估模型生成的回复质量
"""
import os
import json
import sys
import re
import time
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
            
            return total_score, score_data
        else:
            raise ValueError(f"无法从回复中提取JSON：{response_text}")
    
    def evaluate_multiple(self, responses, prompts=None, references=None, simplified=True, max_retries=2):
        """
        一次API调用评估多个回复的质量
        
        Args:
            responses: 要评估的AI回复列表
            prompts: 用户提问列表（可选）
            references: 参考答案列表（可选）
            simplified: 是否使用简化格式返回结果（仅总分）
            max_retries: 最大重试次数
            
        Returns:
            list: 评分列表（0-100）
            list: 详细评分数据列表（如果simplified=False）或None（如果simplified=True）
        """
        if not self.is_available():
            raise ValueError("OpenAI API密钥未设置")
        
        if not responses:
            return [], [] if not simplified else None
        
        retry_count = 0
        last_error = None
        
        while retry_count <= max_retries:
            try:
                # 构建批量评估提示
                evaluation_items = []
                for i, response in enumerate(responses):
                    prompt = prompts[i] if prompts and i < len(prompts) else "未提供"
                    reference = references[i] if references and i < len(references) else None
                    
                    item = f"""
                    回复 #{i+1}:
                    用户提问: {prompt}
                    AI回复: {response}
                    {f"参考答案: {reference}" if reference else ""}
                    """
                    evaluation_items.append(item)
                
                evaluation_items_str = "\n".join(evaluation_items)
                
                if simplified:
                    # 简化版本 - 仅返回总分数组
                    evaluation_prompt = f"""
                    请评估以下 {len(responses)} 个AI回复的质量，对每个回复给出0-100分的评分。评分标准包括：
                    1. 上下文连贯性 (0-33分)
                    2. 逻辑性 (0-33分)
                    3. 问答对应性 (0-34分)
                    
                    {evaluation_items_str}
                    
                    请只返回一个总分数组，格式为JSON数组，如 [95, 78, 65]。
                    数组中的每个数字对应每个回复的总分（0-100之间的整数）。
                    不要返回任何其他文字或解释，只返回分数数组。
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
                            return scores, None
                        
                        # 3. 如果不是直接的JSON数组，尝试提取
                        array_pattern = r'\[(\s*\d+\s*(?:,\s*\d+\s*)*)\]'
                        array_match = re.search(array_pattern, response_text)
                        if array_match:
                            array_str = "[" + array_match.group(1) + "]"
                            scores = json.loads(array_str)
                            scores = [max(0, min(100, float(score))) for score in scores]
                            scores = [round(score, 2) for score in scores]
                            return scores, None
                        
                        # 4. 如果无法提取数组，尝试直接提取数字
                        numbers = re.findall(r'\b\d+(?:\.\d+)?\b', response_text)
                        if len(numbers) >= len(responses):
                            scores = [float(numbers[i]) for i in range(len(responses))]
                            scores = [max(0, min(100, score)) for score in scores]
                            scores = [round(score, 2) for score in scores]
                            return scores, None
                        
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
        
        all_scores = []
        all_details = []
        
        # 按批次处理
        for i in range(0, len(responses), self.batch_size):
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
                                score, detail = self.evaluate(response, prompt, reference)
                                all_scores.append(score)
                                if not simplified:
                                    all_details.append(detail)
                                print(f"单个评估 - 回复 #{item_index+1}: {score}")
                                if not simplified and detail:
                                    print(f"评价: {detail.get('评价', '无评价')}")
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
        
        return all_scores, all_details if not simplified else None

if __name__ == "__main__":
    """
    简单的测试代码，通过实际调用API来测试评估器功能
    """
    # 测试数据
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
        }
    ]
    
    print("开始测试OpenAI评估器...\n")
    
    try:
        # 创建评估器实例
        evaluator = OpenAIEvaluator(batch_size=2)  # 设置批次大小为2进行测试
        
        # 检查API密钥和基础URL设置
        print(f"使用基础URL: {evaluator.base_url}")
        print(f"使用模型: {evaluator.model}")
        print(f"API密钥可用: {evaluator.is_available()}\n")
        
        if not evaluator.is_available():
            print("错误: API密钥未设置，无法进行测试")
            sys.exit(1)
        
        # 逐个测试用例
        for i, case in enumerate(test_cases):
            print(f"测试用例 #{i+1}:")
            print(f"提问: {case['prompt']}")
            print(f"回复: {case['response']}")
            if case['reference']:
                print(f"参考: {case['reference']}")
            
            try:
                # 调用评估函数
                score, details = evaluator.evaluate(
                    case['response'], 
                    case['prompt'], 
                    case['reference']
                )
                
                # 输出结果
                print("\n评分结果:")
                print(f"总分: {score}/100")
                print(f"连贯性: {details.get('连贯性', 'N/A')}")
                print(f"逻辑性: {details.get('逻辑性', 'N/A')}")
                print(f"对应性: {details.get('对应性', 'N/A')}")
                print(f"评价: {details.get('评价', 'N/A')}")
                print("\n" + "-"*50 + "\n")
            except Exception as e:
                print(f"测试失败: {str(e)}\n")
        
        # 批量测试
        print("批量评估测试:")
        responses = [case['response'] for case in test_cases]
        prompts = [case['prompt'] for case in test_cases]
        references = [case['reference'] for case in test_cases]
        
        # 测试批量评估
        print("\n测试一次评估多个回复:")
        scores, details = evaluator.evaluate_multiple(responses, prompts, references)
        print(f"批量评分结果: {scores}")
        
        print("\n测试分批评估:")
        scores, details = evaluator.evaluate_batch(responses, prompts, references)
        print(f"批量评分结果: {scores}\n")
        
        print("测试完成!")
    
    except Exception as e:
        print(f"测试过程中出错: {str(e)}")
        sys.exit(1)
