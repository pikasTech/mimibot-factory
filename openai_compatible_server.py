import threading
import config
import time
import uuid
import json
import re
import argparse
from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
import random
from transformers import TrainerCallback
# 导入模板应用函数
from utils import apply_template


class OpenAICompatibleServer:
    """提供与OpenAI API兼容格式的推理服务器"""

    def __init__(self, port=8000, host="0.0.0.0", base_path="/v1", simulation_mode=False):
        """
        参数:
            port: API服务器端口
            host: API服务器主机地址
            base_path: API基础路径，兼容OpenAI格式
            simulation_mode: 是否使用模拟模式（不实际调用模型）
        """
        self.port = port
        self.host = host
        self.base_path = base_path
        self.simulation_mode = simulation_mode
        self.trainer = None
        self.tokenizer = None
        self.request_queue = {}  # 使用字典存储请求，键为请求ID
        self.result_dict = {}    # 存储结果的字典
        self.server = None
        self.app = None
        self.system_prompt = None
        self.processing_thread = None
        self.stop_processing = False

    def _format_chat_response(self, request_id, completion_text, prompt_tokens, completion_tokens, model_name):
        """格式化聊天完成API的响应"""
        return {
            "id": f"chatcmpl-{request_id}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": completion_text.strip()
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            }
        }

    def _format_completion_response(self, request_id, completion_text, prompt_tokens, completion_tokens, model_name):
        """格式化文本完成API的响应"""
        return {
            "id": f"cmpl-{request_id}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": model_name,
            "choices": [
                {
                    "text": completion_text,
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            }
        }

    def _format_chat_streaming_chunk(self, request_id, content, finish_reason, model_name):
        """格式化流式响应的单个数据块"""
        data = {
            "id": f"chatcmpl-{request_id}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": content},
                    "finish_reason": finish_reason
                }
            ]
        }
        return f"data: {json.dumps(data)}\n\n"

    def _prepare_chat_prompt(self, messages):
        """将聊天消息转换为模型输入格式"""
        # 如果有tokenizer并且需要应用模板，使用apply_template
        if self.tokenizer:
            # 从messages中提取系统提示和用户提示
            system_content = None
            user_content = None

            for msg in messages:
                role = msg.get("role", "")
                content = msg.get("content", "")

                if role == "system":
                    system_content = content
                elif role == "user":
                    # 获取最后一个用户消息
                    user_content = content

            # 如果没有系统消息，使用默认的系统提示
            if not system_content:
                system_content = self.system_prompt

            # 如果有用户消息，应用模板
            if user_content:
                return apply_template(user_content, self.tokenizer, system_content)

        # fallback到原来的处理方式
        prompt = ""
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")

            if role == "system":
                prompt += f"System: {content}\n\n"
            elif role == "user":
                prompt += f"User: {content}\n\n"
            elif role == "assistant":
                prompt += f"Assistant: {content}\n\n"

        prompt += "Assistant: "
        return prompt

    def _estimate_tokens(self, text):
        """估算文本中的token数量"""
        # 简单估算：平均每个单词约1.3个token
        return len(re.findall(r'\w+', text)) + len(re.findall(r'[^\w\s]', text))

    def _generate_simulation_text(self, prompt, max_tokens=100):
        """生成模拟的回复文本"""
        # 保证生成的模拟文本有一定的相关性
        prompt_words = re.findall(r'\w+', prompt.lower())

        # 预定义一些回复模板
        templates = [
            "这是一个模拟回复。您询问了关于{}的问题，这是一个重要的话题。",
            "作为AI助手，我可以告诉您关于{}的信息。这只是一个模拟响应。",
            "您好！您似乎对{}很感兴趣。在实际部署中，我会提供详细信息。",
            "感谢您的提问。关于{}，我在模拟模式下无法提供详细信息，但在实际部署后会给出全面回答。",
            "这是模拟模式下的回复。我注意到您提到了{}，这是一个值得探讨的话题。"
        ]

        # 从提问中提取关键词
        keywords = []
        for word in prompt_words:
            if len(word) > 4 and word not in ["what", "when", "where", "which", "there", "their", "about"]:
                keywords.append(word)

        # 如果没有提取到关键词，使用默认词
        if not keywords:
            keywords = ["您提到的主题"]

        # 随机选择模板和关键词
        template = random.choice(templates)
        keyword = random.choice(keywords) if keywords else "您的问题"

        # 生成回复
        base_response = template.format(keyword)

        # 添加随机的额外句子
        extra_sentences = [
            "模拟模式只提供示例回复，不代表实际模型输出。",
            "在正式环境中，模型将基于训练数据提供更准确的回答。",
            "这只是一个占位响应，用于测试API功能。",
            "请注意这只是一个模拟回复，用于验证系统工作状态。",
            "实际模型会生成更相关、更详细的内容。"
        ]

        # 随机决定要添加的额外句子数量(1-3句)
        num_extra = random.randint(1, min(3, max_tokens // 20))
        selected_extras = random.sample(extra_sentences, num_extra)

        return base_response + " " + " ".join(selected_extras)

    def _stream_simulation_text(self, prompt, max_tokens=100):
        """生成模拟的流式回复文本"""
        simulated_text = self._generate_simulation_text(prompt, max_tokens)
        # 将生成的文本分成小块
        chunks = []
        words = simulated_text.split()

        # 每次发送1-3个单词
        i = 0
        while i < len(words):
            chunk_size = min(random.randint(1, 3), len(words) - i)
            chunk = " ".join(words[i:i+chunk_size])
            # 确保标点符号和单词之间没有多余空格
            chunk = chunk.replace(" .", ".").replace(
                " ,", ",").replace(" !", "!").replace(" ?", "?")
            chunks.append(chunk)
            i += chunk_size

        return chunks

    def set_trainer(self, trainer):
        """设置训练器实例"""
        self.trainer = trainer

    def set_tokenizer(self, tokenizer):
        """设置分词器"""
        self.tokenizer = tokenizer

    def set_system_prompt(self, system_prompt):
        """设置系统提示"""
        self.system_prompt = system_prompt

    def start(self, daemon=True):
        """启动API服务"""
        print(
            f"Starting OpenAI compatible API service on {self.host}:{self.port}{self.base_path}")
        print(f"Simulation mode: {self.simulation_mode}")

        self.app = Flask("OpenAI-Compatible-API")
        CORS(self.app)  # 启用CORS支持

        @self.app.route(f"{self.base_path}/chat/completions", methods=["POST"])
        def chat_completions():
            try:
                # 接收请求
                req_data = request.json
                messages = req_data.get("messages", [])
                if not messages:
                    return jsonify({"error": {"message": "No messages provided", "type": "invalid_request_error"}}), 400

                # 检索参数
                model = req_data.get("model", "default-model")
                temperature = req_data.get("temperature", 0.7)
                max_tokens = req_data.get("max_tokens", 256)
                stream = req_data.get("stream", False)

                # 准备提示
                prompt = self._prepare_chat_prompt(messages) + "<think>"
                prompt_token_count = self._estimate_tokens(prompt)
                request_id = str(uuid.uuid4())[:8]

                # 处理流式响应
                if stream:
                    # 模拟模式的流式响应
                    if self.simulation_mode:
                        def generate():
                            yield self._format_chat_streaming_chunk(request_id, "", None, model)

                            # 生成模拟的流式文本
                            chunks = self._stream_simulation_text(
                                prompt, max_tokens)

                            for i, chunk in enumerate(chunks):
                                # 添加一些延迟使流更真实
                                time.sleep(random.uniform(0.01, 0.1))
                                finish_reason = "stop" if i == len(
                                    chunks) - 1 else None
                                yield self._format_chat_streaming_chunk(request_id, chunk, finish_reason, model)

                            # 结束流
                            yield "data: [DONE]\n\n"

                        return Response(stream_with_context(generate()), mimetype='text/event-stream')

                    # 实际模式的流式响应
                    else:
                        # 将请求添加到队列，标记为流式请求
                        self.request_queue[request_id] = {
                            "type": "chat_stream",
                            "prompt": prompt,
                            "temperature": temperature,
                            "max_tokens": max_tokens,
                            "model": model,
                            "prompt_tokens": prompt_token_count,
                            "timestamp": time.time(),
                            "stream_complete": False
                        }

                        def generate_stream():
                            # 发送初始空块
                            yield self._format_chat_streaming_chunk(request_id, "", None, model)

                            # 等待并处理流式结果
                            start_time = time.time()
                            timeout = req_data.get("timeout", 600)
                            last_chunk_index = -1

                            while time.time() - start_time < timeout:
                                if request_id in self.result_dict:
                                    result = self.result_dict.get(
                                        request_id, {})

                                    # 检查是否有错误
                                    if "error" in result:
                                        # 发送错误信息
                                        error_msg = json.dumps(
                                            {"error": {"message": result["error"], "type": "inference_error"}})
                                        yield f"data: {error_msg}\n\n"
                                        yield "data: [DONE]\n\n"
                                        return

                                    # 获取当前可用的所有块
                                    chunks = result.get("chunks", [])

                                    # 发送新的块
                                    for i in range(last_chunk_index + 1, len(chunks)):
                                        chunk = chunks[i]
                                        is_last = (result.get(
                                            "stream_complete", False) and i == len(chunks) - 1)
                                        finish_reason = "stop" if is_last else None
                                        yield self._format_chat_streaming_chunk(
                                            request_id, chunk, finish_reason, model
                                        )
                                        last_chunk_index = i

                                    # 如果流已完成，结束生成器
                                    if result.get("stream_complete", False) and last_chunk_index >= len(chunks) - 1:
                                        # 清理资源
                                        if request_id in self.result_dict:
                                            del self.result_dict[request_id]
                                        yield "data: [DONE]\n\n"
                                        return

                                time.sleep(0.05)  # 短暂休眠以减少CPU使用率

                            # 如果超时，发送超时消息并结束流
                            error_msg = json.dumps(
                                {"error": {"message": "Request timeout", "type": "timeout_error"}})
                            yield f"data: {error_msg}\n\n"
                            yield "data: [DONE]\n\n"
                            # 清理资源
                            if request_id in self.result_dict:
                                del self.result_dict[request_id]

                        return Response(stream_with_context(generate_stream()), mimetype='text/event-stream')

                # 非流式响应的原有逻辑
                if self.simulation_mode:
                    simulated_text = self._generate_simulation_text(
                        prompt, max_tokens)
                    response = self._format_chat_response(
                        str(uuid.uuid4())[:8],
                        simulated_text,
                        prompt_token_count,
                        self._estimate_tokens(simulated_text),
                        model
                    )
                    # 模拟处理延迟
                    time.sleep(min(0.5 + len(prompt) / 5000, 2.0))
                    return jsonify(response)

                # 下面是实际模式的处理逻辑
                # 生成请求ID
                request_id = str(uuid.uuid4())[:8]

                # 将请求添加到队列
                self.request_queue[request_id] = {
                    "type": "chat",
                    "prompt": prompt,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "model": model,
                    "prompt_tokens": prompt_token_count,
                    "timestamp": time.time()
                }

                # 等待结果
                start_time = time.time()
                timeout = req_data.get("timeout", 600)

                while time.time() - start_time < timeout:
                    if request_id in self.result_dict:
                        result = self.result_dict.pop(request_id)
                        if "error" in result:
                            return jsonify({"error": {"message": result["error"], "type": "inference_error"}}), 500

                        # 格式化为OpenAI格式响应
                        response = self._format_chat_response(
                            request_id,
                            result["text"],
                            prompt_token_count,
                            self._estimate_tokens(result["text"]),
                            model
                        )
                        return jsonify(response)
                    time.sleep(0.1)

                # 超时，返回错误
                del self.request_queue[request_id]
                return jsonify({"error": {"message": "Request timeout", "type": "timeout_error"}}), 408

            except Exception as e:
                return jsonify({"error": {"message": str(e), "type": "server_error"}}), 500

        @self.app.route(f"{self.base_path}/completions", methods=["POST"])
        def completions():
            try:
                # 接收请求
                req_data = request.json
                prompt = req_data.get("prompt", "")
                if not prompt:
                    return jsonify({"error": {"message": "No prompt provided", "type": "invalid_request_error"}}), 400

                # 检索参数
                model = req_data.get("model", "default-model")
                temperature = req_data.get("temperature", 0.7)
                max_tokens = req_data.get("max_tokens", 256)
                prompt_token_count = self._estimate_tokens(prompt)

                # 模拟模式直接返回模拟响应
                if self.simulation_mode:
                    simulated_text = self._generate_simulation_text(
                        prompt, max_tokens)
                    response = self._format_completion_response(
                        str(uuid.uuid4())[:8],
                        simulated_text,
                        prompt_token_count,
                        self._estimate_tokens(simulated_text),
                        model
                    )
                    # 模拟处理延迟
                    time.sleep(min(0.2 + len(prompt) / 8000, 1.5))
                    return jsonify(response)

                # 实际模式处理逻辑
                # 生成请求ID
                request_id = str(uuid.uuid4())[:8]

                # 将请求添加到队列
                self.request_queue[request_id] = {
                    "type": "completion",
                    "prompt": prompt,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "model": model,
                    "prompt_tokens": prompt_token_count,
                    "timestamp": time.time()
                }

                # 等待结果
                start_time = time.time()
                timeout = req_data.get("timeout", 600)

                while time.time() - start_time < timeout:
                    if request_id in self.result_dict:
                        result = self.result_dict.pop(request_id)
                        if "error" in result:
                            return jsonify({"error": {"message": result["error"], "type": "inference_error"}}), 500

                        # 格式化为OpenAI格式响应
                        response = self._format_completion_response(
                            request_id,
                            result["text"],
                            prompt_token_count,
                            self._estimate_tokens(result["text"]),
                            model
                        )
                        return jsonify(response)
                    time.sleep(0.1)

                # 超时，返回错误
                del self.request_queue[request_id]
                return jsonify({"error": {"message": "Request timeout", "type": "timeout_error"}}), 408

            except Exception as e:
                return jsonify({"error": {"message": str(e), "type": "server_error"}}), 500

        @self.app.route(f"{self.base_path}/models", methods=["GET"])
        def list_models():
            """返回可用模型列表"""
            model_name = "trained-model"
            if hasattr(self.trainer, "model") and hasattr(self.trainer.model, "name_or_path"):
                model_name = self.trainer.model.name_or_path

            return jsonify({
                "object": "list",
                "data": [
                    {
                        "id": model_name,
                        "object": "model",
                        "created": int(time.time()),
                        "owned_by": "user"
                    }
                ]
            })

        # 健康检查端点
        @self.app.route(f"{self.base_path}/health", methods=["GET"])
        def health_check():
            return jsonify({
                "status": "ok",
                "simulation_mode": self.simulation_mode,
                "timestamp": time.time()
            })

        # 启动服务器线程
        self.server_thread = threading.Thread(
            target=lambda: self.app.run(
                host=self.host, port=self.port, threaded=True),
            daemon=daemon
        )
        self.server_thread.start()

        # 如果不是模拟模式，启动后台处理线程
        if not self.simulation_mode and self.trainer is not None:
            self.stop_processing = False
            self.processing_thread = threading.Thread(
                target=self._background_processing,
                daemon=daemon
            )
            self.processing_thread.start()
            print("Started background request processing thread")

        return self.server_thread

    def _background_processing(self):
        """后台线程，周期性地处理请求队列"""
        while not self.stop_processing:
            try:
                # 如果有请求，处理它们
                if self.request_queue:
                    self.process_requests()
                # 短暂休眠以减少CPU使用
                time.sleep(0.1)
            except Exception as e:
                print(f"Background processing error: {str(e)}")
                time.sleep(1)  # 发生错误时暂停更长时间

    def stop(self):
        """停止服务器和处理线程"""
        self.stop_processing = True
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=2)
            print("Stopped background processing thread")

    def process_requests(self):
        """处理队列中的推理请求 - 供回调使用"""
        if not self.request_queue or not hasattr(self.trainer, 'llm'):
            return
        # 处理最老的一个请求(如果存在)
        current_time = time.time()
        oldest_request_id = None
        oldest_time = float('inf')

        # 找出最老的请求
        for req_id, req_data in list(self.request_queue.items()):
            if req_data["timestamp"] < oldest_time:
                oldest_time = req_data["timestamp"]
                oldest_request_id = req_id

        if not oldest_request_id:
            return

        try:
            # 获取请求数据
            request_data = self.request_queue.pop(oldest_request_id)
            prompt = request_data["prompt"]
            temperature = request_data["temperature"]
            max_tokens = request_data["max_tokens"]
            request_type = request_data.get("type", "completion")

            print("开始处理API请求...")
            print(f"Chat prompt: {prompt}")

            # 设置vLLM采样参数
            from vllm import SamplingParams
            sampling_params = SamplingParams(
                temperature=temperature,
                max_tokens=max_tokens
            )

            # 初始化结果字典
            self.result_dict[oldest_request_id] = {
                "chunks": [],
                "stream_complete": False,
                "success": True
            }

            outputs = self.trainer.llm.generate(
                [prompt],
                sampling_params=sampling_params,
                use_tqdm=False,
                lora_request=self.trainer.model.load_lora('grpo_trainer_lora_model', load_tensors=True)
            )

            for output in outputs:
                if output.outputs:
                    # 提取当前生成的新内容
                    new_text = output.outputs[0].text
                    # 如果已存在结果，获取之前的文本长度
                    prev_text = ""
                    for chunk in self.result_dict[oldest_request_id]["chunks"]:
                        prev_text += chunk

                    # 仅添加新生成的部分
                    if len(new_text) > len(prev_text):
                        new_chunk = new_text[len(prev_text):]
                        self.result_dict[oldest_request_id]["chunks"].append(
                            new_chunk)

                # 检查是否完成
                if output.finished:
                    self.result_dict[oldest_request_id]["stream_complete"] = True
                    break

        except Exception as e:
            # 处理错误情况
            self.result_dict[oldest_request_id] = {
                "error": str(e),
                "success": False
            }


class OpenAICompatibleCallback(TrainerCallback):
    """用于Trainer的OpenAI兼容API回调"""

    def __init__(self, port=8000, simulation_mode=False, trainer_getter=None, system_prompt=None, tokenizer=None):
        """
        参数:
            port: API服务器端口
            simulation_mode: 是否使用模拟模式
            trainer_getter: 获取trainer实例的函数
            system_prompt: 系统提示文本
        """
        self.port = port
        self.simulation_mode = simulation_mode
        self.server = None
        self.trainer_getter = trainer_getter
        self.system_prompt = system_prompt
        self.tokenizer = tokenizer

    def on_train_begin(self, args, state, control, **kwargs):
        """训练开始时启动API服务"""
        self.trainer = self.trainer_getter() if self.trainer_getter else kwargs.get('trainer')
        if not self.trainer:
            print("Trainer实例未设置，无法启动API服务器")
            return control

        if self.trainer.accelerator.is_main_process:
            self.server = OpenAICompatibleServer(
                port=self.port, simulation_mode=self.simulation_mode)
            self.server.set_trainer(self.trainer)
            self.server.set_tokenizer(self.tokenizer)

            if self.system_prompt:
                self.server.set_system_prompt(self.system_prompt)

            self.server.start()

        return control

    def on_step_begin(self, args, state, control, **kwargs):
        """每个训练步骤开始时处理推理请求"""
        # 只在主进程处理请求
        if not self.trainer or not self.trainer.accelerator.is_main_process or self.simulation_mode or not self.server:
            return control

        # 处理队列中的请求
        self.server.process_requests()

        return control
    
    def on_step_end(self, args, state, control, **kwargs):
        return self.on_step_begin(args, state, control, **kwargs)
    
    def on_substep_end(self, args, state, control, **kwargs):
        return self.on_step_begin(args, state, control, **kwargs)

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        return self.on_step_begin(args, state, control, **kwargs)

    def on_optimizer_step(self, args, state, control, **kwargs):
        return self.on_step_begin(args, state, control, **kwargs)


def load_model_for_inference(base_model_path, lora_model_path=None, device_map="auto", offload_folder="tmp_offload",
                             gpu_memory_utilization=0.8, max_model_len=4096, trust_remote_code=True):
    """
    加载基础模型和可选的LoRA模型用于推理
    
    Args:
        base_model_path: 基础模型的路径
        lora_model_path: LoRA模型的路径，如果不提供则只加载基础模型
        device_map: 模型加载的设备映射策略
        offload_folder: 模型卸载目录路径
        gpu_memory_utilization: GPU内存利用率(0.0-1.0)
        max_model_len: 模型最大序列长度
        trust_remote_code: 是否允许远程代码执行
        
    Returns:
        加载好的模型和分词器
    """
    import os
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from vllm import LLM
    
    print(f"开始加载模型...")
    print(f"基础模型: {base_model_path}")
    if lora_model_path:
        print(f"LoRA模型: {lora_model_path}")
    print(f"设备映射: {device_map}")
    print(f"卸载目录: {offload_folder}")
    print("-" * 50)
    
    # 始终确保offload_folder存在
    if offload_folder:
        os.makedirs(offload_folder, exist_ok=True)
        print(f"创建或确认卸载目录: {offload_folder}")
    
    try:
        print(f"正在加载基础模型和分词器: {base_model_path}")
        # 首先加载分词器
        tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        
        # 初始化LLM
        llm = LLM(
            model=base_model_path,
            tokenizer=base_model_path,
            tensor_parallel_size=1,  # 可以根据GPU数量调整
            gpu_memory_utilization=gpu_memory_utilization,  # 增加GPU内存利用率
            max_model_len=max_model_len,  # 降低最大模型长度，减少KV缓存需求
            quantization=None,  # 可选的量化参数
            trust_remote_code=trust_remote_code  # 允许远程代码执行
        )
        
        print(f"模型加载完成!")
        return tokenizer, llm
        
    except Exception as e:
        print(f"加载模型时出错: {str(e)}")
        raise


# 当直接运行此文件时，以模拟模式启动服务器
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='启动OpenAI兼容的API服务器')
    parser.add_argument('--port', type=int, default=8099, help='服务器端口')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='服务器主机地址')
    parser.add_argument('--base-path', type=str, default='/v1', help='API基础路径')
    parser.add_argument('--no-simulation', default=True, help='禁用模拟模式')
    # 添加模型加载相关参数
    parser.add_argument('--base_model_path', default=config.BASE_MODEL, help='基础模型的路径')
    parser.add_argument('--lora_model_path', default=config.LORA_PATH, help='LoRA模型的路径')
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.95, help='GPU内存利用率(0.0-1.0)')
    parser.add_argument('--max_model_len', type=int, default=256, help='模型最大序列长度')
    parser.add_argument('--device_map', type=str, default="cuda", help='模型加载的设备映射策略')
    parser.add_argument('--offload_folder', type=str, default="tmp_offload", help='模型卸载目录路径')
    parser.add_argument('--trust_remote_code', action='store_true', help='允许远程代码执行')
    parser.add_argument('--system_prompt', type=str, default=None, help='系统提示文本')

    args = parser.parse_args()

    server = OpenAICompatibleServer(
        port=args.port,
        host=args.host,
        base_path=args.base_path,
        simulation_mode=not args.no_simulation
    )
    
    # 当指定了基础模型路径且不在模拟模式时，加载模型
    if args.base_model_path and not server.simulation_mode:
        try:
            # 创建一个包含trainer属性的虚拟对象
            class TrainerSimulator:
                def __init__(self, tokenizer, llm):
                    self.tokenizer = tokenizer
                    self.llm = llm
                    # 添加一个假的accelerator对象，用于兼容原有代码
                    class Accelerator:
                        def __init__(self):
                            self.is_main_process = True
                    self.accelerator = Accelerator()
            
            # 加载模型
            tokenizer, llm = load_model_for_inference(
                args.base_model_path, 
                args.lora_model_path,
                args.device_map,
                args.offload_folder,
                gpu_memory_utilization=args.gpu_memory_utilization,
                max_model_len=args.max_model_len,
                trust_remote_code=args.trust_remote_code
            )
            
            # 创建trainer模拟器
            trainer = TrainerSimulator(tokenizer, llm)
            
            # 设置trainer和tokenizer
            server.set_trainer(trainer)
            server.set_tokenizer(tokenizer)
            
            # 设置系统提示
            if args.system_prompt:
                server.set_system_prompt(args.system_prompt)
                
            print("模型加载成功，准备启动API服务")
        except Exception as e:
            print(f"加载模型失败: {str(e)}")
            print("将以模拟模式启动服务器")
            server.simulation_mode = True

    print("按Ctrl+C停止服务器")
    try:
        server_thread = server.start(daemon=False)
        server_thread.join()
    except KeyboardInterrupt:
        print("服务器已停止")
