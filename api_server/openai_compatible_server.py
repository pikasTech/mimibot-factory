import threading
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
                            timeout = req_data.get("timeout", 30)
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
                timeout = req_data.get("timeout", 30)

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
                timeout = req_data.get("timeout", 30)

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

        return self.server_thread

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

            # 处理流式请求
            if request_type == "chat_stream":
                # 初始化结果字典
                self.result_dict[oldest_request_id] = {
                    "chunks": [],
                    "stream_complete": False,
                    "success": True
                }

                outputs = self.trainer.llm.generate(
                    [prompt],
                    sampling_params=sampling_params,
                    use_tqdm=False
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

            # 处理非流式请求
            else:
                # 生成文本
                outputs = self.trainer.llm.generate(
                    [prompt],
                    sampling_params=sampling_params,
                    use_tqdm=False
                )

                # 提取生成的文本
                generated_text = outputs[0].outputs[0].text

                # 存储结果
                self.result_dict[oldest_request_id] = {
                    "text": generated_text,
                    "success": True
                }

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


# 当直接运行此文件时，以模拟模式启动服务器
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='启动OpenAI兼容的API服务器')
    parser.add_argument('--port', type=int, default=8099, help='服务器端口')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='服务器主机地址')
    parser.add_argument('--base-path', type=str, default='/v1', help='API基础路径')
    parser.add_argument('--no-simulation', action='store_true', help='禁用模拟模式')

    args = parser.parse_args()

    server = OpenAICompatibleServer(
        port=args.port,
        host=args.host,
        base_path=args.base_path,
        simulation_mode=not args.no_simulation
    )

    print("按Ctrl+C停止服务器")
    try:
        server_thread = server.start(daemon=False)
        server_thread.join()
    except KeyboardInterrupt:
        print("服务器已停止")
