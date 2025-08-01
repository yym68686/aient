import os
import re
import json
import copy
import httpx
import asyncio
import requests
from typing import Set
from typing import Union, Optional, Callable, List, Dict, Any
from pathlib import Path


from .base import BaseLLM
from ..plugins import PLUGINS, get_tools_result_async, function_call_list, update_tools_config
from ..utils.scripts import safe_get, async_generator_to_sync, parse_function_xml, parse_continuous_json, convert_functions_to_xml, remove_xml_tags_and_content
from ..core.request import prepare_request_payload
from ..core.response import fetch_response_stream

def get_filtered_keys_from_object(obj: object, *keys: str) -> Set[str]:
    """
    Get filtered list of object variable names.
    :param keys: List of keys to include. If the first key is "not", the remaining keys will be removed from the class keys.
    :return: List of class keys.
    """
    class_keys = obj.__dict__.keys()
    if not keys:
        return set(class_keys)

    # Remove the passed keys from the class keys.
    if keys[0] == "not":
        return {key for key in class_keys if key not in keys[1:]}
    # Check if all passed keys are valid
    if invalid_keys := set(keys) - class_keys:
        raise ValueError(
            f"Invalid keys: {invalid_keys}",
        )
    # Only return specified keys that are in class_keys
    return {key for key in keys if key in class_keys}

class chatgpt(BaseLLM):
    """
    Official ChatGPT API
    """

    def __init__(
        self,
        api_key: str = None,
        engine: str = os.environ.get("GPT_ENGINE") or "gpt-4o",
        api_url: str = (os.environ.get("API_URL") or "https://api.openai.com/v1/chat/completions"),
        system_prompt: str = "You are ChatGPT, a large language model trained by OpenAI. Respond conversationally",
        proxy: str = None,
        timeout: float = 600,
        max_tokens: int = None,
        temperature: float = 0.5,
        top_p: float = 1.0,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        reply_count: int = 1,
        truncate_limit: int = None,
        use_plugins: bool = True,
        print_log: bool = False,
        tools: Optional[Union[list, str, Callable]] = [],
        function_call_max_loop: int = 3,
        cut_history_by_function_name: str = "",
        cache_messages: list = None,
    ) -> None:
        """
        Initialize Chatbot with API key (from https://platform.openai.com/account/api-keys)
        """
        super().__init__(api_key, engine, api_url, system_prompt, proxy, timeout, max_tokens, temperature, top_p, presence_penalty, frequency_penalty, reply_count, truncate_limit, use_plugins=use_plugins, print_log=print_log)
        self.conversation: dict[str, list[dict]] = {
            "default": [
                {
                    "role": "system",
                    "content": self.system_prompt,
                },
            ],
        }
        if cache_messages:
            self.conversation["default"] = cache_messages
        self.function_calls_counter = {}
        self.function_call_max_loop = function_call_max_loop
        self.cut_history_by_function_name = cut_history_by_function_name
        self.latest_file_content = {}


        # 注册和处理传入的工具
        self._register_tools(tools)


    def _register_tools(self, tools):
        """动态注册工具函数并更新配置"""

        self.plugins = copy.deepcopy(PLUGINS)
        self.function_call_list = copy.deepcopy(function_call_list)
        # 如果有新工具，需要注册到registry并更新配置
        self.plugins, self.function_call_list, _ = update_tools_config()

        if isinstance(tools, list):
            self.tools = tools if tools else []
        else:
            self.tools = [tools] if tools else []

        for tool in self.tools:
            tool_name = tool.__name__ if callable(tool) else str(tool)
            if tool_name in self.plugins:
                self.plugins[tool_name] = True
            else:
                raise ValueError(f"Tool {tool_name} not found in plugins")

    def add_to_conversation(
        self,
        message: Union[str, list],
        role: str,
        convo_id: str = "default",
        function_name: str = "",
        total_tokens: int = 0,
        function_arguments: str = "",
        pass_history: int = 9999,
        function_call_id: str = "",
    ) -> None:
        """
        Add a message to the conversation
        """
        # print("role", role, "function_name", function_name, "message", message)
        if convo_id not in self.conversation:
            self.reset(convo_id=convo_id)
        if function_name == "" and message:
            self.conversation[convo_id].append({"role": role, "content": message})
        elif function_name != "" and message:
            # 删除从 cut_history_by_function_name 以后的所有历史记录
            if function_name == self.cut_history_by_function_name:
                matching_message = next(filter(lambda x: safe_get(x, "tool_calls", 0, "function", "name", default="") == 'get_next_pdf', self.conversation[convo_id]), None)
                if matching_message is not None:
                    self.conversation[convo_id] = self.conversation[convo_id][:self.conversation[convo_id].index(matching_message)]

            if not (all(value == False for value in self.plugins.values()) or self.use_plugins == False):
                self.conversation[convo_id].append({
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": function_call_id,
                            "type": "function",
                            "function": {
                                "name": function_name,
                                "arguments": function_arguments,
                            },
                        }
                    ],
                    })
                self.conversation[convo_id].append({"role": role, "tool_call_id": function_call_id, "content": message})
            else:
                last_user_message = self.conversation[convo_id][-1]["content"]
                if last_user_message != message:
                    image_message_list = []
                    if isinstance(function_arguments, str):
                        functions_list = json.loads(function_arguments)
                    else:
                        functions_list = function_arguments
                    for tool_info in functions_list:
                        if tool_info.get("base64_image"):
                            image_message_list.append({"type": "text", "text": safe_get(tool_info, "parameter", "image_path", default="") + " image:"})
                            image_message_list.append({
                                "type": "image_url",
                                "image_url": {
                                    "url": tool_info["base64_image"],
                                }
                            })
                    self.conversation[convo_id].append({"role": "assistant", "content": convert_functions_to_xml(function_arguments)})
                    if image_message_list:
                        self.conversation[convo_id].append({"role": "user", "content": [{"type": "text", "text": message}] + image_message_list})
                    else:
                        self.conversation[convo_id].append({"role": "user", "content": message})
                else:
                    self.conversation[convo_id].append({"role": "assistant", "content": "我已经执行过这个工具了，接下来我需要做什么？"})

        else:
            print('\033[31m')
            print("error: add_to_conversation message is None or empty")
            print("role", role, "function_name", function_name, "message", message)
            print('\033[0m')

        conversation_len = len(self.conversation[convo_id]) - 1
        message_index = 0
        # if self.print_log:
        #     replaced_text = json.loads(re.sub(r';base64,([A-Za-z0-9+/=]+)', ';base64,***', json.dumps(self.conversation[convo_id])))
        #     print(json.dumps(replaced_text, indent=4, ensure_ascii=False))
        while message_index < conversation_len:
            if self.conversation[convo_id][message_index]["role"] == self.conversation[convo_id][message_index + 1]["role"]:
                if self.conversation[convo_id][message_index].get("content") and self.conversation[convo_id][message_index + 1].get("content") \
                and self.conversation[convo_id][message_index].get("content") != self.conversation[convo_id][message_index + 1].get("content"):
                    if type(self.conversation[convo_id][message_index + 1]["content"]) == str \
                    and type(self.conversation[convo_id][message_index]["content"]) == list:
                        self.conversation[convo_id][message_index + 1]["content"] = [{"type": "text", "text": self.conversation[convo_id][message_index + 1]["content"]}]
                    if type(self.conversation[convo_id][message_index]["content"]) == str \
                    and type(self.conversation[convo_id][message_index + 1]["content"]) == list:
                        self.conversation[convo_id][message_index]["content"] = [{"type": "text", "text": self.conversation[convo_id][message_index]["content"]}]
                    if type(self.conversation[convo_id][message_index]["content"]) == dict \
                    and type(self.conversation[convo_id][message_index + 1]["content"]) == str:
                        self.conversation[convo_id][message_index]["content"] = [self.conversation[convo_id][message_index]["content"]]
                        self.conversation[convo_id][message_index + 1]["content"] = [{"type": "text", "text": self.conversation[convo_id][message_index + 1]["content"]}]
                    if type(self.conversation[convo_id][message_index]["content"]) == dict \
                    and type(self.conversation[convo_id][message_index + 1]["content"]) == list:
                        self.conversation[convo_id][message_index]["content"] = [self.conversation[convo_id][message_index]["content"]]
                    if type(self.conversation[convo_id][message_index]["content"]) == dict \
                    and type(self.conversation[convo_id][message_index + 1]["content"]) == dict:
                        self.conversation[convo_id][message_index]["content"] = [self.conversation[convo_id][message_index]["content"]]
                        self.conversation[convo_id][message_index + 1]["content"] = [self.conversation[convo_id][message_index + 1]["content"]]
                    if type(self.conversation[convo_id][message_index]["content"]) == list \
                    and type(self.conversation[convo_id][message_index + 1]["content"]) == dict:
                        self.conversation[convo_id][message_index + 1]["content"] = [self.conversation[convo_id][message_index + 1]["content"]]
                    self.conversation[convo_id][message_index]["content"] += self.conversation[convo_id][message_index + 1]["content"]
                self.conversation[convo_id].pop(message_index + 1)
                conversation_len = conversation_len - 1
            else:
                message_index = message_index + 1

        history_len = len(self.conversation[convo_id])

        history = pass_history
        if pass_history < 2:
            history = 2
        while history_len > history:
            mess_body = self.conversation[convo_id].pop(1)
            history_len = history_len - 1
            if mess_body.get("role") == "user":
                assistant_body = self.conversation[convo_id].pop(1)
                history_len = history_len - 1
                if assistant_body.get("tool_calls"):
                    self.conversation[convo_id].pop(1)
                    history_len = history_len - 1

        if total_tokens:
            self.current_tokens[convo_id] = total_tokens
            self.tokens_usage[convo_id] += total_tokens

    def truncate_conversation(self, convo_id: str = "default") -> None:
        """
        Truncate the conversation
        """
        while True:
            if (
                self.current_tokens[convo_id] > self.truncate_limit
                and len(self.conversation[convo_id]) > 1
            ):
                # Don't remove the first message
                mess = self.conversation[convo_id].pop(1)
                string_mess = json.dumps(mess, ensure_ascii=False)
                self.current_tokens[convo_id] -= len(string_mess) / 4
                print("Truncate message:", mess)
            else:
                break

    def get_latest_file_content(self) -> str:
        """
        获取最新文件内容
        """
        result = ""
        if self.latest_file_content:
            for file_path, content in self.latest_file_content.items():
                result += (
                    "<file>"
                    f"<file_path>{file_path}</file_path>"
                    f"<file_content>{content}</file_content>"
                    "</file>\n\n"
                )
            if result:
                result = (
                    "<latest_file_content>"
                    f"{result}"
                    "</latest_file_content>"
                )
        return result

    async def get_post_body(
        self,
        prompt: str,
        role: str = "user",
        convo_id: str = "default",
        model: str = "",
        pass_history: int = 9999,
        **kwargs,
    ):
        self.conversation[convo_id][0] = {"role": "system","content": self.system_prompt + "\n\n" + self.get_latest_file_content()}

        # 构造 provider 信息
        provider = {
            "provider": "openai",
            "base_url": kwargs.get('api_url', self.api_url.chat_url),
            "api": kwargs.get('api_key', self.api_key),
            "model": [model or self.engine],
            "tools": True if self.use_plugins else False,
            "image": True
        }

        # 构造请求数据
        request_data = {
            "model": model or self.engine,
            "messages": copy.deepcopy(self.conversation[convo_id]) if pass_history else [
                {"role": "system","content": self.system_prompt + "\n\n" + self.get_latest_file_content()},
                {"role": role, "content": prompt}
            ],
            "stream": True,
            "stream_options": {
                "include_usage": True
            },
            "temperature": kwargs.get("temperature", self.temperature)
        }

        if kwargs.get("max_tokens", self.max_tokens):
            request_data["max_tokens"] = kwargs.get("max_tokens", self.max_tokens)

        # 添加工具相关信息
        if kwargs.get("plugins", None):
            self.plugins = kwargs.get("plugins")

        plugins_status = kwargs.get("plugins", self.plugins)
        if not (all(value == False for value in plugins_status.values()) or self.use_plugins == False):
            tools_request_body = []
            for item in plugins_status.keys():
                try:
                    if plugins_status[item]:
                        tools_request_body.append({"type": "function", "function": self.function_call_list[item]})
                except:
                    pass
            if tools_request_body:
                request_data["tools"] = tools_request_body
                request_data["tool_choice"] = "auto"

        # print("request_data", json.dumps(request_data, indent=4, ensure_ascii=False))

        # 调用核心模块的 prepare_request_payload 函数
        url, headers, json_post_body, engine_type = await prepare_request_payload(provider, request_data)

        return url, headers, json_post_body, engine_type

    async def _process_stream_response(
        self,
        response_gen,
        convo_id="default",
        function_name="",
        total_tokens=0,
        function_arguments="",
        function_call_id="",
        model="",
        language="English",
        system_prompt=None,
        pass_history=9999,
        is_async=False,
        **kwargs
    ):
        """
        处理流式响应的共用逻辑

        :param response_gen: 响应生成器(同步或异步)
        :param is_async: 是否使用异步模式
        """
        response_role = None
        full_response = ""
        function_full_response = ""
        function_call_name = ""
        need_function_call = False

        # 处理单行数据的公共逻辑
        def process_line(line):
            nonlocal response_role, full_response, function_full_response, function_call_name, need_function_call, total_tokens, function_call_id

            if not line or (isinstance(line, str) and line.startswith(':')):
                return None

            if isinstance(line, str) and line.startswith('data:'):
                line = line.lstrip("data: ")
                if line == "[DONE]":
                    return "DONE"
            elif isinstance(line, (dict, list)):
                if isinstance(line, dict) and safe_get(line, "choices", 0, "message", "content"):
                    full_response = line["choices"][0]["message"]["content"]
                    return full_response
                else:
                    return str(line)
            else:
                try:
                    if isinstance(line, str):
                        line = json.loads(line)
                        if safe_get(line, "choices", 0, "message", "content"):
                            full_response = line["choices"][0]["message"]["content"]
                            return full_response
                        else:
                            return str(line)
                except:
                    print("json.loads error:", repr(line))
                    return None

            resp = json.loads(line) if isinstance(line, str) else line
            if "error" in resp:
                raise Exception(f"{resp}")

            total_tokens = total_tokens or safe_get(resp, "usage", "total_tokens", default=0)
            delta = safe_get(resp, "choices", 0, "delta")
            if not delta:
                return None

            response_role = response_role or safe_get(delta, "role")
            if safe_get(delta, "content"):
                need_function_call = False
                content = delta["content"]
                full_response += content
                return content

            if safe_get(delta, "tool_calls"):
                need_function_call = True
                function_call_name = function_call_name or safe_get(delta, "tool_calls", 0, "function", "name")
                function_full_response += safe_get(delta, "tool_calls", 0, "function", "arguments", default="")
                function_call_id = function_call_id or safe_get(delta, "tool_calls", 0, "id")
                return None

        # 处理流式响应
        async def process_async():
            nonlocal response_role, full_response, function_full_response, function_call_name, need_function_call, total_tokens, function_call_id

            async for line in response_gen:
                line = line.strip() if isinstance(line, str) else line
                result = process_line(line)
                if result == "DONE":
                    break
                elif result:
                    yield result

        def process_sync():
            nonlocal response_role, full_response, function_full_response, function_call_name, need_function_call, total_tokens, function_call_id

            for line in response_gen:
                line = line.decode("utf-8") if hasattr(line, "decode") else line
                result = process_line(line)
                if result == "DONE":
                    break
                elif result:
                    yield result

        # 使用同步或异步处理器处理响应
        if is_async:
            async for chunk in process_async():
                yield chunk
        else:
            for chunk in process_sync():
                yield chunk

        if self.print_log:
            print("\n\rtotal_tokens", total_tokens)

        if response_role is None:
            response_role = "assistant"

        if self.use_plugins == True:
            full_response = full_response.replace("<tool_code>", "").replace("</tool_code>", "")
            function_parameter = parse_function_xml(full_response)
            if function_parameter:
                invalid_tools = [tool_dict for tool_dict in function_parameter if tool_dict.get("function_name", "") not in self.plugins.keys()]
                function_parameter = [tool_dict for tool_dict in function_parameter if tool_dict.get("function_name", "") in self.plugins.keys()]
                if self.print_log and invalid_tools:
                    print("invalid_tools", invalid_tools)
                    print("function_parameter", function_parameter)
                    print("full_response", full_response)
                if function_parameter:
                    need_function_call = True
                    if isinstance(self.conversation[convo_id][-1]["content"], str) and \
                    "<tool_error>" in self.conversation[convo_id][-1]["content"]:
                        need_function_call = False
                        full_response = remove_xml_tags_and_content(full_response) + "上面是我的分析，还没有实际行动。\n\n接下来我需要做什么？"
                else:
                    need_function_call = False
                    if self.print_log:
                        print("Failed to parse function_parameter full_response", full_response)
                    full_response = ""

        # 处理函数调用
        if need_function_call and self.use_plugins == True:
            if self.print_log:
                print("function_parameter", function_parameter)
                print("function_full_response", function_full_response)

            function_response = ""
            # 定义处理单个工具调用的辅助函数
            async def process_single_tool_call(tool_name, tool_args, tool_id):
                nonlocal function_response

                if not self.function_calls_counter.get(tool_name):
                    self.function_calls_counter[tool_name] = 1
                else:
                    self.function_calls_counter[tool_name] += 1

                tool_response = ""
                has_args = safe_get(self.function_call_list, tool_name, "parameters", "required", default=False)
                if self.function_calls_counter[tool_name] <= self.function_call_max_loop and (tool_args != "{}" or not has_args):
                    function_call_max_tokens = self.truncate_limit - 1000
                    if function_call_max_tokens <= 0:
                        function_call_max_tokens = int(self.truncate_limit / 2)
                    if self.print_log:
                        print(f"\033[32m function_call {tool_name}, max token: {function_call_max_tokens} \033[0m")

                    # 处理函数调用结果
                    if is_async:
                        async for chunk in get_tools_result_async(
                            tool_name, tool_args, function_call_max_tokens,
                            model or self.engine, chatgpt, kwargs.get('api_key', self.api_key),
                            kwargs.get('api_url', self.api_url.chat_url), use_plugins=False, model=model or self.engine,
                            add_message=self.add_to_conversation, convo_id=convo_id, language=language
                        ):
                            yield chunk
                    else:
                        async def run_async():
                            async for chunk in get_tools_result_async(
                                tool_name, tool_args, function_call_max_tokens,
                                model or self.engine, chatgpt, kwargs.get('api_key', self.api_key),
                                kwargs.get('api_url', self.api_url.chat_url), use_plugins=False, model=model or self.engine,
                                add_message=self.add_to_conversation, convo_id=convo_id, language=language
                            ):
                                yield chunk

                        for chunk in async_generator_to_sync(run_async()):
                            yield chunk
                else:
                    tool_response = f"无法找到相关信息，停止使用工具 {tool_name}"

                yield tool_response

            # 使用统一的JSON解析逻辑
            try:
                if function_full_response:
                    function_parameter = parse_continuous_json(function_full_response, function_call_name)
            except Exception as e:
                print(f"解析JSON失败: {e}")
                # 保持原始工具调用
                tool_calls = [{
                    'function_name': function_call_name,
                    'parameter': function_full_response,
                    'function_call_id': function_call_id
                }]

            # 统一处理逻辑，将所有情况转换为列表处理
            if isinstance(function_parameter, list) and function_parameter:
                # 多个工具调用
                tool_calls = function_parameter

            # 处理所有工具调用
            all_responses = []

            for tool_info in tool_calls:
                tool_name = tool_info['function_name']
                tool_args = json.dumps(tool_info['parameter'], ensure_ascii=False) if not isinstance(tool_info['parameter'], str) else tool_info['parameter']
                tool_id = tool_info.get('function_call_id', tool_name + "_tool_call")

                tool_response = ""
                if is_async:
                    async for chunk in process_single_tool_call(tool_name, tool_args, tool_id):
                        if isinstance(chunk, str) and "function_response:" in chunk:
                            tool_response = chunk.replace("function_response:", "")
                        else:
                            yield chunk
                else:
                    for chunk in async_generator_to_sync(process_single_tool_call(tool_name, tool_args, tool_id)):
                        if isinstance(chunk, str) and "function_response:" in chunk:
                            tool_response = chunk.replace("function_response:", "")
                        else:
                            yield chunk
                if tool_name == "read_file" and "<tool_error>" not in tool_response:
                    self.latest_file_content[tool_info['parameter']["file_path"]] = tool_response
                    all_responses.append(f"[{tool_name}({tool_args}) Result]:\n\nRead file successfully! The file content has been updated in the tag <latest_file_content>.")
                elif tool_name == "write_to_file" and "<tool_error>" not in tool_response:
                    all_responses.append(f"[{tool_name} Result]:\n\n{tool_response}")
                elif tool_name == "read_image" and "<tool_error>" not in tool_response:
                    tool_info["base64_image"] = tool_response
                    all_responses.append(f"[{tool_name}({tool_args}) Result]:\n\nRead image successfully!")
                elif tool_response.startswith("data:image/") and ";base64," in tool_response and "<tool_error>" not in tool_response:
                    tool_info["base64_image"] = tool_response
                    all_responses.append(f"[{tool_name}({tool_args}) Result]:\n\nRead image successfully!")
                else:
                    all_responses.append(f"[{tool_name}({tool_args}) Result]:\n\n{tool_response}")

            # 合并所有工具响应
            function_response = "\n\n".join(all_responses).strip()

            # 使用第一个工具的名称和参数作为历史记录
            function_call_name = tool_calls[0]['function_name']
            function_full_response = function_full_response or json.dumps(tool_calls) if not isinstance(tool_calls[0]['parameter'], str) else tool_calls
            function_call_id = tool_calls[0].get('function_call_id', function_call_name + "_tool_call")

            response_role = "tool"

            # 递归处理函数调用响应
            if is_async:
                async for chunk in self.ask_stream_async(
                    function_response, response_role, convo_id=convo_id,
                    function_name=function_call_name, total_tokens=total_tokens,
                    model=model or self.engine, function_arguments=function_full_response,
                    function_call_id=function_call_id, api_key=kwargs.get('api_key', self.api_key),
                    api_url=kwargs.get('api_url', self.api_url.chat_url),
                    plugins=kwargs.get("plugins", self.plugins), system_prompt=system_prompt
                ):
                    yield chunk
            else:
                for chunk in self.ask_stream(
                    function_response, response_role, convo_id=convo_id,
                    function_name=function_call_name, total_tokens=total_tokens,
                    model=model or self.engine, function_arguments=function_full_response,
                    function_call_id=function_call_id, api_key=kwargs.get('api_key', self.api_key),
                    api_url=kwargs.get('api_url', self.api_url.chat_url),
                    plugins=kwargs.get("plugins", self.plugins), system_prompt=system_prompt
                ):
                    yield chunk
        else:
            # 添加响应到对话历史
            self.add_to_conversation(full_response, response_role, convo_id=convo_id, total_tokens=total_tokens, pass_history=pass_history)
            self.function_calls_counter = {}

            # 清理翻译引擎相关的历史记录
            if pass_history <= 2 and len(self.conversation[convo_id]) >= 2 \
            and (
                "You are a translation engine" in self.conversation[convo_id][-2]["content"] \
                or "You are a translation engine" in safe_get(self.conversation, convo_id, -2, "content", 0, "text", default="") \
                or "你是一位精通简体中文的专业翻译" in self.conversation[convo_id][-2]["content"] \
                or "你是一位精通简体中文的专业翻译" in safe_get(self.conversation, convo_id, -2, "content", 0, "text", default="")
            ):
                self.conversation[convo_id].pop(-1)
                self.conversation[convo_id].pop(-1)

    def ask_stream(
        self,
        prompt: list,
        role: str = "user",
        convo_id: str = "default",
        model: str = "",
        pass_history: int = 9999,
        function_name: str = "",
        total_tokens: int = 0,
        function_arguments: str = "",
        function_call_id: str = "",
        language: str = "English",
        system_prompt: str = None,
        **kwargs,
    ):
        """
        Ask a question (同步流式响应)
        """
        # 准备会话
        self.system_prompt = system_prompt or self.system_prompt
        if convo_id not in self.conversation or pass_history <= 2:
            self.reset(convo_id=convo_id, system_prompt=system_prompt)
        self.add_to_conversation(prompt, role, convo_id=convo_id, function_name=function_name, total_tokens=total_tokens, function_arguments=function_arguments, function_call_id=function_call_id, pass_history=pass_history)

        # 获取请求体
        json_post = None
        async def get_post_body_async():
            nonlocal json_post
            url, headers, json_post, engine_type = await self.get_post_body(prompt, role, convo_id, model, pass_history, **kwargs)
            return url, headers, json_post, engine_type

        # 替换原来的获取请求体的代码
        # json_post = next(async_generator_to_sync(get_post_body_async()))
        try:
            url, headers, json_post, engine_type = asyncio.run(get_post_body_async())
        except RuntimeError:
            # 如果已经在事件循环中，则使用不同的方法
            loop = asyncio.get_event_loop()
            url, headers, json_post, engine_type = loop.run_until_complete(get_post_body_async())

        self.truncate_conversation(convo_id=convo_id)

        # 打印日志
        if self.print_log:
            print("api_url", kwargs.get('api_url', self.api_url.chat_url), url)
            print("api_key", kwargs.get('api_key', self.api_key))

        # 发送请求并处理响应
        for _ in range(3):
            if self.print_log:
                replaced_text = json.loads(re.sub(r';base64,([A-Za-z0-9+/=]+)', ';base64,***', json.dumps(json_post)))
                print(json.dumps(replaced_text, indent=4, ensure_ascii=False))

            try:
                # 改进处理方式，创建一个内部异步函数来处理异步调用
                async def process_async():
                    # 异步调用 fetch_response_stream
                    async_generator = fetch_response_stream(
                        self.aclient,
                        url,
                        headers,
                        json_post,
                        engine_type,
                        model or self.engine,
                    )
                    # 异步处理响应流
                    async for chunk in self._process_stream_response(
                        async_generator,
                        convo_id=convo_id,
                        function_name=function_name,
                        total_tokens=total_tokens,
                        function_arguments=function_arguments,
                        function_call_id=function_call_id,
                        model=model,
                        language=language,
                        system_prompt=system_prompt,
                        pass_history=pass_history,
                        is_async=True,
                        **kwargs
                    ):
                        yield chunk

                # 将异步函数转换为同步生成器
                return async_generator_to_sync(process_async())
            except ConnectionError:
                print("连接错误，请检查服务器状态或网络连接。")
                return
            except requests.exceptions.ReadTimeout:
                print("请求超时，请检查网络连接或增加超时时间。")
                return
            except httpx.RemoteProtocolError:
                continue
            except Exception as e:
                print(f"发生了未预料的错误：{e}")
                if "Invalid URL" in str(e):
                    e = "您输入了无效的API URL，请使用正确的URL并使用`/start`命令重新设置API URL。具体错误如下：\n\n" + str(e)
                    raise Exception(f"{e}")
                # 最后一次重试失败，向上抛出异常
                if _ == 2:
                    raise Exception(f"{e}")

    async def ask_stream_async(
        self,
        prompt: list,
        role: str = "user",
        convo_id: str = "default",
        model: str = "",
        pass_history: int = 9999,
        function_name: str = "",
        total_tokens: int = 0,
        function_arguments: str = "",
        function_call_id: str = "",
        language: str = "English",
        system_prompt: str = None,
        **kwargs,
    ):
        """
        Ask a question (异步流式响应)
        """
        # 准备会话
        self.system_prompt = system_prompt or self.system_prompt
        if convo_id not in self.conversation or pass_history <= 2:
            self.reset(convo_id=convo_id, system_prompt=system_prompt)
        self.add_to_conversation(prompt, role, convo_id=convo_id, function_name=function_name, total_tokens=total_tokens, function_arguments=function_arguments, pass_history=pass_history, function_call_id=function_call_id)

        # 获取请求体
        url, headers, json_post, engine_type = await self.get_post_body(prompt, role, convo_id, model, pass_history, **kwargs)
        self.truncate_conversation(convo_id=convo_id)

        # 打印日志
        if self.print_log:
            # print("api_url", kwargs.get('api_url', self.api_url.chat_url) == url)
            # print("api_url", kwargs.get('api_url', self.api_url.chat_url))
            print("api_url", url)
            # print("headers", headers)
            print("api_key", kwargs.get('api_key', self.api_key))

        # 发送请求并处理响应
        for _ in range(3):
            if self.print_log:
                replaced_text = json.loads(re.sub(r';base64,([A-Za-z0-9+/=]+)', ';base64,***', json.dumps(json_post)))
                print(json.dumps(replaced_text, indent=4, ensure_ascii=False))

            try:
                # 使用fetch_response_stream处理响应
                generator = fetch_response_stream(
                    self.aclient,
                    url,
                    headers,
                    json_post,
                    engine_type,
                    model or self.engine,
                )
                # if isinstance(chunk, dict) and "error" in chunk:
                #     # 处理错误响应
                #     if chunk["status_code"] in (400, 422, 503):
                #         json_post, should_retry = await self._handle_response_error(
                #             type('Response', (), {'status_code': chunk["status_code"], 'text': json.dumps(chunk["details"]), 'aread': lambda: asyncio.sleep(0)}),
                #             json_post
                #         )
                #         if should_retry:
                #             break  # 跳出内部循环，继续外部循环重试
                #     raise Exception(f"{chunk['status_code']} {chunk['error']} {chunk['details']}")

                # 处理正常响应
                async for processed_chunk in self._process_stream_response(
                    generator,
                    convo_id=convo_id,
                    function_name=function_name,
                    total_tokens=total_tokens,
                    function_arguments=function_arguments,
                    function_call_id=function_call_id,
                    model=model,
                    language=language,
                    system_prompt=system_prompt,
                    pass_history=pass_history,
                    is_async=True,
                    **kwargs
                ):
                    yield processed_chunk

                # 成功处理，跳出重试循环
                break
            except httpx.RemoteProtocolError:
                continue
            except Exception as e:
                print(f"发生了未预料的错误：{e}")
                import traceback
                traceback.print_exc()
                if "Invalid URL" in str(e):
                    e = "您输入了无效的API URL，请使用正确的URL并使用`/start`命令重新设置API URL。具体错误如下：\n\n" + str(e)
                    raise Exception(f"{e}")
                # 最后一次重试失败，向上抛出异常
                if _ == 2:
                    raise Exception(f"{e}")

    async def ask_async(
        self,
        prompt: str,
        role: str = "user",
        convo_id: str = "default",
        model: str = "",
        pass_history: int = 9999,
        **kwargs,
    ) -> str:
        """
        Non-streaming ask
        """
        response = self.ask_stream_async(
            prompt=prompt,
            role=role,
            convo_id=convo_id,
            pass_history=pass_history,
            model=model or self.engine,
            **kwargs,
        )
        full_response: str = "".join([r async for r in response])
        return full_response

    def rollback(self, n: int = 1, convo_id: str = "default") -> None:
        """
        Rollback the conversation
        """
        for _ in range(n):
            self.conversation[convo_id].pop()

    def reset(self, convo_id: str = "default", system_prompt: str = "You are ChatGPT, a large language model trained by OpenAI. Respond conversationally") -> None:
        """
        Reset the conversation
        """
        self.system_prompt = system_prompt or self.system_prompt
        self.latest_file_content = {}
        self.conversation[convo_id] = [
            {"role": "system", "content": self.system_prompt},
        ]
        self.tokens_usage[convo_id] = 0
        self.current_tokens[convo_id] = 0

    def save(self, file: str, *keys: str) -> None:
        """
        Save the Chatbot configuration to a JSON file
        """
        with open(file, "w", encoding="utf-8") as f:
            data = {
                key: self.__dict__[key]
                for key in get_filtered_keys_from_object(self, *keys)
            }
            # saves session.proxies dict as session
            # leave this here for compatibility
            data["session"] = data["proxy"]
            del data["aclient"]
            json.dump(
                data,
                f,
                indent=2,
            )

    def load(self, file: Path, *keys_: str) -> None:
        """
        Load the Chatbot configuration from a JSON file
        """
        with open(file, encoding="utf-8") as f:
            # load json, if session is in keys, load proxies
            loaded_config = json.load(f)
            keys = get_filtered_keys_from_object(self, *keys_)

            if (
                "session" in keys
                and loaded_config["session"]
                or "proxy" in keys
                and loaded_config["proxy"]
            ):
                self.proxy = loaded_config.get("session", loaded_config["proxy"])
                self.session = httpx.Client(
                    follow_redirects=True,
                    proxies=self.proxy,
                    timeout=self.timeout,
                    cookies=self.session.cookies,
                    headers=self.session.headers,
                )
                self.aclient = httpx.AsyncClient(
                    follow_redirects=True,
                    proxies=self.proxy,
                    timeout=self.timeout,
                    cookies=self.session.cookies,
                    headers=self.session.headers,
                )
            if "session" in keys:
                keys.remove("session")
            if "aclient" in keys:
                keys.remove("aclient")
            self.__dict__.update({key: loaded_config[key] for key in keys})

    def _handle_response_error_common(self, response_text, json_post):
        """通用的响应错误处理逻辑，适用于同步和异步场景"""
        try:
            # 检查内容审核失败
            if "Content did not pass the moral check" in response_text:
                return json_post, False, f"内容未通过道德检查：{response_text[:400]}"

            # 处理函数调用相关错误
            if "function calling" in response_text:
                if "tools" in json_post:
                    del json_post["tools"]
                if "tool_choice" in json_post:
                    del json_post["tool_choice"]
                return json_post, True, None

            # 处理请求格式错误
            elif "invalid_request_error" in response_text:
                for index, mess in enumerate(json_post["messages"]):
                    if type(mess["content"]) == list and "text" in mess["content"][0]:
                        json_post["messages"][index] = {
                            "role": mess["role"],
                            "content": mess["content"][0]["text"]
                        }
                return json_post, True, None

            # 处理角色不允许错误
            elif "'function' is not an allowed role" in response_text:
                if json_post["messages"][-1]["role"] == "tool":
                    mess = json_post["messages"][-1]
                    json_post["messages"][-1] = {
                        "role": "assistant",
                        "name": mess["name"],
                        "content": mess["content"]
                    }
                return json_post, True, None

            # 处理服务器繁忙错误
            elif "Sorry, server is busy" in response_text:
                for index, mess in enumerate(json_post["messages"]):
                    if type(mess["content"]) == list and "text" in mess["content"][0]:
                        json_post["messages"][index] = {
                            "role": mess["role"],
                            "content": mess["content"][0]["text"]
                        }
                return json_post, True, None

            # 处理token超限错误
            elif "is not possible because the prompts occupy" in response_text:
                max_tokens = re.findall(r"only\s(\d+)\stokens", response_text)
                if max_tokens:
                    json_post["max_tokens"] = int(max_tokens[0])
                    return json_post, True, None

            # 默认移除工具相关设置
            else:
                if "tools" in json_post:
                    del json_post["tools"]
                if "tool_choice" in json_post:
                    del json_post["tool_choice"]
                return json_post, True, None

        except Exception as e:
            print(f"处理响应错误时出现异常: {e}")
            return json_post, False, str(e)

    def _handle_response_error_sync(self, response, json_post):
        """处理API响应错误并相应地修改请求体（同步版本）"""
        response_text = response.text

        # 处理空响应
        if response.status_code == 200 and response_text == "":
            for index, mess in enumerate(json_post["messages"]):
                if type(mess["content"]) == list and "text" in mess["content"][0]:
                    json_post["messages"][index] = {
                        "role": mess["role"],
                        "content": mess["content"][0]["text"]
                    }
            return json_post, True

        json_post, should_retry, error_msg = self._handle_response_error_common(response_text, json_post)

        if error_msg:
            raise Exception(f"{response.status_code} {response.reason} {error_msg}")

        return json_post, should_retry

    async def _handle_response_error(self, response, json_post):
        """处理API响应错误并相应地修改请求体（异步版本）"""
        await response.aread()
        response_text = response.text

        json_post, should_retry, error_msg = self._handle_response_error_common(response_text, json_post)

        if error_msg:
            raise Exception(f"{response.status_code} {response.reason_phrase} {error_msg}")

        return json_post, should_retry