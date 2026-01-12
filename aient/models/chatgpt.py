import os
import re
import json
import copy
import httpx
import asyncio
import logging
import inspect
from collections import defaultdict
from typing import Union, Optional, Callable

from .base import BaseLLM
from ..plugins.registry import registry
from ..plugins import PLUGINS, get_tools_result_async, function_call_list, update_tools_config
from ..utils.scripts import safe_get, async_generator_to_sync, parse_function_xml, parse_continuous_json, convert_functions_to_xml, remove_xml_tags_and_content
from ..core.request import prepare_request_payload
from ..core.response import fetch_response_stream, fetch_response
from ..architext.architext import Messages, SystemMessage, UserMessage, AssistantMessage, ToolCalls, ToolResults, Texts, RoleMessage, Images, Files

class ToolResult(Texts):
    def __init__(self, tool_name: str, tool_args: str, tool_response: str, name: Optional[str] = None, visible: bool = True, newline: bool = True):
        super().__init__(text=tool_response, name=name or f"tool_result_{tool_name}", visible=visible, newline=newline)
        self.tool_name = tool_name
        self.tool_args = tool_args

    async def render(self) -> Optional[str]:
        tool_response = await super().render()
        if tool_response is None:
            tool_response = ""
        if self.tool_args:
            return f"[{self.tool_name}({self.tool_args}) Result]:\n\n{tool_response}"
        else:
            return f"[{self.tool_name} Result]:\n\n{tool_response}"

class APITimeoutError(Exception):
    """Custom exception for API timeout errors."""
    pass

class ValidationError(Exception):
    """Custom exception for response validation errors."""
    def __init__(self, message, response_text):
        super().__init__(message)
        self.response_text = response_text

class EmptyResponseError(Exception):
    """Custom exception for empty API responses."""
    pass

class ModelNotFoundError(Exception):
    """Custom exception for model not found (404) errors."""
    pass

class RateLimitError(Exception):
    """Custom exception for rate limit (429) errors."""
    pass

class BadRequestError(Exception):
    """Custom exception for bad request (400) errors."""
    pass

class HTTPError(Exception):
    """Custom exception for HTTP 500 errors."""
    pass

class InputTokenCountExceededError(Exception):
    """Custom exception for input token count exceeding the maximum."""
    pass

class ConfigurationError(Exception):
    """Custom exception for configuration errors."""
    pass

class RetryFailedError(Exception):
    """Custom exception for retry failures."""
    pass

class TaskComplete(Exception):
    """Exception-like signal to indicate the task is complete."""
    def __init__(self, message):
        self.completion_message = message
        super().__init__(f"Task completed with message: {message}")

# 结尾重复响应错误
class RepetitiveResponseError(Exception):
    """Custom exception for detecting repetitive and meaningless generated strings."""
    def __init__(self, message, phrase, count):
        super().__init__(message)
        self.phrase = phrase
        self.count = count


class AllToolsMissingParametersError(Exception):
    """Custom exception for when all tools are missing required parameters."""
    def __init__(self, message, response_text):
        super().__init__(message)
        self.response_text = response_text


class chatgpt(BaseLLM):
    """
    Official ChatGPT API
    """

    def __init__(
        self,
        api_key: str = None,
        engine: str = os.environ.get("MODEL") or "gpt-4o",
        api_url: str = (os.environ.get("BASE_URL") or "https://api.openai.com/v1/chat/completions"),
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
        cache_messages: list = None,
        logger: logging.Logger = None,
        check_done: bool = False,
        retry_count: int = 999999,
    ) -> None:
        """
        Initialize Chatbot with API key (from https://platform.openai.com/account/api-keys)
        """
        super().__init__(api_key, engine, api_url, system_prompt, proxy, timeout, max_tokens, temperature, top_p, presence_penalty, frequency_penalty, reply_count, truncate_limit, use_plugins=use_plugins, print_log=print_log)
        self.conversation: dict[str, Messages] = defaultdict(lambda: Messages(SystemMessage(self.system_prompt)))
        if cache_messages:
            self.conversation["default"] = cache_messages
        self.function_calls_counter = {}
        self.function_call_max_loop = function_call_max_loop
        self.check_done = check_done
        self.retry_count = retry_count
        if logger:
            self.logger = logger
        else:
            # 如果没有提供 logger，创建一个默认的，它只会打印到控制台
            self.logger = logging.getLogger("chatgpt_default")
            self.logger.propagate = False
            if not self.logger.handlers: # 防止重复添加 handler
                handler = logging.StreamHandler()
                formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
                handler.setFormatter(formatter)
                self.logger.addHandler(handler)
                self.logger.setLevel(logging.INFO if print_log else logging.WARNING)

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
        # self.logger.info(f"role: {role}, function_name: {function_name}, message: {message}, function_arguments: {function_arguments}")
        if convo_id not in self.conversation:
            self.reset(convo_id=convo_id)
        if function_name == "" and message:
            self.conversation[convo_id].append(RoleMessage(role, message))
        elif function_name != "" and message:
            if not (all(value == False for value in self.plugins.values()) or self.use_plugins == False):
                tool_calls = [
                    {
                        "id": function_call_id,
                        "type": "function",
                        "function": {
                            "name": function_name,
                            "arguments": function_arguments,
                        },
                    }
                ]
                self.conversation[convo_id].append(ToolCalls(tool_calls))
                self.conversation[convo_id].append(ToolResults(tool_call_id=function_call_id, content=message))
            else:
                last_user_message = self.conversation[convo_id][-1]
                if last_user_message != UserMessage(message):
                    image_message_list = UserMessage()
                    if isinstance(function_arguments, str):
                        functions_list = json.loads(function_arguments)
                    else:
                        functions_list = function_arguments
                    for tool_info in functions_list:
                        if tool_info.get("base64_image"):
                            image_message_list.extend([
                                safe_get(tool_info, "parameter", "image_path", default="") + " image:",
                                Images(tool_info["base64_image"]),
                            ])
                    self.conversation[convo_id].append(AssistantMessage(convert_functions_to_xml(function_arguments)))
                    if image_message_list:
                        self.conversation[convo_id].append(UserMessage(message + image_message_list, Texts("\n\nYour message **must** end with [done] to signify the end of your output.", name="done", visible=self.check_done)))
                    else:
                        self.conversation[convo_id].append(UserMessage(message, Texts("\n\nYour message **must** end with [done] to signify the end of your output.", name="done", visible=self.check_done)))
                else:
                    self.conversation[convo_id].append(AssistantMessage("我已经执行过这个工具了，接下来我需要做什么？"))
        else:
            self.logger.error(f"error: add_to_conversation message is None or empty, role: {role}, function_name: {function_name}, message: {message}")

        # if self.print_log:
        #     replaced_text = json.loads(re.sub(r';base64,([A-Za-z0-9+/=]+)', ';base64,***', json.dumps(self.conversation[convo_id])))
        #     self.logger.info(json.dumps(replaced_text, indent=4, ensure_ascii=False))

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
                self.logger.info(f"Truncate message: {mess}")
            else:
                break

    async def get_post_body(
        self,
        prompt: str,
        role: str = "user",
        convo_id: str = "default",
        model: str = "",
        pass_history: int = 9999,
        stream: bool = True,
        **kwargs,
    ):
        # 构造 provider 信息
        provider = {
            "provider": "openai",
            "base_url": kwargs.get('api_url', self.api_url.chat_url),
            "api": kwargs.get('api_key', self.api_key),
            "model": [model or self.engine],
            "tools": True if self.use_plugins else False,
            "image": True
        }

        done_message = self.conversation[convo_id].provider("done")
        if done_message:
            done_message.visible = False
            if self.check_done and self.conversation[convo_id][-1][-1].name == "done":
                self.conversation[convo_id][-1][-1].visible = True

        # 构造请求数据
        request_data = {
            "model": model or self.engine,
            "messages": await self.conversation[convo_id].render_latest() if pass_history else Messages(
                SystemMessage(self.system_prompt, self.conversation[convo_id].provider("files")),
                UserMessage(prompt)
            ).render(),
            "stream": stream,
            "temperature": kwargs.get("temperature", self.temperature)
        }
        if stream:
            request_data["stream_options"] = {
                "include_usage": True
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

        # self.logger.info(f"request_data: {json.dumps(request_data, indent=4, ensure_ascii=False)}")

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
        stream: bool = True,
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
                    total_tokens = safe_get(line, "usage", "total_tokens", default=0)
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
                    self.logger.error(f"json.loads error: {repr(line)}")
                    return None

            resp = json.loads(line) if isinstance(line, str) else line
            if "error" in resp:
                raise Exception(json.dumps({"type": "api_error", "details": resp}, ensure_ascii=False))

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

        if not full_response.strip() and not need_function_call:
            raise EmptyResponseError("Response is empty")

        if self.print_log:
            self.logger.info(f"total_tokens: {total_tokens}")

        if response_role is None:
            response_role = "assistant"

        missing_required_params = []

        if self.use_plugins == True:
            if self.check_done:
                # self.logger.info(f"worker Response: {full_response}")
                if not full_response.strip().endswith('[done]'):
                    raise ValidationError("Response is not ended with [done]", response_text=full_response)
                else:
                    full_response = full_response.strip().rstrip('[done]')
            full_response = full_response.replace("<tool_code>", "").replace("</tool_code>", "")
            function_parameter = parse_function_xml(full_response)
            if function_parameter:
                invalid_tools = [tool_dict for tool_dict in function_parameter if tool_dict.get("function_name", "") not in self.plugins.keys()]
                function_parameter = [tool_dict for tool_dict in function_parameter if tool_dict.get("function_name", "") in self.plugins.keys()]

                # Check for missing required parameters
                valid_function_parameters = []
                for tool_dict in function_parameter:
                    tool_name = tool_dict.get("function_name")
                    # tool_name must be in registry.tools, because it is in self.plugins which is from registry.tools
                    func = registry.tools.get(tool_name)
                    if not func:
                        continue

                    sig = inspect.signature(func)
                    provided_params = tool_dict.get("parameter", {})
                    # Ensure provided_params is a dictionary
                    if not isinstance(provided_params, dict):
                        self.logger.warning(f"Parameters for {tool_name} are not a dict: {provided_params}. Skipping.")
                        continue

                    missing_required_params = []
                    for param in sig.parameters.values():
                        # Check if the parameter has no default value and is not in the provided parameters
                        if param.default is inspect.Parameter.empty and param.name not in provided_params:
                            missing_required_params.append(param.name)

                    if not missing_required_params:
                        valid_function_parameters.append(tool_dict)
                    else:
                        if self.print_log:
                            self.logger.warning(
                                f"Skipping tool call for '{tool_name}' due to missing required parameters: {missing_required_params}"
                            )
                            missing_required_params.append(f"Error: {tool_name} missing required parameters: {missing_required_params}")
                function_parameter = valid_function_parameters

                if not function_parameter and missing_required_params:
                    raise AllToolsMissingParametersError("\n\n".join(missing_required_params), response_text=full_response)

                # 删除 task_complete 跟其他工具一起调用的情况，因为 task_complete 必须单独调用
                if len(function_parameter) > 1:
                    function_parameter = [tool_dict for tool_dict in function_parameter if tool_dict.get("function_name", "") != "task_complete"]
                    # 仅当存在其他工具时，才删除 get_task_result
                    if any(tool.get("function_name") != "get_task_result" for tool in function_parameter):
                        function_parameter = [tool_dict for tool_dict in function_parameter if tool_dict.get("function_name", "") != "get_task_result"]
                if len(function_parameter) == 1 and function_parameter[0].get("function_name", "") == "task_complete":
                    raise TaskComplete(safe_get(function_parameter, 0, "parameter", "message", default="The task has been completed."))

                if self.print_log and invalid_tools:
                    self.logger.error(f"invalid_tools: {invalid_tools}")
                    self.logger.error(f"function_parameter: {function_parameter}")
                    self.logger.error(f"full_response: {full_response}")
                if function_parameter:
                    need_function_call = True
                    if isinstance(self.conversation[convo_id][-1]["content"], str) and \
                    "<tool_error>" in self.conversation[convo_id][-1]["content"]:
                        need_function_call = False
                        full_response = remove_xml_tags_and_content(full_response) + "上面是我的分析，还没有实际行动。\n\n接下来我需要做什么？"
                else:
                    need_function_call = False
                    if self.print_log:
                        self.logger.error(f"Failed to parse function_parameter full_response: {full_response}")
                    full_response = ""

        # 处理函数调用
        if need_function_call and self.use_plugins == True:
            if self.print_log:
                if function_parameter:
                    self.logger.info(f"function_parameter: {function_parameter}")
                else:
                    self.logger.info(f"function_full_response: {function_full_response}")

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
                    if self.print_log:
                        self.logger.info(f"Tool use, calling: {tool_name}")

                    # 处理函数调用结果
                    if is_async:
                        async for chunk in get_tools_result_async(
                            tool_name, tool_args, model or self.engine, chatgpt, kwargs.get('api_key', self.api_key),
                            kwargs.get('api_url', self.api_url.chat_url), use_plugins=False, model=model or self.engine,
                            add_message=self.add_to_conversation, convo_id=convo_id, language=language
                        ):
                            yield chunk
                    else:
                        async def run_async():
                            async for chunk in get_tools_result_async(
                                tool_name, tool_args, model or self.engine, chatgpt, kwargs.get('api_key', self.api_key),
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
                    function_parameter = parse_continuous_json(function_full_response, function_call_name, function_call_id)
            except Exception as e:
                self.logger.error(f"解析JSON失败: {e}")
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
            all_responses = UserMessage()

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
                final_tool_response = tool_response
                if "<tool_error>" not in tool_response:
                    if tool_name == "read_file":
                        self.conversation[convo_id].provider("files").update(tool_info['parameter']["file_path"], tool_response, head=safe_get(tool_info, 'parameter', "head", default=None))
                        final_tool_response = "Read file successfully! The file content has been updated in the tag <latest_file_content>."
                    elif tool_name == "get_knowledge_graph_tree":
                        self.conversation[convo_id].provider("knowledge_graph").visible = True
                        final_tool_response = "Get knowledge graph tree successfully! The knowledge graph tree has been updated in the tag <knowledge_graph_tree>."
                    elif tool_name.endswith("goal"):
                        goal_provider = self.conversation[convo_id].provider("goal")
                        if goal_provider:
                            goal_provider += f"\n\n<{tool_name}>{tool_response}</{tool_name}>"
                        final_tool_response = "Get goal successfully! The goal has been updated in the tag <goal>."
                    elif tool_name == "write_to_file":
                        tool_args = None
                    elif tool_name == "read_image":
                        tool_info["base64_image"] = tool_response
                        final_tool_response = "Read image successfully!"
                    elif tool_response.startswith("data:image/") and ";base64," in tool_response:
                        tool_info["base64_image"] = tool_response
                        final_tool_response = "Read image successfully!"
                all_responses.append(ToolResult(tool_name, tool_args, final_tool_response))

            # 合并所有工具响应
            function_response = all_responses
            if missing_required_params:
                function_response.append(Texts("\n\n".join(missing_required_params)))

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
                    plugins=kwargs.get("plugins", self.plugins), system_prompt=system_prompt, stream=stream
                ):
                    yield chunk
            else:
                for chunk in self.ask_stream(
                    function_response, response_role, convo_id=convo_id,
                    function_name=function_call_name, total_tokens=total_tokens,
                    model=model or self.engine, function_arguments=function_full_response,
                    function_call_id=function_call_id, api_key=kwargs.get('api_key', self.api_key),
                    api_url=kwargs.get('api_url', self.api_url.chat_url),
                    plugins=kwargs.get("plugins", self.plugins), system_prompt=system_prompt, stream=stream
                ):
                    yield chunk
        else:
            # 添加响应到对话历史
            self.add_to_conversation(full_response, response_role, convo_id=convo_id, total_tokens=total_tokens, pass_history=pass_history)
            self.function_calls_counter = {}

    async def _ask_stream_handler(
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
        stream: bool = True,
        **kwargs,
    ):
        """
        Unified stream handler (async)
        """
        # 准备会话
        if system_prompt and system_prompt != self.system_prompt:
            self.system_prompt = system_prompt or self.system_prompt
            self.conversation[convo_id][0] = SystemMessage(self.system_prompt)
        if convo_id not in self.conversation or pass_history <= 2:
            self.reset(convo_id=convo_id, system_prompt=self.system_prompt)
        self.add_to_conversation(prompt, role, convo_id=convo_id, function_name=function_name, total_tokens=total_tokens, function_arguments=function_arguments, pass_history=pass_history, function_call_id=function_call_id)

        # 获取请求体
        url, headers, json_post, engine_type = await self.get_post_body(prompt, role, convo_id, model, pass_history, stream=stream, **kwargs)
        self.truncate_conversation(convo_id=convo_id)

        # 打印日志
        if self.print_log:
            self.logger.debug(f"api_url: {kwargs.get('api_url', self.api_url.chat_url)}")
            self.logger.debug(f"api_key: {kwargs.get('api_key', self.api_key)}")
        need_done_prompt = False

        # 发送请求并处理响应
        retry_times = 0
        error_to_raise = None
        while retry_times < self.retry_count:
            retry_times += 1
            tmp_post_json = copy.deepcopy(json_post)
            if need_done_prompt:
                tmp_post_json["messages"].extend(need_done_prompt)
            if self.print_log:
                replaced_text = json.loads(re.sub(r';base64,([A-Za-z0-9+/=]+)', ';base64,***', json.dumps(tmp_post_json)))
                replaced_text_str = json.dumps(replaced_text, indent=4, ensure_ascii=False)
                self.logger.info(f"Request Body:\n{replaced_text_str}")

            try:
                if prompt and "</" in prompt and "<instructions>" not in prompt and convert_functions_to_xml(parse_function_xml(prompt)).strip() == prompt:
                    tmp_response = {
                        "id": "chatcmpl-zXCi5TxWy953TCcxFocSienhvx0BB",
                        "object": "chat.completion.chunk",
                        "created": 1754588695,
                        "model": model or self.engine,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"role": "assistant", "content": prompt},
                                "finish_reason": "stop",
                            }
                        ],
                        "system_fingerprint": "fp_d576307f90",
                    }
                    async def _mock_response_generator():
                        yield f"data: {json.dumps(tmp_response)}\n\n"
                    generator = _mock_response_generator()
                else:
                    if stream:
                        generator = fetch_response_stream(
                            self.aclient, url, headers, tmp_post_json, engine_type, model or self.engine,
                        )
                    else:
                        generator = fetch_response(
                            self.aclient, url, headers, tmp_post_json, engine_type, model or self.engine,
                        )

                # 处理正常响应
                index = 0
                async for processed_chunk in self._process_stream_response(
                    generator, convo_id=convo_id, function_name=function_name,
                    total_tokens=total_tokens, function_arguments=function_arguments,
                    function_call_id=function_call_id, model=model, language=language,
                    system_prompt=system_prompt, pass_history=pass_history, is_async=True, stream=stream, **kwargs
                ):
                    if index == 0:
                        if "HTTP Error', 'status_code': 524" in processed_chunk:
                            raise APITimeoutError("Response timeout")
                        if "HTTP Error', 'status_code': 404" in processed_chunk:
                            raise ModelNotFoundError(f"Model: {model or self.engine} not found!")
                        if "HTTP Error', 'status_code': 429" in processed_chunk:
                            raise RateLimitError(f"Rate limit exceeded for model: {model or self.engine}")
                        if "HTTP Error', 'status_code': 413" in processed_chunk:
                            raise InputTokenCountExceededError(processed_chunk)
                        if "HTTP Error', 'status_code': 400" in processed_chunk:
                            raise BadRequestError(f"Bad Request: {processed_chunk}")
                        if "HTTP Error', 'status_code': " in processed_chunk:
                            raise HTTPError(f"HTTP Error: {processed_chunk}")
                    yield processed_chunk
                    index += 1

                # 成功处理，跳出重试循环
                break
            except (httpx.ConnectError, httpx.ReadTimeout, httpx.PoolTimeout):
                self.logger.error("Connection or read timeout.")
                return # Stop iteration
            except httpx.RemoteProtocolError:
                continue
            except httpx.ReadError as e:
                self.logger.warning(f"{e}, retrying...")
                continue
            except APITimeoutError:
                self.logger.warning("API response timeout (524), retrying...")
                continue
            except HTTPError as e:
                self.logger.warning(f"{e}, retrying...")
                continue
            except RateLimitError as e:
                self.logger.warning(f"{e}, retrying...")
                continue
            except InputTokenCountExceededError as e:
                self.logger.error(f"The request body is too long: {e}")
                error_to_raise = e
                break
            except BadRequestError as e:
                self.logger.error(f"Bad request error: {e}")
                raise
            except GeneratorExit:
                self.logger.warning("GeneratorExit caught, closing stream.")
                break
            except ValidationError as e:
                self.logger.warning(f"Validation failed: {e}. Retrying with corrective prompt.")
                need_done_prompt = [
                    {"role": "assistant", "content": e.response_text},
                    {"role": "user", "content": "你的消息没有以[done]结尾，请重新输出"}
                ]
                continue
            except AllToolsMissingParametersError as e:
                self.logger.warning(f"All tools are missing required parameters: {e}. Retrying with corrective prompt.")
                need_done_prompt = [
                    {"role": "assistant", "content": e.response_text},
                    {"role": "user", "content": f"{str(e)}，请重新输出"}
                ]
                continue
            except EmptyResponseError as e:
                self.logger.warning(f"{e}, retrying...")
                continue
            except RepetitiveResponseError as e:
                self.logger.warning(f"{e}, retrying...")
                continue
            except TaskComplete as e:
                raise
            except ModelNotFoundError as e:
                raise
            except Exception as e:
                self.logger.error(f"{e}")
                import traceback
                self.logger.error(traceback.format_exc())
                if "Invalid URL" in str(e):
                    error_message = "您输入了无效的API URL，请使用正确的URL并使用`/start`命令重新设置API URL。具体错误如下：\n\n" + str(e)
                    raise ConfigurationError(error_message)
                # 最后一次重试失败，向上抛出异常
                if retry_times == self.retry_count:
                    raise RetryFailedError(str(e))

        if error_to_raise:
            raise error_to_raise

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
        stream: bool = True,
        **kwargs,
    ):
        """
        Ask a question (同步流式响应)
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        async_gen = self._ask_stream_handler(
            prompt, role, convo_id, model, pass_history, function_name, total_tokens,
            function_arguments, function_call_id, language, system_prompt, stream, **kwargs
        )
        for chunk in async_generator_to_sync(async_gen):
            yield chunk

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
        stream: bool = True,
        **kwargs,
    ):
        """
        Ask a question (异步流式响应)
        """
        async for chunk in self._ask_stream_handler(
            prompt, role, convo_id, model, pass_history, function_name, total_tokens,
            function_arguments, function_call_id, language, system_prompt, stream, **kwargs
        ):
            yield chunk

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
            stream=True,
            **kwargs,
        )
        full_response: str = "".join([r async for r in response])
        return full_response

    def ask(
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
        response = self.ask_stream(
            prompt=prompt,
            role=role,
            convo_id=convo_id,
            pass_history=pass_history,
            model=model or self.engine,
            stream=True,
            **kwargs,
        )
        full_response: str = "".join([r for r in response])
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
        self.conversation[convo_id] = Messages(
            SystemMessage(self.system_prompt, self.conversation[convo_id].provider("files")),
        )
        self.tokens_usage[convo_id] = 0
        self.current_tokens[convo_id] = 0
