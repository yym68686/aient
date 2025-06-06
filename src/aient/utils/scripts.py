import os
import json
import requests
import urllib.parse

from ..core.utils import get_image_message

def get_doc_from_url(url):
    filename = urllib.parse.unquote(url.split("/")[-1])
    response = requests.get(url, stream=True)
    with open(filename, 'wb') as f:
        for chunk in response.iter_content(chunk_size=1024):
            f.write(chunk)
    return filename

from io import BytesIO
def get_audio_message(file_bytes):
    try:
        # 创建一个字节流对象
        audio_stream = BytesIO(file_bytes)

        # 直接使用字节流对象进行转录
        import config
        transcript = config.whisperBot.generate(audio_stream)
        # print("transcript", transcript)

        return transcript

    except Exception as e:
        return f"处理音频文件时出错： {str(e)}"

async def Document_extract(docurl, docpath=None, engine_type = None):
    filename = docpath
    text = None
    prompt = None
    if docpath and docurl and "paper.pdf" != docpath:
        filename = get_doc_from_url(docurl)
        docpath = os.getcwd() + "/" + filename
    if filename and filename[-3:] == "pdf":
        from pdfminer.high_level import extract_text
        text = extract_text(docpath)
    if filename and (filename[-3:] == "txt" or filename[-3:] == ".md" or filename[-3:] == ".py" or filename[-3:] == "yml"):
        with open(docpath, 'r') as f:
            text = f.read()
    if text:
        prompt = (
            "Here is the document, inside <document></document> XML tags:"
            "<document>"
            "{}"
            "</document>"
        ).format(text)
    if filename and filename[-3:] == "jpg" or filename[-3:] == "png" or filename[-4:] == "jpeg":
        prompt = await get_image_message(docurl, engine_type)
    if filename and filename[-3:] == "wav" or filename[-3:] == "mp3":
        with open(docpath, "rb") as file:
            file_bytes = file.read()
        prompt = get_audio_message(file_bytes)
        prompt = (
            "Here is the text content after voice-to-text conversion, inside <voice-to-text></voice-to-text> XML tags:"
            "<voice-to-text>"
            "{}"
            "</voice-to-text>"
        ).format(prompt)
    if os.path.exists(docpath):
        os.remove(docpath)
    return prompt

def split_json_strings(input_string):
    # 初始化结果列表和当前 JSON 字符串
    json_strings = []
    current_json = ""
    brace_count = 0

    # 遍历输入字符串的每个字符
    for char in input_string:
        current_json += char
        if char == '{':
            brace_count += 1
        elif char == '}':
            brace_count -= 1

            # 如果花括号配对完成，我们找到了一个完整的 JSON 字符串
            if brace_count == 0:
                # 尝试解析当前 JSON 字符串
                try:
                    json.loads(current_json)
                    json_strings.append(current_json)
                    current_json = ""
                except json.JSONDecodeError:
                    # 如果解析失败，继续添加字符
                    pass
    if json_strings == []:
        json_strings.append(input_string)
    return json_strings

def check_json(json_data):
    while True:
        try:
            result = split_json_strings(json_data)
            if len(result) > 0:
                json_data = result[0]
            json.loads(json_data)
            break
        except json.decoder.JSONDecodeError as e:
            print("JSON error：", e)
            print("JSON body", repr(json_data))
            if "Invalid control character" in str(e):
                json_data = json_data.replace("\n", "\\n")
            elif "Unterminated string starting" in str(e):
                json_data += '"}'
            elif "Expecting ',' delimiter" in str(e):
                json_data =  {"prompt": json_data}
            elif "Expecting ':' delimiter" in str(e):
                json_data = '{"prompt": ' + json.dumps(json_data) + '}'
            elif "Expecting value: line 1 column 1" in str(e):
                if json_data.startswith("prompt: "):
                    json_data = json_data.replace("prompt: ", "")
                json_data = '{"prompt": ' + json.dumps(json_data) + '}'
            else:
                json_data = '{"prompt": ' + json.dumps(json_data) + '}'
    return json_data

def is_surrounded_by_chinese(text, index):
    left_char = text[index - 1]
    if 0 < index < len(text) - 1:
        right_char = text[index + 1]
        return '\u4e00' <= left_char <= '\u9fff' or '\u4e00' <= right_char <= '\u9fff'
    if index == len(text) - 1:
        return '\u4e00' <= left_char <= '\u9fff'
    return False

def replace_char(string, index, new_char):
    return string[:index] + new_char + string[index+1:]

def claude_replace(text):
    Punctuation_mapping = {",": "，", ":": "：", "!": "！", "?": "？", ";": "；"}
    key_list = list(Punctuation_mapping.keys())
    for i in range(len(text)):
        if is_surrounded_by_chinese(text, i) and (text[i] in key_list):
            text = replace_char(text, i, Punctuation_mapping[text[i]])
    return text

def safe_get(data, *keys, default=None):
    for key in keys:
        try:
            data = data[key] if isinstance(data, (dict, list)) else data.get(key)
        except (KeyError, IndexError, AttributeError, TypeError):
            return default
    return data

import asyncio
def async_generator_to_sync(async_gen):
    """
    将异步生成器转换为同步生成器的工具函数

    Args:
        async_gen: 异步生成器函数

    Yields:
        异步生成器产生的每个值
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        async def collect_chunks():
            chunks = []
            async for chunk in async_gen:
                chunks.append(chunk)
            return chunks

        chunks = loop.run_until_complete(collect_chunks())
        for chunk in chunks:
            yield chunk

    except Exception as e:
        print(f"Error during async execution: {e}")
        raise
    finally:
        try:
            # 清理所有待处理的任务
            tasks = [t for t in asyncio.all_tasks(loop) if not t.done()]
            if tasks:
                loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.close()
        except Exception as e:
            print(f"Error during cleanup: {e}")

def parse_tools_from_cursor_prompt(text):
    import json
    import re

    # 从 cursor_prompt 中提取 <tools> 标签内的 JSON 字符串
    tools_match = re.search(r"<tools>\n(.*?)\n</tools>", text, re.DOTALL)
    if tools_match:
        tools_json_string = tools_match.group(1).strip()
        try:
            tools_list_data = json.loads(tools_json_string, strict=False)
            return tools_list_data
        except json.JSONDecodeError as e:
            print(f"解析 JSON 时出错: {e}")
    return []

from dataclasses import dataclass
from typing import List, Callable, Optional, TypeVar, Generic, Union, Dict, Any

# 定义结果类型
@dataclass
class XmlMatcherResult:
    matched: bool
    data: str = ""

# 泛型类型变量，用于 transform 的返回类型
R = TypeVar('R')

class XmlMatcher(Generic[R]):
    def __init__(self,
                 tag_name: str,
                 transform: Optional[Callable[[XmlMatcherResult], R]] = None,
                 position: int = 0):
        self.tag_name: str = tag_name
        self.transform: Optional[Callable[[XmlMatcherResult], R]] = transform
        self.position: int = position

        self.index: int = 0
        self.chunks: List[XmlMatcherResult] = []
        self.cached: List[str] = []
        self.matched: bool = False
        self.state: str = "TEXT"  # "TEXT", "TAG_OPEN", "TAG_CLOSE"
        self.depth: int = 0
        self.pointer: int = 0

    def _collect(self):
        """将缓存的字符收集到 chunks 中"""
        if not self.cached:
            return

        data = "".join(self.cached)
        # 检查是否需要合并到上一个 chunk
        # 仅当当前缓存的匹配状态与上一个 chunk 相同时才合并
        last = self.chunks[-1] if self.chunks else None
        current_matched_state = self.matched if self.state == "TEXT" else (self.depth > 0) # 在标签解析过程中，匹配状态取决于深度

        if last and last.matched == current_matched_state:
            last.data += data
        else:
            # 只有当 data 不为空时才添加新的 chunk
            if data:
                 self.chunks.append(XmlMatcherResult(data=data, matched=current_matched_state))

        self.cached = []

    def _pop(self) -> List[Union[XmlMatcherResult, R]]:
        """返回处理过的 chunks 并清空列表"""
        chunks_to_return = self.chunks
        self.chunks = []
        if not self.transform:
            # 如果没有 transform 函数，直接返回原始结果列表
            # 需要显式类型转换，因为泛型 R 默认为 XmlMatcherResult
            return [chunk for chunk in chunks_to_return] # type: ignore
        # 应用 transform 函数
        return [self.transform(chunk) for chunk in chunks_to_return]

    def _update(self, chunk: str):
        """处理输入字符串块的核心逻辑"""
        for char in chunk:
            current_char_processed = False # 标记当前字符是否已被状态机逻辑处理

            if self.state == "TEXT":
                if char == "<" and (self.pointer >= self.position or self.matched):
                    self._collect()
                    self.state = "TAG_OPEN"
                    self.cached.append(char)
                    self.index = 0 # 重置 index 以开始匹配标签名或跳过空格
                    current_char_processed = True
                # else: 保持在 TEXT 状态，字符将在循环末尾添加到 cached

            elif self.state == "TAG_OPEN":
                self.cached.append(char)
                current_char_processed = True

                tag_name_len = len(self.tag_name)

                # 状态: 刚进入 < 之后
                if self.index == 0:
                    if char == "/":
                        self.state = "TAG_CLOSE"
                        # index 保持 0，准备匹配闭合标签名或跳过空格
                    elif char.isspace():
                        # 跳过 < 后的空格
                        pass # index 保持 0
                    elif char == self.tag_name[0]:
                        # 开始匹配标签名
                        self.index = 1
                    else:
                        # 无效标签开头 (不是 /，不是空格，不是 tag_name[0])
                        self.state = "TEXT"
                        current_char_processed = True
                # 状态: 正在匹配标签名
                elif self.index < tag_name_len:
                    if self.tag_name[self.index] == char:
                        self.index += 1
                    # 允许在标签名匹配过程中遇到空格，视为属性或无效字符处理
                    elif char.isspace():
                         # 遇到空格，表示标签名已结束，进入属性/结束符处理
                         # 将 index 设置为 tag_name_len 以便后续逻辑处理
                         # 但前提是当前 index 确实匹配到了 tag_name
                         # 如果是 <t hink> 这种情况，这里会失败
                         # 为了简化，我们不允许标签名内部有空格，如果需要，逻辑会更复杂
                         # 因此，如果这里遇到空格但 index < tag_name_len，视为无效
                         self.state = "TEXT"
                         current_char_processed = True
                    else:
                        # 字符不匹配标签名
                        self.state = "TEXT"
                        current_char_processed = True
                # 状态: 标签名已完全匹配 (self.index == tag_name_len)
                else: # self.index >= tag_name_len (实际是 ==)
                    if char == ">":
                        # 找到了开始标签的结束符
                        self.state = "TEXT"
                        self.depth += 1
                        self.matched = True
                        self.cached = [] # 清空缓存，丢弃 <tag ...>
                    elif char.isspace():
                        # 忽略标签名后的空格
                        pass # 保持在 TAG_OPEN 状态，等待 > 或属性
                    else:
                        # 字符是属性的一部分，忽略它，继续等待 '>'
                        pass # 保持在 TAG_OPEN 状态

            elif self.state == "TAG_CLOSE":
                self.cached.append(char)
                current_char_processed = True # 默认设为 True

                tag_name_len = len(self.tag_name)

                # 状态: 刚进入 </ 之后
                if self.index == 0:
                    if char.isspace():
                        # 跳过 </ 后的空格
                        pass # index 保持 0
                    elif char == self.tag_name[0]:
                        # 开始匹配标签名
                        self.index = 1
                    else:
                        # 无效闭合标签 (不是空格，不是 tag_name[0])
                        self.state = "TEXT"
                        current_char_processed = True
                # 状态: 正在匹配标签名
                elif self.index < tag_name_len:
                    if self.tag_name[self.index] == char:
                        self.index += 1
                    else:
                        # 字符不匹配标签名
                        self.state = "TEXT"
                        current_char_processed = True
                # 状态: 标签名已完全匹配 (self.index == tag_name_len)
                else: # self.index == tag_name_len
                    if char == ">":
                        # 找到了 '>'
                        was_inside_tag = self.depth > 0
                        self.state = "TEXT" # 无论如何都回到 TEXT 状态

                        if was_inside_tag:
                            # 确实在标签内部，正常处理闭合标签
                            self.depth -= 1
                            self.matched = self.depth > 0
                            self.cached = [] # 清空缓存，丢弃 </tag>
                            # current_char_processed 保持 True
                        else:
                            # 不在标签内部，这是一个无效/意外的闭合标签
                            # 将其视为普通文本，但阻止最后的 > 被添加到缓存
                            # 保留 cached 中已有的 '</tag' 部分，它们将在下次 collect 时作为文本处理
                            current_char_processed = True # 标记 '>' 已处理，防止循环末尾再次添加

                    elif char.isspace():
                        # 允许 </tag >, 继续等待 '>'
                        pass # 保持在 TAG_CLOSE 状态, current_char_processed 保持 True
                    else:
                        # 闭合标签名后出现非空格、非 > 的字符
                        self.state = "TEXT"
                        current_char_processed = True

            # 如果当前字符未被状态机逻辑处理（即应视为普通文本）
            if not current_char_processed:
                # 确保状态是 TEXT
                if self.state != "TEXT":
                     # 如果之前在尝试匹配标签但失败了，缓存的内容应视为文本
                     self.state = "TEXT"

                self.cached.append(char)

            self.pointer += 1

        # 在处理完整个 chunk 后，如果状态是 TEXT，收集剩余缓存
        if self.state == "TEXT":
             self._collect()


    def final(self, chunk: Optional[str] = None) -> List[Union[XmlMatcherResult, R]]:
        """处理最后一块数据并返回所有结果"""
        if chunk:
            self._update(chunk)
        # 确保所有剩余缓存都被收集
        # 即使状态不是 TEXT，也需要收集，以防有未闭合的标签等情况
        self._collect()
        return self._pop()

    def update(self, chunk: str) -> List[Union[XmlMatcherResult, R]]:
        """处理一块数据并返回当前处理的结果"""
        self._update(chunk)
        return self._pop()

def parse_function_xml(xml_content: str, check_line_start: bool = True) -> List[Dict[str, Any]]:
    """
    解析XML格式的函数调用信息，转换为字典数组格式
    只解析倒数两层XML标签，忽略更高层级的XML标签
    当 check_line_start 为 True 时，只解析行首的XML标签。

    参数:
        xml_content: 包含一个或多个函数调用的XML字符串
        check_line_start: 布尔值，指示是否只解析行首的XML标签

    返回:
        包含所有函数调用信息的字典数组，每个字典包含函数名和参数
    """
    result_functions = []

    # 第一步：识别XML中的顶层标签（可能是函数调用）
    position = 0
    while position < len(xml_content):
        # 寻找下一个开始标签
        tag_start = xml_content.find("<", position)
        if tag_start == -1:
            break  # 没有找到更多的标签

        # 新增：如果 check_line_start 为 True，检查标签是否在行首
        # 如果 '<' 不在行首 (即 tag_start > 0 且其前一个字符不是换行符)，
        # 则将其视为普通文本的一部分，移动 position 并继续搜索
        if check_line_start:
            # 检查标签是否在行首，或者行首到标签之间只有空格
            is_start_of_line_or_only_spaces_before = True
            if tag_start > 0:
                # 从 tag_start - 1 向前检查，直到行首或遇到非空格字符
                check_pos = tag_start - 1
                while check_pos >= 0 and xml_content[check_pos] != '\n':
                    if not xml_content[check_pos].isspace():
                        is_start_of_line_or_only_spaces_before = False
                        break
                    check_pos -= 1

            if not is_start_of_line_or_only_spaces_before:
                position = tag_start + 1  # 从 '<' 之后继续搜索
                continue

        # 检查是否是XML标签的开始（不是闭合标签）
        if tag_start + 1 < len(xml_content) and xml_content[tag_start + 1] == '/':
            # 这是一个结束标签，跳过
            position = tag_start + 1
            continue

        # 找到标签的结束位置
        tag_end = xml_content.find(">", tag_start)
        if tag_end == -1:
            break  # 标签未正确关闭

        # 提取标签名
        tag_content = xml_content[tag_start+1:tag_end].strip()
        # 处理可能有属性的情况
        tag_name = tag_content.split()[0] if " " in tag_content else tag_content

        if not tag_name:
            position = tag_end + 1
            continue  # 空标签名，跳过

        # 查找整个标签的起止范围
        full_start_tag = f"<{tag_name}"
        full_end_tag = f"</{tag_name}>"

        # 从当前位置找到开始标签
        start_pos = xml_content.find(full_start_tag, position)
        if start_pos == -1:
            position = tag_end + 1
            continue

        # 找到对应的结束标签
        end_pos = xml_content.find(full_end_tag, start_pos)
        if end_pos == -1:
            # 没有找到结束标签，可能是未闭合标签
            position = tag_end + 1
            continue

        # 标签的内容（不包括开始和结束标签）
        tag_inner_content = xml_content[tag_end+1:end_pos]

        # 如果是普通辅助标签（如tool_call），则在其内部寻找函数调用
        if tag_name in ["tool_call", "function_call", "tool", "function"]:
            # 递归处理内部内容，此时不再检查行首条件
            nested_functions = parse_function_xml(tag_inner_content, check_line_start=False)
            result_functions.extend(nested_functions)
        else:
            # 将当前标签作为函数名，解析其内部标签作为参数
            parameters = {}

            # 解析内部标签作为参数
            param_position = 0
            while param_position < len(tag_inner_content):
                param_tag_start = tag_inner_content.find("<", param_position)
                if param_tag_start == -1:
                    break

                # 跳过闭合标签
                if param_tag_start + 1 < len(tag_inner_content) and tag_inner_content[param_tag_start + 1] == '/':
                    param_position = param_tag_start + 1
                    continue

                param_tag_end = tag_inner_content.find(">", param_tag_start)
                if param_tag_end == -1:
                    break

                # 提取参数名
                param_name = tag_inner_content[param_tag_start+1:param_tag_end].strip()
                if " " in param_name:  # 处理有属性的情况
                    param_name = param_name.split()[0]

                if not param_name:
                    param_position = param_tag_end + 1
                    continue

                # 查找参数标签的结束位置
                param_end_tag = f"</{param_name}>"
                param_end_pos = tag_inner_content.find(param_end_tag, param_tag_end)

                if param_end_pos == -1:
                    # 参数标签未闭合
                    param_position = param_tag_end + 1
                    continue

                # 提取参数值
                param_value = tag_inner_content[param_tag_end+1:param_end_pos].strip()
                parameters[param_name] = param_value

                # 更新位置到当前参数标签之后
                param_position = param_end_pos + len(param_end_tag)

            # 添加解析结果
            result_functions.append({
                'function_name': tag_name,
                'parameter': parameters
            })

        # 更新位置到当前标签之后
        position = end_pos + len(full_end_tag)

    return result_functions

def parse_continuous_json(json_str: str, function_name: str = "") -> List[Dict[str, Any]]:
    """
    解析JSON字符串，无论是单个JSON对象还是多个连续的JSON对象
    都能正确解析并转换为结构化的函数调用格式列表

    Args:
        json_str: JSON字符串，可能是单个JSON对象或多个连续JSON对象
        function_name: 函数名称，默认为空字符串

    Returns:
        包含函数调用信息的字典列表
    """
    if not json_str or not json_str.strip():
        return []

    # 尝试直接解析为单个JSON
    try:
        json_obj = json.loads(json_str)
        tool_id = function_name + "_single" if function_name else "tool_single"
        return [{
            'function_name': function_name or "default_function",
            'parameter': json_obj,
            'function_call_id': tool_id
        }]
    except json.JSONDecodeError:
        # 如果不是单个JSON，尝试解析为连续JSON
        pass

    result = []
    idx = 0
    length = len(json_str)

    while idx < length:
        # 找到JSON对象的开始
        if json_str[idx] != '{':
            idx += 1
            continue

        # 跟踪括号的平衡
        balance = 1
        start = idx
        idx += 1

        # 寻找匹配的右括号
        while idx < length and balance > 0:
            if json_str[idx] == '{':
                balance += 1
            elif json_str[idx] == '}':
                balance -= 1
            idx += 1

        if balance == 0:
            # 提取出一个完整的JSON对象
            json_obj_str = json_str[start:idx]
            try:
                # 解析JSON对象
                json_obj = json.loads(json_obj_str)
                # 构造函数调用信息
                tool_id = function_name + "_" + str(len(result)) if function_name else "tool_" + str(len(result))
                result.append({
                    'function_name': function_name or "default_function",
                    'parameter': json_obj,
                    'function_call_id': tool_id
                })
            except json.JSONDecodeError:
                # 忽略解析错误
                pass

    return result

def convert_functions_to_xml(functions_list):
    """
    将函数调用列表转换为XML格式的字符串

    参数:
        functions_list: 函数调用列表，每个元素是包含function_name和parameter的字典

    返回:
        XML格式的字符串
    """
    xml_result = ""

    if isinstance(functions_list, str):
        try:
            # 提取并解析JSON字符串
            functions_list = json.loads(functions_list)
            # 确保解析结果是列表
            if not isinstance(functions_list, list):
                print(f"提取的工具调用不是列表格式: {functions_list}")
        except json.JSONDecodeError as e:
            print(f"从文本中提取的工具调用JSON解析失败: {e}")

    for func in functions_list:
        # 获取函数名和参数
        function_name = func.get('function_name', '')
        parameters = func.get('parameter', {})

        # 开始函数标签
        xml_result += f"<{function_name}>\n"

        # 添加所有参数
        for param_name, param_value in parameters.items():
            xml_result += f"<{param_name}>{param_value}</{param_name}>\n"

        # 结束函数标签
        xml_result += f"</{function_name}>\n"

    return xml_result

if __name__ == "__main__":

    os.system("clear")
    test_xml = """
✅ 好的，我现在读取 `README.md` 文件。
<tool_call>
<read_file>
<file_path>/Users/yanyuming/Downloads/GitHub/llama3_interpretability_sae/README.md</file_path>
</read_file>
</tool_call>好的，我现在读取 `README.md` 文件。
"""
    # test_xml = """首先使用read_file工具读取论文内容，然后使用excute_command工具克隆代码仓库到本地。\n```xml\n<read_file>\n<file_path>/Users/yanyuming/Downloads/GitHub/OceanSynthesis/papers/2412.06410v1.pdf</file_path>\n</read_file>\n\n<excute_command>\n<command>git clone https://github.com/bartbussmann/BatchTopK.git</command>\n</excute_command>\n```"""
    test_xml = """
✅ 好的，我现在读取 `README.md` 文件。
    <read_file>
    <file_path>README.md</file_path>
    </read_file>
    <read_file>
    <file_path>README.md</file_path>
    </read_file>

<tool_call>
<read_file>
<file_path>README.md</file_path>
</read_file>
</tool_call>
好的，我现在读取 `README.md` 文件。`<answer> </answer>`
"""
    test_xml = """
好的，我来报告上一组指令的执行结果：

1.  **创建日志目录 `/root/ctm-server/ctm_tinyimagenet_logs_v5_drastic_reduction`**:
    <excute_command>
    <command>mkdir -p /root/ctm-server/ctm_tinyimagenet_logs_v5_drastic_reduction</command>
    </excute_command>
    *(我需要等待这个命令的实际执行结果才能确认。假设它会成功，因为 `mkdir -p` 通常不会失败。)*

2.  **执行第五次实验的训练命令**:
    <excute_command>
    <command>python /root/ctm-server/continuous-thought-machines/tasks/image_classification/train.py \
        --model ctm \
        --dataset tinyimagenet \
        --data_root /root/ctm-server/tiny-imagenet-200 \
        --batch_size 1024 \
        --training_iterations 10000 \
        --device 0 \
        --log_dir /root/ctm-server/ctm_tinyimagenet_logs_v5_drastic_reduction \
        --save_every 1000 \
        --track_every 200 \
        --num_workers_train 2 \
        --d_model 128 \
        --iterations 10 \
        --d_input 64 \
        --heads 1 \
        --synapse_depth 1 \
        --n_synch_out 128 \
        --n_synch_action 128 \
        --memory_length 8 \
        --backbone_type resnet18-1 \
        --positional_embedding_type none \
        --dropout 0.5 \
        --weight_decay 1e-3 \
        --use_amp</command>
    </excute_command>
    *(我需要等待这个训练命令的实际执行结果来确认训练是否完成。训练通常需要一些时间。)*

在上述两个 `excute_command` 的结果返回后，并且如果训练确认已完成，我将执行下一步：

3.  **读取第五次实验的准确率图像**:
    <read_image>
    <image_path>/root/ctm-server/ctm_tinyimagenet_logs_v5_drastic_reduction/accuracies.png</image_path>
    </read_image>

请提供前两个 `excute_command` 的执行结果。
"""

    print(parse_function_xml(test_xml))

# 运行本文件：python -m beswarm.aient.src.aient.utils.scripts
