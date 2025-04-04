import subprocess
from .registry import register_tool

# 执行命令
@register_tool()
def excute_command(command):
    """
    执行命令并返回输出结果
    禁止用于查看pdf

    参数:
        command: 要执行的命令，可以克隆仓库，安装依赖，运行代码等

    返回:
        命令执行的输出结果或错误信息
    """
    try:
        # 使用subprocess.run捕获命令输出
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        # 返回命令的标准输出
        return f"执行命令成功:\n{result.stdout}"
    except subprocess.CalledProcessError as e:
        # 如果命令执行失败，返回错误信息和错误输出
        return f"执行命令失败 (退出码 {e.returncode}):\n错误: {e.stderr}\n输出: {e.stdout}"
    except Exception as e:
        return f"执行命令时发生异常: {e}"