import os
import psutil
import re
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple


def print_mem_usage(prefix=""):
    """打印当前进程的内存使用情况"""
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / 1024**2  # 常驻内存 (MB)
    print(f"{prefix} Memory usage: {mem:.2f} MB")


def parse_memory_log(
    filepath: str, node_names: List[str]
) -> Tuple[List[int], Dict[str, List[float]]]:
    """
    解析内存日志文件，提取每个节点的内存使用数据

    Args:
        filepath: 日志文件路径
        node_names: 要监测的节点名称列表

    Returns:
        steps: 步骤索引列表
        mem_data: 字典，键为节点名，值为该节点的内存使用列表
    """
    # 初始化数据结构
    steps = []
    mem_data = {name: [] for name in node_names}
    step_counter = 0

    with open(filepath, "r") as f:
        for line in f:
            # 检查每一行是否包含任何节点的内存信息
            for node_name in node_names:
                pattern = rf"{node_name} Memory usage: ([\d\.]+) MB"
                match = re.search(pattern, line)
                if match:
                    mem_value = float(match.group(1))
                    mem_data[node_name].append(mem_value)

                    # 如果是第一个节点，增加步骤计数
                    if node_name == node_names[0]:
                        steps.append(step_counter)
                        step_counter += 1
                    break  # 找到匹配后跳出内层循环

    return steps, mem_data


def plot_memory(
    steps: List[int], mem_data: Dict[str, List[float]], node_names: List[str]
):
    """
    绘制内存使用情况图表

    Args:
        steps: 步骤索引列表
        mem_data: 包含各节点内存使用数据的字典
        node_names: 要绘制的节点名称列表
    """
    # 确保所有数据长度一致
    min_length = min(len(steps), *(len(mem_data[name]) for name in node_names))
    steps = steps[:min_length]

    plt.figure(figsize=(12, 7))

    # 为每个节点绘制曲线
    for name in node_names:
        plt.plot(
            steps, mem_data[name][:min_length], label=name, marker="o", markersize=3
        )

    plt.xlabel("Step")
    plt.ylabel("Memory Usage (MB)")
    plt.title("Node Memory Usage Over Time")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 配置参数
    log_file = "mem_log.txt"  # 日志文件路径
    nodes_to_monitor = ["clip_node", "ram_plus_node"]  # 要监测的节点名称列表

    # 解析日志并绘图
    steps, mem_data = parse_memory_log(log_file, nodes_to_monitor)
    plot_memory(steps, mem_data, nodes_to_monitor)
