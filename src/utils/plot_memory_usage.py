import re
import matplotlib.pyplot as plt


def parse_memory_log(filepath):
    mvits_mem = []
    sam_mem = []
    clip_mem = []
    steps = []

    with open(filepath, "r") as f:
        lines = f.readlines()

    step = 0
    for line in lines:
        if "mvits_node Memory usage" in line:
            m = re.search(r"mvits_node Memory usage: ([\d\.]+) MB", line)
            if m:
                mvits_mem.append(float(m.group(1)))
                steps.append(step)
                step += 1
        elif "sam_node Memory usage" in line:
            m = re.search(r"sam_node Memory usage: ([\d\.]+) MB", line)
            if m:
                sam_mem.append(float(m.group(1)))
        elif "clip_node Memory usage" in line:
            m = re.search(r"clip_node Memory usage: ([\d\.]+) MB", line)
            if m:
                clip_mem.append(float(m.group(1)))
    return steps, mvits_mem, sam_mem, clip_mem


def plot_memory(steps, mvits_mem, sam_mem, clip_mem):
    plt.figure(figsize=(10, 6))
    plt.plot(steps, mvits_mem, label="mvits_node")
    plt.plot(steps, sam_mem, label="sam_node")
    plt.plot(steps, clip_mem, label="clip_node")
    plt.xlabel("Step")
    plt.ylabel("Memory Usage (MB)")
    plt.title("Node Memory Usage Over Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    log_txt = "memory_log.txt"  # Change to your txt filename
    steps, mvits_mem, sam_mem, clip_mem = parse_memory_log(log_txt)
    plot_memory(steps, mvits_mem, sam_mem, clip_mem)
