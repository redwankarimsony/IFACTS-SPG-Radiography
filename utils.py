import subprocess
import re
import time

def get_gpu_usage():
    """Returns a list of tuples containing GPU id and its memory usage."""
    try:
        nvidia_smi_output = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"]).decode('utf-8')
        gpu_usages = []
        for line in nvidia_smi_output.strip().split("\n"):
            index, memory = line.split(", ")
            gpu_usages.append((int(index), int(memory)))
        return gpu_usages
    except Exception as e:
        print(f"Error obtaining GPU information: {e}")
        return []

def get_gpu_with_least_memory_over_period(period=20, interval=1):
    """Returns the GPU index with the least average memory usage over a period."""
    samples = int(period / interval)
    accumulated_usages = {}

    for _ in range(samples):
        for index, memory in get_gpu_usage():
            if index not in accumulated_usages:
                accumulated_usages[index] = []
            accumulated_usages[index].append(memory)
        time.sleep(interval)

    avg_usages = {index: sum(usages) / len(usages) for index, usages in accumulated_usages.items()}
    if not avg_usages:
        return None
    
    return min(avg_usages, key=avg_usages.get)


if __name__ == "__main__":
    gpu_index = get_gpu_with_least_memory_over_period()
    if gpu_index is not None:
        print(f"GPU index with the least amount of resources being used over the last 20 seconds: {gpu_index}")
    else:
        print("Unable to obtain GPU information. Please ensure you have NVIDIA GPUs and nvidia-smi installed.")
