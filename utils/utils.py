import subprocess
import cv2
import time
import torch


def get_gpu_usage():
    """Returns a list of tuples containing GPU id and its memory usage."""
    try:
        nvidia_smi_output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"]).decode('utf-8')
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


def histogram_normalization(img, img_mode="RGB"):
    # Histogram normalization in v channel
    if img_mode == "RGB":
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    else:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])

    if img_mode == "RGB":
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    else:
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return img


def l2_norm(inp, axis=1):
    """l2 normalize
    """
    norm = torch.norm(inp, 2, axis, True)
    output = torch.div(inp, norm)
    return output, norm


def fuse_features_with_norm(stacked_embeddings, stacked_norms):
    assert stacked_embeddings.ndim == 3  # (n_features_to_fuse, batch_size, channel)
    assert stacked_norms.ndim == 3  # (n_features_to_fuse, batch_size, 1)

    pre_norm_embeddings = stacked_embeddings * stacked_norms
    fused = pre_norm_embeddings.sum(dim=0)
    fused, fused_norm = l2_norm(fused, axis=1)

    return fused, fused_norm


def get_verification_pairs(file_names):
    """
    Args:
        file_names:

    Returns:

    """
    pairs = []
    for i in range(len(file_names)):
        for j in range(i + 1, len(file_names)):
            if file_names[i].split("_")[0] == file_names[j].split('_')[0]:
                # print(idx, file_names[i], file_names[j], 1)
                pairs.append((i, j, 1))
            else:
                # print(idx, file_names[i], file_names[j], 0)
                pairs.append((i, j, 0))

        # idx += 1
    return pairs


if __name__ == "__main__":
    # gpu_index = get_gpu_with_least_memory_over_period()
    # if gpu_index is not None:
    #     print(f"GPU index with the least amount of resources being used over the last 20 seconds: {gpu_index}")
    # else:
    #     print("Unable to obtain GPU information. Please ensure you have NVIDIA GPUs and nvidia-smi installed.")

    img = cv2.imread(
        "/research/iprobe-sonymd/MSU-SPG-Radiography-Dataset/cropped_images/t1-t5/valid/023/023_PM_Chest_AP2.png")

    import matplotlib.pyplot as plt
