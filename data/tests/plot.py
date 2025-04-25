# Compares two similarity Tensors and outputs the difference as heatmap.
# https://stackoverflow.com/questions/33282368/plotting-a-2d-heatmap

import torch
import matplotlib.pyplot as plt
import argparse


def load_tensor(path):
    return torch.load(path)


def compare_matrices(t1, t2, threshold=0.01):
    if t1.shape != t2.shape:
        raise ValueError(f"Shape mismatch: {t1.shape} != {t2.shape}")

    diff = torch.abs(t1 - t2)
    mean_diff = diff.mean().item()
    max_diff = diff.max().item()
    num_diff = (diff > threshold).sum().item()
    percent_diff = 100 * num_diff / diff.numel()

    return diff, mean_diff, max_diff, percent_diff


def save_heatmap(diff_tensor, out, title):
    plt.imshow(diff_tensor.numpy(), cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title(title)
    plt.xlabel("Note Index")
    plt.ylabel("Note Index")
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    print(f"Heatmap saved to {out}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare two similarity.pt files")
    parser.add_argument("file1")
    parser.add_argument("file2")
    parser.add_argument("--threshold", type=float, default=0.01)
    parser.add_argument("--out", default="sim_diff_heatmap.png")
    parser.add_argument("--title", default="Similarity Tensor Comparrison")
    args = parser.parse_args()

    sim1 = load_tensor(args.file1)
    sim2 = load_tensor(args.file2)

    diff, mean_diff, max_diff, percent_diff = compare_matrices(
        sim1, sim2, args.threshold)

    print(f"Mean absolute difference: {mean_diff:.6f}")
    print(f"Max absolute difference: {max_diff:.6f}")
    print(f"Percent of elements > {args.threshold}: {percent_diff:.2f}%")

    save_heatmap(diff, args.out, args.title)


if __name__ == "__main__":
    main()
