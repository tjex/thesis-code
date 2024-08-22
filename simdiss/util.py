import torch


def row_means(tensor):
    means = []

    for row in tensor:
        mean = torch.mean(row)
        means.append(mean)

    return means
