import torch

# the max value of a similarity tensor is always 1.0
# this will skew subsequent calculations such as averages and other ranges.
def unbiased_min_max(tensor, note_index) -> tuple[float, float]:
    t = torch.cat((tensor[:note_index], tensor[note_index + 1 :]))

    min = torch.min(t).item()
    max = torch.max(t).item()

    return min, max
