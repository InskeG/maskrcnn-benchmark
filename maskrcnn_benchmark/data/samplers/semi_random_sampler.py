import torch
from torch.utils.data.sampler import Sampler

class SemiRandomSubsetSampler(Sampler):
    def __init__(self, fixed_indices, variable_indices, num_noisy_samples=50000, without_replacement=True):
        self.fixed_indices = fixed_indices
        self.variable_indices = variable_indices
        self.num_noisy_samples = num_noisy_samples
        self.without_replacement = without_replacement

    # @property
    # def num_noisy_samples(self):
    #     return self.num_noisy_samples

    def __iter__(self):
        if self.num_noisy_samples < len(self.variable_indices):
            indices = torch.randperm(len(self.variable_indices), dtype=torch.int64)[: self.num_noisy_samples].tolist()
        else:
            indices = torch.randperm(len(self.variable_indices), dtype=torch.int64).tolist()


        indices = [self.variable_indices[i] for i in indices]

        if self.without_replacement:
            for idx in indices:
                self.variable_indices.remove(idx)

        indices.extend(self.fixed_indices)

        return (indices[i] for i in torch.randperm(len(indices)))

    def __len__(self):
        return len(self.fixed_indices) + self.num_noisy_samples