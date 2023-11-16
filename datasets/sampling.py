from torch.utils.data import Subset

def compute_samples_per_subset(sample_count, validate_on_at_least_n_samples):
    validate_on_at_least_n_samples = min(validate_on_at_least_n_samples, sample_count)
    number_subsets = int(sample_count / validate_on_at_least_n_samples)
    samples_per_subset = int(sample_count / number_subsets)
    extra_sample_subsets = sample_count % samples_per_subset
    normal_subsets = number_subsets - extra_sample_subsets
    return samples_per_subset, normal_subsets, extra_sample_subsets

def create_sequential_subsets(dataset, validate_on_at_least_n_samples):
    samples_per_subset, normal_subsets, extra_sample_subsets = compute_samples_per_subset(len(dataset), validate_on_at_least_n_samples)
    subsets = []
    index = 0
    for _ in range(normal_subsets):
        subsets.append(Subset(dataset, range(index, index + samples_per_subset)))
        index += samples_per_subset
    for _ in range(extra_sample_subsets):
        subsets.append(Subset(dataset, range(index, index + samples_per_subset + 1)))
        index += samples_per_subset + 1
    return subsets