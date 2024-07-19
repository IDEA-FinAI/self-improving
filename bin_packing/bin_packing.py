import numpy as np
import json

datasets = {}

# Data from the OR3 benchmark, containing 20 bin packing instances each with 500 items.
with open('OR3.txt', 'r') as file:
    datasets['OR3'] = json.load(file)

# Data from the Weibull 5k test dataset, containing 5 bin packing instances each with 5,000 items.
with open('Weibull 5k.txt', 'r') as file:
    datasets['Weibull 5k'] = json.load(file)

def l1_bound(items: tuple[int, ...], capacity: int) -> float:
    """Computes L1 lower bound on OPT for bin packing.

    Args:
        items: Tuple of items to pack into bins.
        capacity: Capacity of bins.

    Returns:
        Lower bound on number of bins required to pack items.
    """
    return np.ceil(np.sum(items) / capacity)

def l1_bound_dataset(instances: dict) -> float:
    """Computes the mean L1 lower bound across a dataset of bin packing instances.

    Args:
        instances: Dictionary containing a set of bin packing instances.

    Returns:
        Average L1 lower bound on number of bins required to pack items.
    """
    l1_bounds = []
    for name in instances:
        instance = instances[name]
        l1_bounds.append(l1_bound(instance['items'], instance['capacity']))
    return np.mean(l1_bounds)

opt_num_bins = {}
for name, dataset in datasets.items():
    opt_num_bins[name] = l1_bound_dataset(dataset)

def get_valid_bin_indices(item: float, bins: np.ndarray) -> np.ndarray:
    """Returns indices of bins in which item can fit."""
    return np.nonzero((bins - item) >= 0)[0]

def online_binpack(items: tuple[float, ...], bins: np.ndarray):
    """Performs online binpacking of `items` into `bins`."""
    # Track which items are added to each bin.
    packing = [[] for _ in bins]
    # Add items to bins.
    for item in items:
        # Extract bins that have sufficient space to fit item.
        valid_bin_indices = get_valid_bin_indices(item, bins)
        # Score each bin based on heuristic.
        priorities = priority(item, bins[valid_bin_indices])
        # Add item to bin with highest priority.
        best_bin = valid_bin_indices[np.argmax(priorities)]
        bins[best_bin] -= item
        packing[best_bin].append(item)
    # Remove unused bins from packing.
    packing = [bin_items for bin_items in packing if bin_items]
    return packing, bins

def evaluate(instances: dict) -> float:
    """Evaluate heuristic function on a set of online binpacking instances."""
    # List storing number of bins used for each instance.
    num_bins = []
    # Perform online binpacking for each instance.
    for name in instances:
        instance = instances[name]
        capacity = instance['capacity']
        items = instance['items']
        # Create num_items bins so there will always be space for all items,
        # regardless of packing order. Array has shape (num_items,).
        bins = np.array([capacity for _ in range(instance['num_items'])])
        # Pack items into bins and return remaining capacity in bins_packed, which
        # has shape (num_items,).
        _, bins_packed = online_binpack(items, bins)
        # If remaining capacity in a bin is equal to initial capacity, then it is
        # unused. Count number of used bins.
        num_bins.append((bins_packed != capacity).sum())
    # Score of heuristic function is negative of average number of bins used
    # across instances (as we want to minimize number of bins).
    return -np.mean(num_bins)

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Returns priority with which we want to add item to each bin.

    Args:
        item: Size of item to be added to the bin.
        bins: Array of capacities for each bin.

    Return:
        Array of same size as bins with priority score of each bin.
    """
    return -(bins - item)

for name, dataset in datasets.items():
    avg_num_bins = -evaluate(dataset)
    excess = (avg_num_bins - opt_num_bins[name]) / opt_num_bins[name]
    print(name)
    print(f'\t Average number of bins: {avg_num_bins}')
    print(f'\t Lower bound on optimum: {opt_num_bins[name]}')
    print(f'\t Excess: {100 * excess:.2f}%')

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Heuristic discovered for the OR datasets."""
    def s(bin, item):
        if bin - item <= 2:
            return 4
        elif (bin - item) <= 3:
            return 3
        elif (bin - item) <= 5:
            return 2
        elif (bin - item) <= 7:
            return 1
        elif (bin - item) <= 9:
            return 0.9
        elif (bin - item) <= 12:
            return 0.95
        elif (bin - item) <= 15:
            return 0.97
        elif (bin - item) <= 18:
            return 0.98
        elif (bin - item) <= 20:
            return 0.98
        elif (bin - item) <= 21:
            return 0.98
        else:
            return 0.99

    return np.array([s(b, item) for b in bins])

# Test performance of heuristic on OR3 dataset
avg_num_bins = -evaluate(datasets['OR3'])
excess = (avg_num_bins - opt_num_bins['OR3']) / opt_num_bins['OR3']
print('OR3')
print(f'\t Average number of bins: {avg_num_bins}')
print(f'\t Lower bound on optimum: {opt_num_bins["OR3"]}')
print(f'\t Excess: {100 * excess:.2f}%')

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Heuristic discovered for the Weibull datasets."""
    max_bin_cap = max(bins)
    score = (bins - max_bin_cap)**2 / item + bins**2 / (item**2)
    score += bins**2 / item**3
    score[bins > item] = -score[bins > item]
    score[1:] -= score[:-1]
    return score

# Test performance of heuristic on Weibull 5k dataset
avg_num_bins = -evaluate(datasets['Weibull 5k'])
excess = (avg_num_bins - opt_num_bins['Weibull 5k']) / opt_num_bins['Weibull 5k']
print('Weibull 5k')
print(f'\t Average number of bins: {avg_num_bins}')
print(f'\t Lower bound on optimum: {opt_num_bins["Weibull 5k"]}')
print(f'\t Excess: {100 * excess:.2f}%')

def is_valid_packing(packing, items: list[float], capacity: float) -> bool:
    """Returns whether `packing` is valid.

    Returns whether `packing` is a valid packing of `items` into bins of size
    `capacity`.

    Args:
        packing: Packing of items into bins. List of bins, where each bin contains
            a list of items packed into that bin.
        items: List of item sizes.
        capacity: Capacity of each bin.
    """
    # Check that items in packing are exactly the same as list of input items.
    packed_items = sum(packing, [])  # Join items in each bin into a single list.
    if sorted(packed_items) != sorted(items):
        return False

    # Check that each bin contains less than `capacity` items.
    for bin_items in packing:
        if sum(bin_items) > capacity:
            return False

    return True

for name in datasets['Weibull 5k']:
    instance = datasets['Weibull 5k'][name]
    capacity = instance['capacity']
    items = instance['items']
    # Create num_items bins so there will always be space for all items,
    # regardless of packing order. Array has shape (num_items,).
    bins = np.array([capacity for _ in range(instance['num_items'])])
    # Compute packing under heuristic.
    packing, _ = online_binpack(items, bins)
    # Check that packing is valid.
    assert is_valid_packing(packing, items, capacity)
