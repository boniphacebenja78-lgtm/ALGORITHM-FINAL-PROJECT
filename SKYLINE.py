import gc
import random
import time
import matplotlib.pyplot as plt


# 1. Baseline Algorithm, O(n^2)
def baseline_skyline(buildings):
    if not buildings:
        return []
    x_coords = sorted(set(x for b in buildings for x in (b[0], b[1])))
    skyline = []
    prev_h = 0
    for x in x_coords:
        max_h = 0
        for L, R, H in buildings:
            if L <= x < R and H > max_h:
                max_h = H
        if max_h != prev_h:
            skyline.append((x, max_h))
            prev_h = max_h
    return skyline


# 2. Divide and Conquer Algorithm, O(n log n)
def merge_skylines(left, right):
    i = j = 0
    h1 = h2 = 0
    prev_h = 0
    merged = []
    n1, n2 = len(left), len(right)

    while i < n1 and j < n2:
        if left[i][0] < right[j][0]:
            x, h1 = left[i]
            i += 1
        elif right[j][0] < left[i][0]:
            x, h2 = right[j]
            j += 1
        else:
            x = left[i][0]
            h1 = left[i][1]
            h2 = right[j][1]
            i += 1
            j += 1
        max_h = h1 if h1 > h2 else h2
        if max_h != prev_h:
            merged.append((x, max_h))
            prev_h = max_h

    while i < n1:
        x, h = left[i]
        i += 1
        if h != prev_h:
            merged.append((x, h))
            prev_h = h

    while j < n2:
        x, h = right[j]
        j += 1
        if h != prev_h:
            merged.append((x, h))
            prev_h = h

    return merged


def dc_skyline(buildings):
    n = len(buildings)
    if n == 0:
        return []
    if n == 1:
        L, R, H = buildings[0]
        return [(L, H), (R, 0)]
    mid = n // 2
    left = dc_skyline(buildings[:mid])
    right = dc_skyline(buildings[mid:])
    return merge_skylines(left, right)


# 3. Synthetically Generated Datasets
def gen_disjoint(n):
    return [(i * 10, i * 10 + 5, random.randint(10, 100)) for i in range(n)]


def gen_overlapping(n):
    return [
        (random.randint(0, 100), random.randint(900, 1000), random.randint(10, 500))
        for _ in range(n)
    ]


def gen_nested(n):
    return [(i, 2 * n + 10 - i, (n - i) * 5 + 10) for i in range(n)]


def gen_random(n):
    return [
        (L := random.randint(0, n * 5), L + random.randint(5, 100), random.randint(10, 500))
        for _ in range(n)
    ]


# 4. Benchmark Laufer und Plotter
def run_benchmarks():
    random.seed(42)
    sizes = [100, 250, 500, 1000, 2000, 3000]
    generators = {
        "Disjoint": gen_disjoint,
        "Heavily Overlapping": gen_overlapping,
        "Nested": gen_nested,
        "Random": gen_random,
    }

    results = {name: {"base": [], "dc": [], "k": []} for name in generators}

    for name, gen in generators.items():
        for n in sizes:
            data = gen(n)
            base_times, dc_times = [], []

            for _ in range(3):
                gc.disable()
                t0 = time.perf_counter()
                res_base = baseline_skyline(data)
                t1 = time.perf_counter()

                t2 = time.perf_counter()
                res_dc = dc_skyline(data)
                t3 = time.perf_counter()
                gc.enable()

                assert res_base == res_dc, f"Correctness failure on {name}, n={n}"
                base_times.append((t1 - t0) * 1000)
                dc_times.append((t3 - t2) * 1000)

            results[name]["base"].append(round(min(base_times), 3))
            results[name]["dc"].append(round(min(dc_times), 3))
            results[name]["k"].append(len(res_dc))

    # Grid 2x2 for Comparing Baseline with Divide and Conquer
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (name, res) in zip(axes.flat, results.items()):
        ax.plot(sizes, res["base"], "o-", label="Baseline O(n²)", color="#d9534f", linewidth=2)
        ax.plot(sizes, res["dc"], "s-", label="Divide & Conquer O(n log n)", color="#0275d8", linewidth=2)
        ax.set_title(f"{name} Buildings", fontsize=11, fontweight="bold")
        ax.set_xlabel("Number of Buildings (n)")
        ax.set_ylabel("Execution Time (ms)")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()

    plt.suptitle("Skyline Algorithm Performance Comparison Across Distributions", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("skyline_benchmark.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    run_benchmarks()