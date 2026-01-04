#!/usr/bin/env python3
# Author: Wang Jinpeng
# Affiliation: MingXing Team, School of Physics and Electronic Information Engineering, Guilin University of Technology
# Email: wangjinpeng@glut.edu.cn
# Created by WangJinPeng, MingXing Team.
# then replace Si with 0/1 Mg neighbors back to Mg, saving both configurations with CSV table output,
# configurable 0/1 neighbor tolerance, and random seed recording
# Date: April 2, 2025

import ase
import numpy as np
from pymatgen.core import Structure
from ase.io import read, write
from pymatgen.io.ase import AseAtomsAdaptor
import random
import math
import multiprocessing
import os
import json
import csv
from collections import Counter

# ============= USER CONFIGURATION AREA =============
INPUT_FILE = "Na16-P1-Mg.xsd"
ORIGINAL_ELEMENT = "Mg"
NEW_ELEMENTS = ["Si"]
PERCENT_CHANGES = [None, 60]
N_CONFIGS = 100
DISTANCE_THRESHOLD = 3.5
NEIGHBOR_ELEMENT = "Mg"
TARGET_DISTRIBUTION = {
    4: 0.15,
    3: 0.52,
    2: 0.33,
    1: 0.0,
    0: 0.0
}
# Simulated Annealing parameters
INITIAL_TEMP = 20000.0
COOLING_RATE = 0.998
MIN_TEMP = 0.001
MAX_ITERATIONS = 30000
TOLERANCE = 0.05  # Tolerance for total replacements and distribution
BAD_NEIGHBOR_TOLERANCE = 0.1  # Tolerance for Si atoms with 0/1 Mg neighbors (e.g., 0.1 = 10%)
BASE_SEED = 42  # Base seed for reproducibility of seed generation
# Output settings
OUTPUT_JSON = "si_mg_distribution.json"
OUTPUT_CSV = "si_mg_distribution.csv"
OUTPUT_DIR = "conf"
# ============= END OF CONFIGURATION =============

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Created by WangJinPeng, MingXing Team.")
print("This script generates SA configurations and replaces Si with 0/1 Mg neighbors, saving both with CSV table output and random seeds.")
print(f"Input file: {INPUT_FILE}")
print(f"Number of configurations to generate: {N_CONFIGS}")
print(f"Target distribution: {TARGET_DISTRIBUTION}")
print(f"0/1 Mg neighbor tolerance: {BAD_NEIGHBOR_TOLERANCE*100}% of total Si atoms")
print(f"Base seed for seed generation: {BASE_SEED}")

# 载入结构
try:
    ase_structure = read(INPUT_FILE)
    structure = AseAtomsAdaptor.get_structure(ase_structure)
except Exception as e:
    raise Exception(f"Error loading '{INPUT_FILE}': {e}")

original_atoms = [site for site in structure if site.species_string == ORIGINAL_ELEMENT]
num_origin_atoms = len(original_atoms)
if not original_atoms:
    raise ValueError(f"No {ORIGINAL_ELEMENT} atoms found in the structure")
print(f"Number of {ORIGINAL_ELEMENT} atoms in original structure: {num_origin_atoms}")

total_to_replace = int((PERCENT_CHANGES[1] / 100) * num_origin_atoms + 0.5)
target_counts = {
    4: int(total_to_replace * TARGET_DISTRIBUTION[4] + 0.5),
    3: int(total_to_replace * TARGET_DISTRIBUTION[3] + 0.5),
    2: int(total_to_replace * TARGET_DISTRIBUTION[2] + 0.5)
}
while sum(target_counts.values()) < total_to_replace:
    target_counts[3] += 1

print(f"Total atoms to replace: {total_to_replace}")
print(f"Target counts: {target_counts}")
print("\nInitial target counts for Si neighbors:")
for neighbors, count in target_counts.items():
    error_margin = int(total_to_replace * TOLERANCE + 0.5)
    print(f"Si with {neighbors} Mg neighbors: Target = {count}, Tolerance = ±{error_margin}")

# ---------------- 辅助函数定义 ----------------

def get_neighbor_count(struct, site_idx, target_element):
    try:
        site = struct[site_idx]
        neighbors = struct.get_neighbors(site, DISTANCE_THRESHOLD)
        return sum(1 for n in neighbors if n.species_string == target_element)
    except Exception as e:
        print(f"Error calculating neighbors for index {site_idx}: {e}")
        return 0

def compute_energy(struct, substituted_indices, target_dist):
    counts = {4: 0, 3: 0, 2: 0, 1: 0, 0: 0}
    for idx in substituted_indices:
        count = get_neighbor_count(struct, idx, ORIGINAL_ELEMENT)
        counts[count] += 1
    energy = 0
    for k in target_dist:
        diff = counts.get(k, 0) - target_dist[k]
        energy += diff ** 2 * 20
    energy += (counts.get(0, 0) * 100000) + (counts.get(1, 0) * 100000)
    return energy, counts

def initial_substitution(struct, total_to_replace, original, new, target_dist):
    new_struct = struct.copy()
    all_mg_indices = [i for i, site in enumerate(new_struct) if site.species_string == original]
    random.shuffle(all_mg_indices)
    
    substituted_indices = []
    current_counts = {4: 0, 3: 0, 2: 0}
    
    for idx in all_mg_indices:
        temp_struct = new_struct.copy()
        temp_struct[idx] = {new: 1.0}
        count = get_neighbor_count(temp_struct, idx, original)
        if count >= 2 and (count not in target_dist or current_counts.get(count, 0) < target_dist[count]):
            new_struct[idx] = {new: 1.0}
            substituted_indices.append(idx)
            current_counts[count] = current_counts.get(count, 0) + 1
            if len(substituted_indices) >= total_to_replace:
                break

    if len(substituted_indices) < total_to_replace:
        remaining = total_to_replace - len(substituted_indices)
        for idx in all_mg_indices:
            if idx not in substituted_indices and get_neighbor_count(new_struct, idx, original) >= 2:
                new_struct[idx] = {new: 1.0}
                substituted_indices.append(idx)
                remaining -= 1
                if remaining <= 0:
                    break

    if len(substituted_indices) < total_to_replace:
        raise ValueError("Cannot find enough Mg atoms with >= 2 neighbors")
    return new_struct, substituted_indices

def substitute_atoms_with_sa(struct, total_to_replace, original, new, target_dist):
    new_struct, substituted_indices = initial_substitution(struct, total_to_replace, original, new, target_dist)
    current_energy, current_counts = compute_energy(new_struct, substituted_indices, target_dist)
    
    temp = INITIAL_TEMP
    best_struct = new_struct.copy()
    best_indices = substituted_indices.copy()
    best_energy = current_energy
    best_counts = current_counts.copy()
    
    iteration = 0
    all_mg_indices = [i for i, site in enumerate(struct) if site.species_string == original]
    
    while temp > MIN_TEMP and iteration < MAX_ITERATIONS:
        new_struct_trial = new_struct.copy()
        si_idx = random.choice(substituted_indices)
        mg_candidates = [i for i in all_mg_indices if i not in substituted_indices]
        mg_idx = max(mg_candidates, key=lambda x: get_neighbor_count(new_struct, x, original), default=None)
        if mg_idx is None:
            continue
        
        new_struct_trial[si_idx], new_struct_trial[mg_idx] = {original: 1.0}, {new: 1.0}
        new_indices = substituted_indices.copy()
        new_indices.remove(si_idx)
        new_indices.append(mg_idx)
        
        new_energy, new_counts = compute_energy(new_struct_trial, new_indices, target_dist)
        delta_energy = new_energy - current_energy
        
        if new_counts.get(0, 0) + new_counts.get(1, 0) <= current_counts.get(0, 0) + current_counts.get(1, 0) and \
           (delta_energy < 0 or random.random() < math.exp(-delta_energy / temp)):
            new_struct = new_struct_trial
            substituted_indices = new_indices
            current_energy = new_energy
            current_counts = new_counts.copy()
            if current_energy < best_energy:
                best_struct = new_struct.copy()
                best_indices = substituted_indices.copy()
                best_energy = current_energy
                best_counts = current_counts.copy()
        temp *= COOLING_RATE
        iteration += 1

    # Final adjustment to eliminate 0/1 neighbors
    max_adjustments = 2000
    adjustments = 0
    mg_indices = [i for i in all_mg_indices if i not in best_indices]
    while adjustments < max_adjustments:
        bad_indices = [idx for idx in best_indices if get_neighbor_count(best_struct, idx, original) < 2]
        if not bad_indices:
            break
        si_idx = random.choice(bad_indices)
        mg_candidates = [i for i in mg_indices if get_neighbor_count(best_struct, i, original) >= 2]
        if not mg_candidates:
            break
        mg_idx = max(mg_candidates, key=lambda x: get_neighbor_count(best_struct, x, original))
        best_struct[si_idx], best_struct[mg_idx] = {original: 1.0}, {new: 1.0}
        best_indices.remove(si_idx)
        best_indices.append(mg_idx)
        mg_indices.remove(mg_idx)
        mg_indices.append(si_idx)
        adjustments += 1

    final_energy, final_counts = compute_energy(best_struct, best_indices, target_dist)
    print(f"Finished SA for config {iteration} iterations. Best energy: {final_energy}")
    return best_struct, best_indices

# 替换程序的函数
def count_all_si_neighbors(struct, target_element="Mg"):
    si_indices = [i for i, site in enumerate(struct) if site.species_string == "Si"]
    if not si_indices:
        print("No Si atoms found in structure.")
        return {}
    
    si_mg_counts = {}
    for si_idx in si_indices:
        mg_count = get_neighbor_count(struct, si_idx, target_element)
        si_mg_counts[si_idx] = mg_count
    
    return si_mg_counts

def replace_si_with_mg(structure, si_mg_counts, substituted_indices):
    si_to_replace = [idx for idx, count in si_mg_counts.items() if count <= 1]
    if not si_to_replace:
        return structure, substituted_indices, 0
    
    print(f"Replacing {len(si_to_replace)} Si atoms with <= 1 Mg neighbor:")
    for idx in si_to_replace:
        print(f" - Index {idx} (Mg neighbors: {si_mg_counts[idx]})")
        structure[idx] = {ORIGINAL_ELEMENT: 1.0}
        if idx in substituted_indices:
            substituted_indices.remove(idx)
    
    return structure, substituted_indices, len(si_to_replace)

def verify_configuration(struct, substituted_indices, target_counts, total_to_replace, tolerance, bad_neighbor_tolerance):
    final_counts = {4: 0, 3: 0, 2: 0, 1: 0, 0: 0}
    for idx in substituted_indices:
        count = get_neighbor_count(struct, idx, NEIGHBOR_ELEMENT)
        final_counts[count] += 1

    total_substituted = sum(final_counts.values())
    total_ok = abs(total_substituted - total_to_replace) <= total_to_replace * tolerance
    allowed_diff = int(total_to_replace * tolerance + 0.5)
    
    dist_ok = True
    for k in target_counts:
        if abs(final_counts.get(k, 0) - target_counts[k]) > allowed_diff:
            dist_ok = False
            break

    # 检查 0/1 邻居容差
    max_bad_neighbors = max(1, int(total_to_replace * bad_neighbor_tolerance + 0.5))
    bad_neighbor_count = final_counts.get(0, 0) + final_counts.get(1, 0)
    bad_neighbor_ok = bad_neighbor_count <= max_bad_neighbors

    return total_ok and dist_ok and bad_neighbor_ok, final_counts, bad_neighbor_count

def generate_table(counts, total, config_name, seed):
    target_ratios = {4: 0.15, 3: 0.52, 2: 0.33, 1: 0.0, 0: 0.0}
    if total == 0:
        ratios = {k: 0.0 for k in range(5)}
        diffs = {k: target_ratios[k] for k in range(5)}
    else:
        ratios = {k: v / total for k, v in counts.items()}
        diffs = {k: abs(ratios.get(k, 0) - target_ratios[k]) for k in range(5)}
    
    # 格式化表格行
    row = [config_name, seed]
    for k in [4, 3, 2, 1, 0]:
        row.extend([
            counts.get(k, 0),
            f"{ratios.get(k, 0)*100:.2f}%",
            f"{diffs.get(k, 0)*100:.2f}%"
        ])
    return row

# ---------------- 多进程处理函数 ----------------

def process_configuration(config_idx, structure, total_to_replace, original_element, new_elements, target_counts, bad_neighbor_tolerance, base_seed):
    # 设置每个配置的唯一随机种子
    config_seed = base_seed + config_idx
    random.seed(config_seed)
    
    # SA 生成配置
    doped_structure = structure.copy()
    doped_structure, substituted_indices = substitute_atoms_with_sa(doped_structure, total_to_replace, original_element, new_elements[0], target_counts)
    
    # 验证 SA 配置
    sa_valid, sa_counts, sa_bad_neighbors = verify_configuration(doped_structure, substituted_indices, target_counts, total_to_replace, TOLERANCE, bad_neighbor_tolerance)
    sa_total = sum(sa_counts.values())
    sa_table = generate_table(sa_counts, sa_total, f"doped_{config_idx + 1}.xsd", config_seed)
    
    # 保存 SA 配置
    sa_output_file = os.path.join(OUTPUT_DIR, f"doped_{config_idx + 1}.xsd")
    ase_doped_structure = AseAtomsAdaptor.get_atoms(doped_structure)
    write(sa_output_file, ase_doped_structure, format='xsd')
    
    # 替换 Si with 0/1 Mg neighbors
    si_mg_counts = count_all_si_neighbors(doped_structure, original_element)
    modified_structure, modified_indices, num_replaced = replace_si_with_mg(doped_structure.copy(), si_mg_counts, substituted_indices.copy())
    
    # 验证替换后配置
    mod_valid, mod_counts, mod_bad_neighbors = verify_configuration(modified_structure, modified_indices, target_counts, total_to_replace, TOLERANCE, bad_neighbor_tolerance)
    mod_total = sum(mod_counts.values())
    mod_table = generate_table(mod_counts, mod_total, f"doped_{config_idx + 1}_th.xsd", config_seed)
    
    # 保存替换后配置
    mod_output_file = os.path.join(OUTPUT_DIR, f"doped_{config_idx + 1}_th.xsd")
    ase_modified_structure = AseAtomsAdaptor.get_atoms(modified_structure)
    write(mod_output_file, ase_modified_structure, format='xsd')
    
    # 输出信息
    table_rows = [sa_table, mod_table]
    headers = ["Config", "Seed", "4 Count", "4 Ratio", "4 Diff", "3 Count", "3 Ratio", "3 Diff",
               "2 Count", "2 Ratio", "2 Diff", "1 Count", "1 Ratio", "1 Diff",
               "0 Count", "0 Ratio", "0 Diff"]
    table_str = "\n".join([" | ".join(f"{v:<10}" for v in headers)])
    table_str += "\n" + "-" * 190
    for row in table_rows:
        table_str += "\n" + " | ".join(f"{v:<10}" for v in [str(x) for x in row])
    
    print(f"\nConfig {config_idx + 1} (Seed: {config_seed}):")
    print(f"SA Valid: {'Yes' if sa_valid else 'No'}, Total Si: {sa_total} (Target: {total_to_replace}), Bad Neighbors: {sa_bad_neighbors}")
    print(f"Modified Valid: {'Yes' if mod_valid else 'No'}, Total Si: {mod_total}, Replaced Si: {num_replaced}, Bad Neighbors: {mod_bad_neighbors}")
    print(f"SA Config saved to {sa_output_file}")
    print(f"Modified Config saved to {mod_output_file}")
    if not sa_valid or not mod_valid or config_idx < 5:
        print("Statistics Table:")
        print(table_str)
    
    # 统计信息
    stats = {
        "sa": {
            "valid": sa_valid,
            "total_si": sa_total,
            "bad_neighbors": sa_bad_neighbors,
            "distribution": sa_counts,
            "table": sa_table,
            "seed": config_seed
        },
        "modified": {
            "valid": mod_valid,
            "total_si": mod_total,
            "replaced_si": num_replaced,
            "bad_neighbors": mod_bad_neighbors,
            "distribution": mod_counts,
            "table": mod_table,
            "seed": config_seed
        }
    }
    
    return stats

def generate_configurations_parallel(N_CONFIGS, structure, total_to_replace, original_element, new_elements, target_counts, bad_neighbor_tolerance, base_seed):
    results = []
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    
    print(f"\nGenerating and processing {N_CONFIGS} configurations...")
    all_table_rows = []
    headers = ["Config", "Seed", "4 Count", "4 Ratio", "4 Diff", "3 Count", "3 Ratio", "3 Diff",
               "2 Count", "2 Ratio", "2 Diff", "1 Count", "1 Ratio", "1 Diff",
               "0 Count", "0 Ratio", "0 Diff"]
    
    for i in range(N_CONFIGS):
        result = pool.apply_async(process_configuration, (i, structure, total_to_replace, original_element, new_elements, target_counts, bad_neighbor_tolerance, base_seed))
        results.append(result)
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{N_CONFIGS} configurations")
    
    pool.close()
    pool.join()
    
    # 汇总统计
    sa_valid_configs = 0
    mod_valid_configs = 0
    sa_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    mod_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    total_replaced = 0
    total_bad_neighbors = 0
    json_results = {}
    
    for i, result in enumerate(results):
        stats = result.get()
        json_results[f"config_{i + 1}"] = stats
        all_table_rows.append(stats["sa"]["table"])
        all_table_rows.append(stats["modified"]["table"])
        if stats["sa"]["valid"]:
            sa_valid_configs += 1
        if stats["modified"]["valid"]:
            mod_valid_configs += 1
        for k, v in stats["sa"]["distribution"].items():
            sa_counts[k] += v
        for k, v in stats["modified"]["distribution"].items():
            mod_counts[k] += v
        total_replaced += stats["modified"]["replaced_si"]
        total_bad_neighbors += stats["modified"]["bad_neighbors"]
    
    # 保存 CSV 表格
    csv_file = os.path.join(OUTPUT_DIR, OUTPUT_CSV)
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(all_table_rows)
    print(f"Table saved to {csv_file}")
    
    sa_total = sum(sa_counts.values())
    mod_total = sum(mod_counts.values())
    sa_ratios = {k: f"{(v / sa_total)*100:.2f}%" for k, v in sa_counts.items() if v > 0}
    mod_ratios = {k: f"{(v / mod_total)*100:.2f}%" for k, v in mod_counts.items() if v > 0}
    
    # 打印总结
    print(f"\n=== Summary ===")
    print(f"SA Valid configurations: {sa_valid_configs}/{N_CONFIGS} ({sa_valid_configs / N_CONFIGS:.2%})")
    print(f"Modified Valid configurations: {mod_valid_configs}/{N_CONFIGS} ({mod_valid_configs / N_CONFIGS:.2%})")
    print(f"Total Si replaced: {total_replaced}")
    print(f"Total bad neighbors (0/1) in modified configs: {total_bad_neighbors}")
    print(f"SA Average distribution: {sa_counts}")
    print(f"SA Average ratios: {sa_ratios}")
    print(f"Modified Average distribution: {mod_counts}")
    print(f"Modified Average ratios: {mod_ratios}")
    
    # 保存统计到 JSON
    with open(os.path.join(OUTPUT_DIR, OUTPUT_JSON), "w", encoding="utf-8") as f:
        json.dump(json_results, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {os.path.join(OUTPUT_DIR, OUTPUT_JSON)}")
    
    return mod_valid_configs

# ---------------- 开始多进程生成配置 ----------------
if __name__ == "__main__":
    generate_configurations_parallel(N_CONFIGS, structure, total_to_replace, ORIGINAL_ELEMENT, NEW_ELEMENTS, target_counts, BAD_NEIGHBOR_TOLERANCE, BASE_SEED)
