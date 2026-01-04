#!/usr/bin/env python3
# Author: Wang Jinpeng
# Affiliation: MingXing Team, School of Physics and Electronic Information Engineering, Guilin University of Technology
# Email: wangjinpeng@glut.edu.cn
# Created by WangJinPeng, MingXing Team.
# Purpose: Replace 68.75% of Mg with Si using Enhanced SA, ensuring max 3 Mg neighbors initially,
# then replace Si with 4 Mg neighbors back to Mg, saving both configurations with CSV table output,
# configurable 0/1 neighbor tolerance, and random seed recording
# Date: May 20, 2025

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
INPUT_FILE = "Na174_554.xsd"
ORIGINAL_ELEMENT = "Mg"
NEW_ELEMENTS = ["Si"]
PERCENT_CHANGES = [None, 68.75]
N_CONFIGS = 40
DISTANCE_THRESHOLD = 3.4
NEIGHBOR_ELEMENT = "Mg"
TARGET_DISTRIBUTION = {
    4: 0.0,  # No Si atoms with 4 Mg neighbors
    3: 0.15,
    2: 0.57,
    1: 0.2,   # Adjusted to ensure 1 + 0 = 0.28, and 1 > 0
    0: 0.08
}
# Simulated Annealing parameters
INITIAL_TEMP = 20000.0
COOLING_RATE = 0.996
MIN_TEMP = 0.001
MAX_ITERATIONS = 50000
# ============= USER CONFIGURATION AREA =============
TOLERANCE = 0.1  # Tolerance for total replacements
DIST_TOLERANCE = 0.3  # Tolerance for neighbor distribution
BAD_NEIGHBOR_TOLERANCE = 0.3  # Tolerance for Si atoms with 0/1 Mg neighbors
BASE_SEED = 0  # Base seed for reproducibility of seed generation
MAX_2_NEIGHBOR_RATIO = 0.57  # Strict limit for Si atoms with 2 Mg neighbors
MAX_3_NEIGHBOR_RATIO = 0.15  # Strict limit for Si atoms with 3 Mg neighbors
# Output settings
FILE_PREFIX = "554_68"  # 文件夹前缀
OUTPUT_JSON = f"{FILE_PREFIX}_distribution.json"
OUTPUT_CSV = f"{FILE_PREFIX}_distribution.csv"
VALID_OUTPUT_CSV = f"{FILE_PREFIX}_valid_distribution.csv"
OUTPUT_DIR = "0_1920"
# ============= END OF CONFIGURATION =============

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Created by WangJinPeng, MingXing Team.")
print("This script generates SA configurations with max 3 Mg neighbors initially, replaces Si with 4 Mg neighbors back to Mg, saving both with CSV table output and random seeds.")
print(f"Input file: {INPUT_FILE}")
print(f"Number of configurations to generate: {N_CONFIGS}")
print(f"Target distribution: {TARGET_DISTRIBUTION}")
print(f"0/1 Mg neighbor tolerance: {BAD_NEIGHBOR_TOLERANCE*100}% of total Si atoms")
print(f"Base seed for seed generation: {BASE_SEED}")
print(f"Output file prefix: {FILE_PREFIX}")

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
    2: int(total_to_replace * TARGET_DISTRIBUTION[2] + 0.5),
    1: int(total_to_replace * TARGET_DISTRIBUTION[1] + 0.5),
    0: int(total_to_replace * TARGET_DISTRIBUTION[0] + 0.5)
}
while sum(target_counts.values()) < total_to_replace:
    target_counts[1] += 1  # Prioritize 1-neighbor to ensure 1 > 0

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

    total_substituted = sum(counts.values())
    energy = 0

    if total_substituted > 0:
        # 基于目标比例偏差的惩罚
        for k in target_dist:
            actual_ratio = counts.get(k, 0) / total_substituted
            target_ratio = target_dist[k]
            diff = actual_ratio - target_ratio
            energy += diff ** 2 * total_substituted * 10000
        # 惩罚 4 配位
        energy += counts.get(4, 0) * 10000  # 高惩罚以避免 4 配位
    return energy, counts

def initial_substitution(struct, total_to_replace, original, new, target_dist):
    new_struct = struct.copy()
    all_mg_indices = [i for i, site in enumerate(new_struct) if site.species_string == original]
    random.shuffle(all_mg_indices)
    
    substituted_indices = []
    current_counts = {3: 0, 2: 0, 1: 0, 0: 0}
    
    # 优先选择 ≤ 3 邻居的 Mg 原子
    for idx in all_mg_indices:
        temp_struct = new_struct.copy()
        temp_struct[idx] = {new: 1.0}
        count = get_neighbor_count(temp_struct, idx, original)
        if count <= 3 and (count not in target_dist or current_counts.get(count, 0) < target_dist[count] * total_to_replace):
            new_struct[idx] = {new: 1.0}
            substituted_indices.append(idx)
            current_counts[count] = current_counts.get(count, 0) + 1
            if len(substituted_indices) >= total_to_replace:
                break
    
    # 如果不足，允许选择任意 Mg 原子
    if len(substituted_indices) < total_to_replace:
        remaining = total_to_replace - len(substituted_indices)
        for idx in all_mg_indices:
            if idx not in substituted_indices:
                new_struct[idx] = {new: 1.0}
                substituted_indices.append(idx)
                remaining -= 1
                if remaining <= 0:
                    break
    
    if len(substituted_indices) < total_to_replace:
        raise ValueError("Cannot find enough Mg atoms to replace")
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
        mg_idx = max(mg_candidates, key=lambda x: get_neighbor_count(new_struct, x, original) if get_neighbor_count(new_struct, x, original) <= 3 else -1, default=None)
        if mg_idx is None:
            continue
        
        new_struct_trial[si_idx], new_struct_trial[mg_idx] = {original: 1.0}, {new: 1.0}
        new_indices = substituted_indices.copy()
        new_indices.remove(si_idx)
        new_indices.append(mg_idx)
        
        new_energy, new_counts = compute_energy(new_struct_trial, new_indices, target_dist)
        delta_energy = new_energy - current_energy
        
        if new_counts.get(4, 0) == 0 and \
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

    # No adjustment for 0/1 neighbors, only ensure max 3 neighbors
    max_adjustments = 2000
    adjustments = 0
    mg_indices = [i for i in all_mg_indices if i not in best_indices]
    while adjustments < max_adjustments:
        bad_indices = [idx for idx in best_indices if get_neighbor_count(best_struct, idx, ORIGINAL_ELEMENT) > 3]
        if not bad_indices:
            break
        si_idx = random.choice(bad_indices)
        mg_candidates = [i for i in mg_indices if get_neighbor_count(best_struct, i, ORIGINAL_ELEMENT) <= 3]
        if not mg_candidates:
            break
        mg_idx = max(mg_candidates, key=lambda x: get_neighbor_count(best_struct, x, ORIGINAL_ELEMENT))
        best_struct[si_idx], best_struct[mg_idx] = {original: 1.0}, {new: 1.0}
        best_indices.remove(si_idx)
        best_indices.append(mg_idx)
        mg_indices.remove(mg_idx)
        mg_indices.append(si_idx)
        adjustments += 1

    final_energy, final_counts = compute_energy(best_struct, best_indices, target_dist)
    print(f"Finished SA for config {iteration} iterations. Best energy: {final_energy}")
    return best_struct, best_indices

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
    # Modified to replace only Si with 4 Mg neighbors
    si_to_replace = [idx for idx, count in si_mg_counts.items() if count == 4]
    if not si_to_replace:
        return structure, substituted_indices, 0
    
    for idx in si_to_replace:
        structure[idx] = {ORIGINAL_ELEMENT: 1.0}
        if idx in substituted_indices:
            substituted_indices.remove(idx)
    
    return structure, substituted_indices, len(si_to_replace)

def verify_configuration(struct, substituted_indices, target_counts, total_to_replace, tolerance, dist_tolerance, bad_neighbor_tolerance):
    final_counts = {4: 0, 3: 0, 2: 0, 1: 0, 0: 0}
    for idx in substituted_indices:
        count = get_neighbor_count(struct, idx, NEIGHBOR_ELEMENT)
        final_counts[count] += 1

    mg_count = sum(1 for site in struct if site.species_string == ORIGINAL_ELEMENT)
    si_count = sum(1 for site in struct if site.species_string == NEW_ELEMENTS[0])
    mg_si_total = mg_count + si_count
    si_ratio = si_count / mg_si_total if mg_si_total > 0 else 0.0
    target_ratio = PERCENT_CHANGES[1] / 100
    total_ok = abs(si_ratio - target_ratio) <= tolerance

    allowed_diff = int(total_to_replace * dist_tolerance + 0.5)
    dist_ok = True
    for k in target_counts:
        if abs(final_counts.get(k, 0) - target_counts[k]) > allowed_diff:
            dist_ok = False
            break

    max_bad_neighbors = max(1, int(total_to_replace * bad_neighbor_tolerance + 0.5))
    bad_neighbor_count = final_counts.get(0, 0) + final_counts.get(1, 0)
    bad_neighbor_ok = bad_neighbor_count <= max_bad_neighbors

    # Ensure no Si atoms with 4 Mg neighbors in final configuration
    four_neighbor_ok = final_counts.get(4, 0) == 0

    return total_ok and dist_ok and bad_neighbor_ok and four_neighbor_ok, final_counts, bad_neighbor_count


def generate_table(counts, total, config_name, seed, struct):
    target_ratios = {4: 0.0, 3: 0.15, 2: 0.57, 1: 0.2, 0: 0.08}
    target_01_ratio = target_ratios[0] + target_ratios[1]  # 0.28
    if total == 0:
        ratios = {k: 0.0 for k in range(5)}
        diffs = {k: target_ratios[k] for k in range(5)}
        ratio_01 = 0.0
        diff_01 = target_01_ratio
    else:
        ratios = {k: v / total for k, v in counts.items()}
        diffs = {k: abs(ratios.get(k, 0) - target_ratios[k]) for k in range(5)}
        ratio_01 = (counts.get(0, 0) + counts.get(1, 0)) / total
        diff_01 = abs(ratio_01 - target_01_ratio)
    
    mg_count = sum(1 for site in struct if site.species_string == ORIGINAL_ELEMENT)
    si_count = sum(1 for site in struct if site.species_string == NEW_ELEMENTS[0])
    mg_si_total = mg_count + si_count
    si_ratio = total / mg_si_total if mg_si_total > 0 else 0.0
    
    # 使用 0+1 配位误差代替 0 和 1 的单独误差
    total_diff = diffs[4] + diffs[3] + diffs[2] + diff_01
    
    row = [config_name, seed]
    for k in [4, 3, 2, 1, 0]:
        row.extend([
            counts.get(k, 0),
            f"{ratios.get(k, 0)*100:.2f}%",
            f"{diffs.get(k, 0)*100:.2f}%"
        ])
    row.extend([
        counts.get(0, 0) + counts.get(1, 0),  # 0+1 Count
        f"{ratio_01*100:.2f}%",              # 0+1 Ratio
        f"{diff_01*100:.2f}%",               # 0+1 Diff
        total,                               # Total Substituted
        f"{si_ratio*100:.2f}%",              # Si/(Mg+Si) Ratio
        f"{total_diff*100:.2f}%"             # Total Diff
    ])
    return row
def process_configuration(config_idx, structure, total_to_replace, original_element, new_elements, target_counts, bad_neighbor_tolerance, base_seed):
    config_seed = base_seed + config_idx
    random.seed(config_seed)
    
    doped_structure = structure.copy()
    doped_structure, substituted_indices = substitute_atoms_with_sa(doped_structure, total_to_replace, original_element, new_elements[0], target_counts)
    
    sa_valid, sa_counts, sa_bad_neighbors = verify_configuration(doped_structure, substituted_indices, target_counts, total_to_replace, TOLERANCE, DIST_TOLERANCE, BAD_NEIGHBOR_TOLERANCE)
    sa_total = sum(sa_counts.values())
    sa_table = generate_table(sa_counts, sa_total, f"{FILE_PREFIX}_seed{config_seed}.cif", config_seed, doped_structure)
    
    sa_output_file = os.path.join(OUTPUT_DIR, f"{FILE_PREFIX}_seed{config_seed}.cif")
    if sa_valid:
        ase_doped_structure = AseAtomsAdaptor.get_atoms(doped_structure)
        write(sa_output_file, ase_doped_structure, format='cif')
    
    si_mg_counts = count_all_si_neighbors(doped_structure, original_element)
    modified_structure, modified_indices, num_replaced = replace_si_with_mg(doped_structure.copy(), si_mg_counts, substituted_indices.copy())
    
    mod_valid, mod_counts, mod_bad_neighbors = verify_configuration(modified_structure, modified_indices, target_counts, total_to_replace, TOLERANCE, DIST_TOLERANCE, BAD_NEIGHBOR_TOLERANCE)
    mod_total = sum(mod_counts.values())
    mod_table = generate_table(mod_counts, mod_total, f"{FILE_PREFIX}_seed{config_seed}_th.cif", config_seed, modified_structure)
    
    mod_output_file = os.path.join(OUTPUT_DIR, f"{FILE_PREFIX}_seed{config_seed}_th.cif")
    if mod_valid:
        ase_modified_structure = AseAtomsAdaptor.get_atoms(modified_structure)
        write(mod_output_file, ase_modified_structure, format='cif')
    
    headers = ["Config", "Seed", 
                   "4 Count", "4 Ratio", "4 Diff", 
                   "3 Count", "3 Ratio", "3 Diff",
                   "2 Count", "2 Ratio", "2 Diff", 
                   "1 Count", "1 Ratio", "1 Diff",
                   "0 Count", "0 Ratio", "0 Diff",
                   "0+1 Count", "0+1 Ratio", "0+1 Diff",
                   "Total Substituted", "Si/(Mg+Si) Ratio", "Total Diff"]
    table_rows = [sa_table, mod_table]
    table_str = "\n".join([" | ".join(f"{v:<10}" for v in headers)])
    table_str += "\n" + "-" * (10 * len(headers) + 3 * (len(headers) - 1))
    for row in table_rows:
        table_str += "\n" + " | ".join(f"{v:<10}" for v in [str(x) for x in row])
    
    print(f"\n配置 {config_idx + 1} (种子: {config_seed}):")
    print(f"SA 有效: {'是' if sa_valid else '否'}, Si总数: {sa_total} (目标: {total_to_replace}), 坏邻居: {sa_bad_neighbors}")
    print(f"修改后有效: {'是' if mod_valid else '否'}, Si总数: {mod_total}, 替换Si: {num_replaced}, 坏邻居: {mod_bad_neighbors}")
    print(f"SA 配置 {'已保存至 ' + sa_output_file if sa_valid else '未保存（无效配置）'}")
    print(f"修改后配置 {'已保存至 ' + mod_output_file if mod_valid else '未保存（无效配置）'}")
    print("统计表格:")
    print(table_str)
    
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
    
    print(f"\n生成并处理 {N_CONFIGS} 个配置...")
    all_table_rows = []
    valid_table_rows = []
    headers = ["Config", "Seed", 
               "4 Count", "4 Ratio", "4 Diff", 
               "3 Count", "3 Ratio", "3 Diff",
               "2 Count", "2 Ratio", "2 Diff", 
               "1 Count", "1 Ratio", "1 Diff",
               "0 Count", "0 Ratio", "0 Diff",
               "0+1 Count", "0+1 Ratio", "0+1 Diff",
               "Total Substituted", "Si/(Mg+Si) Ratio", "Total Diff"]
    
    for i in range(N_CONFIGS):
        result = pool.apply_async(process_configuration, (i, structure, total_to_replace, original_element, new_elements, target_counts, bad_neighbor_tolerance, base_seed))
        results.append(result)
        if (i + 1) % 100 == 0:
            print(f"已处理 {i + 1}/{N_CONFIGS} 个配置")
    
    pool.close()
    pool.join()
    
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
        if stats["sa"]["valid"] and stats["modified"]["valid"]:
            valid_table_rows.append(stats["sa"]["table"])
            valid_table_rows.append(stats["modified"]["table"])
        for k, v in stats["sa"]["distribution"].items():
            sa_counts[k] += v
        for k, v in stats["modified"]["distribution"].items():
            mod_counts[k] += v
        total_replaced += stats["modified"]["replaced_si"]
        total_bad_neighbors += stats["modified"]["bad_neighbors"]
    
    csv_file = os.path.join(OUTPUT_DIR, OUTPUT_CSV)
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(all_table_rows)
    print(f"所有配置表格已保存至 {csv_file}")
    
    valid_csv_file = os.path.join(OUTPUT_DIR, VALID_OUTPUT_CSV)
    with open(valid_csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(valid_table_rows)
    print(f"有效配置表格已保存至 {valid_csv_file}")
    
    sa_total = sum(sa_counts.values())
    mod_total = sum(mod_counts.values())
    sa_ratios = {k: f"{(v / sa_total)*100:.2f}%" for k, v in sa_counts.items() if v > 0}
    mod_ratios = {k: f"{(v / mod_total)*100:.2f}%" for k, v in mod_counts.items() if v > 0}
    
    print(f"\n=== 总结 ===")
    print(f"SA 有效配置: {sa_valid_configs}/{N_CONFIGS} ({sa_valid_configs / N_CONFIGS:.2%})")
    print(f"修改后有效配置: {mod_valid_configs}/{N_CONFIGS} ({mod_valid_configs / N_CONFIGS:.2%})")
    print(f"总计替换 Si: {total_replaced}")
    print(f"修改后配置中坏邻居 (0/1) 总数: {total_bad_neighbors}")
    print(f"SA 平均分布: {sa_counts}")
    print(f"SA 平均比率: {sa_ratios}")
    print(f"修改后平均分布: {mod_counts}")
    print(f"修改后平均比率: {mod_ratios}")
    
    with open(os.path.join(OUTPUT_DIR, OUTPUT_JSON), "w", encoding="utf-8") as f:
        json.dump(json_results, f, indent=2, ensure_ascii=False)
    print(f"结果已保存至 {os.path.join(OUTPUT_DIR, OUTPUT_JSON)}")
    
    return mod_valid_configs

# ---------------- 开始多进程生成配置 ----------------
if __name__ == "__main__":
    generate_configurations_parallel(N_CONFIGS, structure, total_to_replace, ORIGINAL_ELEMENT, NEW_ELEMENTS, target_counts, BAD_NEIGHBOR_TOLERANCE, BASE_SEED)
