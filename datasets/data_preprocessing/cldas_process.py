import numpy as np
import os
import glob
from tqdm import tqdm

# === 配置路径 ===
cldas_base_dir = './data/cldas'               # CLDAS源目录 (包含日期子目录)
cldas_processed_dir = './data/cldas_processed_2025'  # 输出目录

# 目标 CLDAS 通道 (变量名)
target_regional_vars = ['SSRA', 'TCDC']

# 原始数据的通道顺序
original_regional_vars_order = ['PRS', 'TMP', 'DPT', 'SHU', 'WIU', 'WIV', 'WIN', 'SSRA', 'TCDC']

# 原始数据的地理范围和分辨率
original_lat_max = 53.75
original_lat_min = 16.25
original_lon_min = 70.5
original_lon_max = 139.5
resolution = 0.05

# 目标裁剪的范围
target_lat_end = 46.75
target_lat_start = 21.20
target_lon_start = 101.20
target_lon_end = 126.75

# 创建输出目录
os.makedirs(cldas_processed_dir, exist_ok=True)

# === 递归查找 2025 年的 .npy 文件 ===
print(f"Scanning recursively for 2025 CLDAS .npy files in '{cldas_base_dir}'...")
search_pattern_all = os.path.join(cldas_base_dir, '**', '*.npy')
all_cldas_files = glob.glob(search_pattern_all, recursive=True)

# 只保留 2025 年数据
files_to_process = [f for f in all_cldas_files if os.path.basename(f).startswith('2025')]

if not files_to_process:
    print("No 2025 CLDAS .npy files found")
    exit()

total_files_found = len(files_to_process)
print(f"Found {total_files_found} CLDAS files for 2025 (all times kept)")

# === 生成裁剪索引 ===
num_lats_total = int(round((original_lat_max - original_lat_min) / resolution)) + 1
num_lons_total = int(round((original_lon_max - original_lon_min) / resolution)) + 1

full_lats = np.linspace(original_lat_max, original_lat_min, num_lats_total)
full_lons = np.linspace(original_lon_min, original_lon_max, num_lons_total)

try:
    north_lat_idx = np.where(full_lats <= target_lat_end)[0][0]
    south_lat_idx_start = np.where(full_lats >= target_lat_start)[0][-1]
except IndexError:
    print("Latitude index lookup failed, check latitude range settings")
    exit()

lat_slice_end_idx = south_lat_idx_start
lat_slice_start_idx = max(0, south_lat_idx_start - 511)

try:
    start_lon_idx = np.where(full_lons >= target_lon_start)[0][0]
except IndexError:
    print(f"Cannot find starting longitude {target_lon_start}")
    exit()

end_lon_idx = start_lon_idx + 512

print("\n--- Crop Range Verification ---")
print(f"Latitude: {full_lats[lat_slice_start_idx]:.2f}° - {full_lats[lat_slice_end_idx]:.2f}°")
print(f"Longitude: {full_lons[start_lon_idx]:.2f}° - {full_lons[end_lon_idx-1]:.2f}°")

# 选择目标通道索引
selected_indices = []
for var_name in target_regional_vars:
    try:
        idx = original_regional_vars_order.index(var_name)
        selected_indices.append(idx)
    except ValueError:
        print(f"Target variable {var_name} not found in original channel order")
        exit()

print(f"Target channels {target_regional_vars} indices: {selected_indices}")

# === 处理文件 ===
processed_count = 0
skipped_count = 0

for cldas_file_path in tqdm(files_to_process, desc="Processing CLDAS", total=total_files_found):
    try:
        file_name = os.path.basename(cldas_file_path)
        cldas_data = np.load(cldas_file_path).astype(np.float32)
        
        # 如果数据是 [1,H,W,C] 先去掉 batch 维度
        if cldas_data.ndim == 4 and cldas_data.shape[0] == 1:
            cldas_data = cldas_data[0]

        if cldas_data.shape[2] != len(original_regional_vars_order):
            skipped_count += 1
            continue

        # 选择目标通道
        selected_channels_data = cldas_data[:, :, selected_indices]

        # 裁剪
        cropped_data = selected_channels_data[lat_slice_start_idx:lat_slice_end_idx+1,
                                              start_lon_idx:end_lon_idx, :]

        # 转换维度 [H,W,C] → [C,H,W]
        final_data = np.moveaxis(cropped_data, 2, 0)

        if final_data.shape != (len(target_regional_vars), 512, 512):
            skipped_count += 1
            continue

        # 输出文件名 YYYYMMDD_HHMM.npy
        base_name_without_ext = os.path.splitext(file_name)[0]  # YYYYMMDDHHMM
        output_filename = f'{base_name_without_ext[:8]}_{base_name_without_ext[8:]}.npy'
        output_path = os.path.join(cldas_processed_dir, output_filename)

        np.save(output_path, final_data)
        processed_count += 1

    except Exception:
        skipped_count += 1

# === 结果输出 ===
print(f"Total matching files: {total_files_found}")
print(f"Successfully processed: {processed_count}")
print(f"Skipped: {skipped_count}")
print(f"Results saved to: {cldas_processed_dir}")
