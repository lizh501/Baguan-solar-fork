import netCDF4
import numpy as np
import xarray as xr
import os
import glob
from tqdm import tqdm

# --- 配置 ---
himawari_base_dir = './data/Himawari-9/sensor-13'
output_base_dir = './data/himawari_processed_2025'

# 经纬度范围
target_lat_start = 21.2
target_lat_end = 46.8
target_lon_start = 101.2
target_lon_end = 126.8

# 要提取的变量
target_variable_names = ['albedo_03', 'tbb_07', 'tbb_10', 'tbb_14']

os.makedirs(output_base_dir, exist_ok=True)

print(f"扫描 '{himawari_base_dir}' 下的所有 2025 年 Himawari 文件...")

# --- 匹配所有 2025 年、NC_H08/H09、任意分辨率的文件
search_pattern = os.path.join(
    himawari_base_dir,
    '**',
    'NC_H0?_2025????_????_R21_FLDK.*.nc'
)
all_nc_files = glob.glob(search_pattern, recursive=True)

if not all_nc_files:
    print(f"未找到任何匹配 2025 年的 Himawari 文件。")
    exit()

total_files_found = len(all_nc_files)
print(f"共找到 {total_files_found} 个匹配文件。")

processed_count = 0
skipped_count = 0

for file_path in tqdm(all_nc_files, desc="Processing Himawari files", total=total_files_found, unit="file"):
    try:
        file_name = os.path.basename(file_path)
        parts = file_name.split('_')
        if len(parts) < 4:
            skipped_count += 1
            continue

        date_str = parts[2]  # YYYYMMDD
        hour_str = parts[3]  # HHMM

        # OUTPUT 文件名保持原来模式
        output_npy_path = os.path.join(output_base_dir, f'{date_str}_{hour_str}.npy')

        ds = None
        try:
            ds = xr.open_dataset(file_path)

            # 找纬度经度变量
            lat_var_name = None
            lon_var_name = None
            possible_lat_vars = ['latitude', 'nav_lat', 'lat']
            possible_lon_vars = ['longitude', 'nav_lon', 'lon']

            # 先试 nav_lat/nav_lon
            for var in ds.variables:
                if var == 'nav_lat':
                    lat_var_name = var
                elif var == 'nav_lon':
                    lon_var_name = var

            # 再试 latitude/longitude
            if lat_var_name is None or lon_var_name is None:
                for var in ds.variables:
                    if var in possible_lat_vars and lat_var_name is None:
                        lat_var_name = var
                    if var in possible_lon_vars and lon_var_name is None:
                        lon_var_name = var

            if lat_var_name is None or lon_var_name is None:
                skipped_count += 1
                continue

            latitudes_all = ds[lat_var_name]
            longitudes_all = ds[lon_var_name]

            # 提取目标变量
            selected_data_vars = {}
            for target_var in target_variable_names:
                if target_var not in ds.data_vars:
                    continue
                data_array = ds[target_var]
                if not all(dim in data_array.dims for dim in [lat_var_name, lon_var_name]):
                    continue
                selected_data_vars[target_var] = data_array

            if not selected_data_vars:
                skipped_count += 1
                continue

            # 经纬度切片
            lat_indices = np.where((latitudes_all.values >= target_lat_start) & (latitudes_all.values < target_lat_end))[0]
            lon_indices = np.where((longitudes_all.values >= target_lon_start) & (longitudes_all.values < target_lon_end))[0]

            if len(lat_indices) == 0 or len(lon_indices) == 0:
                skipped_count += 1
                continue

            isel_args = {lat_var_name: lat_indices, lon_var_name: lon_indices}
            sliced_data_vars_list = []
            for var_name, data_array in selected_data_vars.items():
                sliced_data = data_array.isel(isel_args)
                sliced_data_vars_list.append(sliced_data)

            final_sliced_dataset = xr.merge(sliced_data_vars_list)

            numpy_arrays_in_order = []
            for var_name in target_variable_names:
                if var_name in final_sliced_dataset.data_vars:
                    numpy_arrays_in_order.append(final_sliced_dataset[var_name].values)

            if not numpy_arrays_in_order:
                skipped_count += 1
            else:
                stacked_numpy_array = np.stack(numpy_arrays_in_order, axis=0)
                np.save(output_npy_path, stacked_numpy_array)
                processed_count += 1

        except Exception:
            skipped_count += 1
        finally:
            if ds is not None:
                ds.close()

    except Exception:
        skipped_count += 1

print("\n--- 处理完成 ---")
print(f"总文件数: {total_files_found}")
print(f"成功处理: {processed_count}")
print(f"跳过: {skipped_count}")