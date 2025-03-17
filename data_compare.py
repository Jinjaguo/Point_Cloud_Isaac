import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#####################################
# 1. 定义两个目录：base_dir, compare_dir
#####################################
base_dir = './data/experiments/allegro_screwdriver_diff_only./csvgd'
compare_dir = './data/experiments/allegro_screwdriver_diff_init_sdf_guided_1./csvgd'
save_dir = './data/experiments/'

# trial 和 stage 的范围
num_trials = 6
num_stages = 12


#####################################
# 2. 写一个函数，用来读取某个目录下所有 trial/stage 的数据
#####################################
def process_directory(directory):
    """
    读取给定目录下 trial_1 到 trial_N，每个 trial 有 stage_0.csv 到 stage_{num_stages-1}.csv
    返回:
        - trail_yaws: list，长度 = num_trials，每个元素是该 trial 的平均 yaw
        - trail_delta_yaws: list，长度 = num_trials，每个元素是该 trial 的平均 delta yaw
        - sdf_data: dict, key 为 finger('index','middle','thumb'),
                    value 为一个 list，长度 = num_trials，
                    每个元素是 shape=(num_stages,) 的数组，对应该 trial 在 12 个 stage 的平均 SDF
    """
    trail_yaws = []
    trail_delta_yaws = []
    sdf_data = {'index': [], 'middle': [], 'thumb': []}

    for trial_idx in range(1, num_trials + 1):
        trial_dir = os.path.join(directory, f'trial_{trial_idx}')

        # 存储当前 trial 下的各 stage 统计
        stage_yaws = []
        stage_delta_yaws = []
        stage_sdf = {'index': [], 'middle': [], 'thumb': []}

        for stage_idx in range(num_stages):
            csv_path = os.path.join(trial_dir, f'yaw_sdf_results_{stage_idx}.csv')
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"CSV file not found: {csv_path}")

            df = pd.read_csv(csv_path)

            # 该 stage 可能有 12 步，每步 3 行(3个 finger)，所以 yaw 有 12 个 unique 值
            yaws = np.abs(df['yaw'].unique()) # shape=(12,)
            mean_yaw_this_stage = np.mean(yaws)
            stage_yaws.append(mean_yaw_this_stage)

            # delta yaw: 对 12 个 yaw 做差分 -> 11 个 delta
            delta_yaws = np.abs(np.diff(yaws)) # shape=(11,)
            if len(delta_yaws) > 0:
                mean_delta_this_stage = np.mean(delta_yaws)
            else:
                # 万一只有一个 yaw，diff 为空
                mean_delta_this_stage = 0.0
            stage_delta_yaws.append(mean_delta_this_stage)

            # 处理 3 个 finger 的 SDF
            for finger in ['index', 'middle', 'thumb']:
                # df 中 finger == 当前 finger 的行
                sdf_values = df[df['finger'] == finger]['sdf'].apply(
                    lambda x: float(x.strip('[]'))
                )
                # 该 stage 下 12 步的 SDF 求均值
                stage_sdf[finger].append(sdf_values.mean())

        # 当前 trial 的 12 个 stage yaw/delta_yaw 的平均
        trail_yaws.append(np.mean(stage_yaws))
        trail_delta_yaws.append(np.mean(stage_delta_yaws))

        # 当前 trial 的 12 个 stage sdf
        for finger in ['index', 'middle', 'thumb']:
            # stage_sdf[finger] 是一个长度=12的列表，每个元素是该 stage 的均值
            sdf_data[finger].append(stage_sdf[finger])

    return trail_yaws, trail_delta_yaws, sdf_data


#####################################
# 3. 分别读取 base_dir 和 compare_dir
#####################################
base_trail_yaws, base_trail_delta_yaws, base_sdf_data = process_directory(base_dir)
compare_trail_yaws, compare_trail_delta_yaws, compare_sdf_data = process_directory(compare_dir)

#####################################
# 4. 可视化对比：Yaw & Delta Yaw
#####################################

# -- 4.1 对比 Yaw --
plt.figure(figsize=(8, 5))
plt.plot(range(1, num_trials + 1), base_trail_yaws, marker='o', label='Base - Yaw')
plt.plot(range(1, num_trials + 1), compare_trail_yaws, marker='x', label='Compare - Yaw')
plt.xlabel('Trial')
plt.ylabel('Mean Yaw')
plt.title('Mean Yaw Comparison (Base vs Compare)')
plt.grid(True)
plt.legend()
plt.savefig(f'{save_dir}/mean_yaw_comparison.png')
plt.show()
plt.close()

# -- 4.2 对比 Delta Yaw --
plt.figure(figsize=(8, 5))
plt.plot(range(1, num_trials + 1), base_trail_delta_yaws, marker='o', label='Base - Delta Yaw')
plt.plot(range(1, num_trials + 1), compare_trail_delta_yaws, marker='x', label='Compare - Delta Yaw')
plt.xlabel('Trial')
plt.ylabel('Mean Delta Yaw')
plt.title('Mean Delta Yaw Comparison (Base vs Compare)')
plt.grid(True)
plt.legend()
plt.savefig(f'{save_dir}/mean_delta_yaw_comparison.png')
plt.show()
plt.close()
#####################################
# 5. 可视化对比：SDF (index, middle, thumb)
#####################################
# 说明：每个 finger 有 shape=(num_trials, num_stages) 的数据
#       我们可以对每个 finger 画一张图，把 base & compare 都叠加。
#####################################
fingers = ['index', 'middle', 'thumb']

for finger in fingers:
    # 转为 np.array 方便处理
    base_sdf_array = np.array(base_sdf_data[finger])  # shape=(num_trials, num_stages)
    compare_sdf_array = np.array(compare_sdf_data[finger])  # shape=(num_trials, num_stages)

    # 计算均值与标准差
    base_mean = base_sdf_array.mean(axis=0)  # shape=(num_stages,)
    base_std = base_sdf_array.std(axis=0)  # shape=(num_stages,)
    compare_mean = compare_sdf_array.mean(axis=0)
    compare_std = compare_sdf_array.std(axis=0)

    plt.figure(figsize=(8, 5))
    # （可选）先画出每个 trial 的曲线，便于观察离散程度
    for i in range(num_trials):
        plt.plot(range(num_stages), base_sdf_array[i, :], color='blue', alpha=0.15)
        plt.plot(range(num_stages), compare_sdf_array[i, :], color='red', alpha=0.15)

    # 再画 mean + std
    plt.plot(range(num_stages), base_mean, color='blue', linewidth=2, label='Base Mean')
    plt.fill_between(
        range(num_stages),
        base_mean - base_std,
        base_mean + base_std,
        color='blue', alpha=0.2
    )

    plt.plot(range(num_stages), compare_mean, color='red', linewidth=2, label='Compare Mean')
    plt.fill_between(
        range(num_stages),
        compare_mean - compare_std,
        compare_mean + compare_std,
        color='red', alpha=0.2
    )

    plt.xlabel('Stage')
    plt.ylabel('SDF')
    plt.title(f'SDF Comparison for {finger.capitalize()} Finger')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{save_dir}/{finger.capitalize()}_sdf_comparison.png')
    plt.show()
    plt.close()

