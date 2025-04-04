import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 两个文件夹路径
dirs = {
    'diff_only': './data/experiments/allegro_screwdriver_diff_only_2./csvgd/trial_1',
    'csvto': './data/experiments/allegro_screwdriver_diff_csvto_1./csvgd/trial_1'
}

finger_list = ["index", "middle", "thumb"]
force_components = ["fx", "fy", "fz"]
torque_components = ["tx", "ty", "tz"]

# 用于保存每个 label、每个 finger 的力模和力矩模的均值（共12个文件，每个文件一个值）
results = {
    label: {
        finger: {
            "force_magnitude_mean": [],
            "torque_magnitude_mean": []
        } for finger in finger_list
    } for label in dirs
}

for label, data_dir in dirs.items():
    file_list = sorted([f for f in os.listdir(data_dir) if f.endswith(".csv")])

    for fname in file_list:
        file_path = os.path.join(data_dir, fname)
        df = pd.read_csv(file_path)

        for finger in finger_list:
            df_finger = df[df['finger'] == finger]

            # 力的模、力矩的模（对每个step，计算模长 -> 再求均值）
            force_mags = np.sqrt(df_finger[force_components].pow(2).sum(axis=1))
            torque_mags = np.sqrt(df_finger[torque_components].pow(2).sum(axis=1))

            force_mean = force_mags.mean()
            torque_mean = torque_mags.mean()

            results[label][finger]["force_magnitude_mean"].append(force_mean)
            results[label][finger]["torque_magnitude_mean"].append(torque_mean)


# 绘图
for finger in finger_list:
    # for metric in ["force_magnitude_mean", "torque_magnitude_mean"]:
    for metric in ["force_magnitude_mean"]:
        plt.figure(figsize=(8, 4))
        for label in dirs:
            y = results[label][finger][metric]
            plt.plot(range(1, len(y) + 1), y, marker='o', label=label)
        plt.xlabel("step (1-12)")
        plt.ylabel(f"{finger.capitalize()} {metric.replace('_', ' ').capitalize()}")
        plt.title(f"{finger.capitalize()} {metric.replace('_', ' ').capitalize()} Comparison")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

plt.show()


# compare 文件夹路径
compare_dir = './data/experiments/allegro_screwdriver_diff_csvto_1./csvgd/trial_1'

finger_list = ["index", "middle", "thumb"]
force_components = ["fx", "fy", "fz"]

# 存储每个 finger 的所有力的模值（用于找最小）
finger_force_magnitudes = {finger: [] for finger in finger_list}

# 遍历 compare 文件夹中的所有 CSV
file_list = sorted([f for f in os.listdir(compare_dir) if f.endswith(".csv")])
for fname in file_list:
    df = pd.read_csv(os.path.join(compare_dir, fname))

    for finger in finger_list:
        df_finger = df[df['finger'] == finger]
        force_mags = np.sqrt(df_finger[force_components].pow(2).sum(axis=1))
        finger_force_magnitudes[finger].extend(force_mags.values)

# 输出每个 finger 的最小力模值
print("=== 最小接触信号阈值（compare 文件夹） ===")
for finger in finger_list:
    min_force = np.min(finger_force_magnitudes[finger])
    print(f"{finger.capitalize()} 最小力模长: {min_force:.6f}")
