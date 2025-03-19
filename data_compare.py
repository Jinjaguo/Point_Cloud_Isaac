import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#####################################
# 1. 定义两个目录：base_dir, compare_dir
#####################################
base_dir = './data/experiments/allegro_screwdriver_diff_only_1./csvgd'
compare_dir = './data/experiments/allegro_screwdriver_diff_init_sdf_guided_1./csvgd'
save_dir = './data/experiments/'

# trial 和 stage 的范围
num_trials = 30
num_stages = 12


#####################################
# 2. 写一个函数，用来读取某个目录下所有 trial/stage 的数据
#####################################
def process_directory(directory, num_trials = num_trials, num_stages = num_stages, num_steps=12):
    """
    读取给定目录下 trial_1 到 trial_N，每个 trial 有 stage_0.csv 到 stage_{num_stages-1}.csv
    每个csv包含3个finger的yaw和sdf数据，每行对应一个step，共12个step
    返回:
        - trail_yaws: list，长度 = num_trials，每个元素是该 trial 的平均 yaw
        - trail_delta_yaws: list，长度 = num_trials，每个元素是该 trial 的平均 delta yaw
        - stepwise_sdf_means: dict, key 为 finger('index','middle','thumb'),
                    value 为一个 list，长度 = 12 (step数)，
                    每个元素是 10 个 trial 中该 step 的 SDF 均值
    """
    trail_yaws = []
    trail_delta_yaws = []

    # 初始化 stepwise_sdf_means，用于存储 12 个 step 的 SDF 平均值
    stepwise_sdf_means = {'index': [[] for _ in range(num_steps)],
                          'middle': [[] for _ in range(num_steps)],
                          'thumb': [[] for _ in range(num_steps)]}

    for trial_idx in range(1, num_trials + 1):
        trial_dir = os.path.join(directory, f'trial_{trial_idx}')

        # 存储当前 trial 下的各 stage 统计
        stage_yaws = []
        stage_delta_yaws = []

        # 先存储每个 step 的 SDF 值，等遍历完 stage 后再求均值
        stepwise_sdf_trial = {'index': [[] for _ in range(num_steps)],
                              'middle': [[] for _ in range(num_steps)],
                              'thumb': [[] for _ in range(num_steps)]}

        for stage_idx in range(num_stages):
            csv_path = os.path.join(trial_dir, f'yaw_sdf_results_{stage_idx}.csv')
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"CSV file not found: {csv_path}")

            df = pd.read_csv(csv_path)

            # 计算 yaw 范围
            yaws = np.abs(df['yaw'].unique())  # shape=(12,)
            mean_yaw_this_stage = np.max(yaws) - np.min(yaws)
            stage_yaws.append(mean_yaw_this_stage)

            # 计算 delta yaw
            delta_yaws = np.abs(np.diff(yaws))  # shape=(11,)
            if len(delta_yaws) > 0:
                mean_delta_this_stage = np.mean(delta_yaws)
            else:
                mean_delta_this_stage = 0.0
            stage_delta_yaws.append(mean_delta_this_stage)

            # 处理 3 个 finger 的 SDF，按 step 累积
            for step_idx in range(num_steps):  # 每个 stage 有 num_steps 个 step
                for finger in ['index', 'middle', 'thumb']:
                    step_sdf_values = df[(df['finger'] == finger)].iloc[step_idx]['sdf']
                    step_sdf = float(step_sdf_values.strip('[]'))
                    stepwise_sdf_trial[finger][step_idx].append(step_sdf)

        # 计算当前 trial 下 12 个 step 的 SDF 平均值，并存入 stepwise_sdf_means
        for step_idx in range(num_steps):
            for finger in ['index', 'middle', 'thumb']:
                mean_sdf_this_step = np.mean(stepwise_sdf_trial[finger][step_idx])
                stepwise_sdf_means[finger][step_idx].append(mean_sdf_this_step)


        # 计算当前 trial 的 yaw 和 delta yaw 均值
        trail_yaws.append(np.mean(stage_yaws))
        trail_delta_yaws.append(np.mean(stage_delta_yaws))

    # # 计算 10 个 trial 下的最终 stepwise_sdf 均值
    # 确保最终存储的数据为 (10, 12)
    final_stepwise_sdf_means = {'index': [[] for _ in range(num_trials)],
                                'middle': [[] for _ in range(num_trials)],
                                'thumb': [[] for _ in range(num_trials)]}

    for finger in ['index', 'middle', 'thumb']:
        for trial_idx in range(num_trials):  # 遍历 10 个 trial
            trial_sdf_means = []
            for step_idx in range(num_steps):  # 遍历 12 个 step
                trial_sdf_means.append(np.mean(stepwise_sdf_means[finger][step_idx][trial_idx]))
            final_stepwise_sdf_means[finger][trial_idx] = trial_sdf_means  # 存入该 trial

    return trail_yaws, trail_delta_yaws, final_stepwise_sdf_means


#####################################
# 3. 分别读取 base_dir 和 compare_dir
#####################################
base_trail_yaws, base_trail_delta_yaws, base_sdf_data = process_directory(base_dir)
print(np.mean(base_trail_yaws))
compare_trail_yaws, compare_trail_delta_yaws, compare_sdf_data = process_directory(compare_dir)
print(np.mean(compare_trail_yaws))
#####################################
# 4. 可视化对比：Yaw & Delta Yaw
#####################################

# -- 4.1 对比 Yaw --
plt.figure(figsize=(8, 5))
plt.bar(range(1, num_trials + 1), base_trail_yaws, label='Base - Yaw', width=0.4, align='center', alpha=0.7)
plt.bar(range(1, num_trials + 1), compare_trail_yaws, label='Compare - Yaw', width=0.4, align='edge', alpha=0.7)
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
plt.bar(range(1, num_trials + 1), base_trail_delta_yaws, label='Base - Yaw', width=0.4, align='center', alpha=0.7)
plt.bar(range(1, num_trials + 1), compare_trail_delta_yaws, label='Compare - Yaw', width=0.4, align='edge', alpha=0.7)
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
num_steps=12
for finger in fingers:
    # 转换数据为 numpy 数组，确保 shape 为 (num_trials, num_steps)
    base_sdf_array = np.array(base_sdf_data[finger])
    compare_sdf_array = np.array(compare_sdf_data[finger])

    if base_sdf_array.ndim == 1:
        base_sdf_array = base_sdf_array.reshape(num_trials, num_steps)
    if compare_sdf_array.ndim == 1:
        compare_sdf_array = compare_sdf_array.reshape(num_trials, num_steps)

    # 计算均值和标准差
    base_mean = base_sdf_array.mean(axis=0)  # shape=(num_steps,)
    base_std = base_sdf_array.std(axis=0)    # shape=(num_steps,)
    compare_mean = compare_sdf_array.mean(axis=0)
    compare_std = compare_sdf_array.std(axis=0)

    # 绘图
    plt.figure(figsize=(8, 5))

    # 画每个 trial 的曲线
    for i in range(num_trials):
        plt.plot(range(num_steps), base_sdf_array[i, :], color='blue', alpha=0.15)
        plt.plot(range(num_steps), compare_sdf_array[i, :], color='red', alpha=0.15)

    # 画 Base 均值和标准差范围
    plt.plot(range(num_steps), base_mean, color='blue', linewidth=2, label='Base Mean')
    plt.fill_between(range(num_steps), base_mean - base_std, base_mean + base_std, color='blue', alpha=0.2)

    # 画 Compare 均值和标准差范围
    plt.plot(range(num_steps), compare_mean, color='red', linewidth=2, label='Compare Mean')
    plt.fill_between(range(num_steps), compare_mean - compare_std, compare_mean + compare_std, color='red', alpha=0.2)

    # 图表设置
    plt.xlabel('Step')
    plt.ylabel('SDF')
    plt.title(f'SDF Comparison for {finger.capitalize()} Finger')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{save_dir}/{finger.capitalize()}_sdf_comparison.png')
    plt.show()
    plt.close()

