import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#####################################
# 1. 定义两个目录：base_dir, compare_dir
#####################################
base_dir = './data/experiments/allegro_screwdriver_diff_only_1./csvgd'
compare_dir = './data/experiments/allegro_screwdriver_diff_init_sdf_guided./csvgd'
save_dir = './data/experiments/'

# trial 和 stage 的范围
num_trials = 7
num_stages = 12


#####################################
# 2. 写一个函数，用来读取某个目录下所有 trial/stage 的数据
#####################################
def process_directory(directory, num_trials=num_trials  , num_stages=num_stages, num_steps=12):
    """
    读取给定目录下 trial_1 到 trial_N，每个 trial 有 stage_0.csv 到 stage_{num_stages-1}.csv
    每个 csv 包含 3 个 finger 的 yaw 和 sdf 数据，每行对应一个 step，共 num_steps 个 step

    返回:
        - stage_yaws: list，长度 = num_trials * num_stages，每个元素是该 stage 的平均 yaw
        - stage_delta_yaws: list，长度 = num_trials * num_stages，每个元素是该 stage 的平均 delta yaw
        - stage_stepwise_sdf_means: dict, key 为 finger ('index','middle','thumb'),
            value 为一个 list，长度 = num_trials * num_stages，每个元素是一个长度为 num_steps 的 list，
            表示该 stage 中每个 step 的 SDF 均值
    """
    stage_yaws = []         # 存储每个 stage 的 yaw
    stage_delta_yaws = []   # 存储每个 stage 的 delta yaw
    stage_stepwise_sdf_means = {'index': [], 'middle': [], 'thumb': []}  # 存储每个 stage 的 12 个 step 的 SDF 值

    # 遍历每个 trial
    for trial_idx in range(1, num_trials + 1):
        trial_dir = os.path.join(directory, f'trial_{trial_idx}')

        # 遍历该 trial 下的每个 stage
        for stage_idx in range(num_stages):
            csv_path = os.path.join(trial_dir, f'yaw_sdf_results_{stage_idx}.csv')
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"CSV file not found: {csv_path}")

            df = pd.read_csv(csv_path)

            # 计算该 stage 的 yaw 值
            # 假设每个 stage 有 12 行（12 个 step）的数据，每行 yaw 相同或者可以取 unique 后计算范围
            yaws = np.abs(df['yaw'].unique())  # 得到该 stage 中所有 step 的 yaw（一般有 12 个值）
            # 计算 yaw 范围：最大值 - 最小值
            mean_yaw_this_stage = np.max(yaws) - np.min(yaws)
            stage_yaws.append(mean_yaw_this_stage)

            # 计算 delta yaw：对 yaws 做差分，并取平均（11 个 delta）
            delta_yaws = np.abs(np.diff(yaws))
            if len(delta_yaws) > 0:
                mean_delta_this_stage = np.mean(delta_yaws)
            else:
                mean_delta_this_stage = 0.0
            stage_delta_yaws.append(mean_delta_this_stage)

            # 对于每个 finger，提取该 stage 中 12 个 step 的 SDF 值
            for finger in ['index', 'middle', 'thumb']:
                # 筛选出当前 finger 的所有行，顺序应对应 12 个 step
                df_finger = df[df['finger'] == finger]
                # 如果该 finger 数据不满 12 行，可考虑报错或填充
                if len(df_finger) < num_steps:
                    raise ValueError(f"Not enough steps for finger {finger} in {csv_path}")

                # 依次提取每个 step 的 sdf 值，注意这里假设 sdf 字符串形如 "[[0.00123]]"
                step_sdf_values = []
                for step_idx in range(num_steps):
                    val_str = df_finger.iloc[step_idx]['sdf']
                    # 解析字符串中的数值
                    step_sdf = float(val_str.strip('[]'))
                    step_sdf_values.append(step_sdf)
                # 将当前 stage 的 12 个 step 的 SDF 均值（这里已是每个 step 的值）存入结果字典
                stage_stepwise_sdf_means[finger].append(step_sdf_values)

    print(np.array(stage_yaws).shape, np.array(stage_delta_yaws).shape)
    return stage_yaws, stage_delta_yaws, stage_stepwise_sdf_means



#####################################
# 3. 分别读取 base_dir 和 compare_dir
#####################################
base_stage_yaws, base_stage_delta_yaws, base_sdf_data = process_directory(base_dir)
compare_stage_yaws, compare_stage_delta_yaws, compare_sdf_data = process_directory(compare_dir)
#####################################
# 4. 可视化对比：Yaw & Delta Yaw
#####################################
# 计算每个 stage 的平均 Yaw 和 Delta Yaw
num_total_stages = num_trials * num_stages

# 绘制 Yaw 比较图（添加连线）
# 计算差值：compare_stage_yaws - base_stage_yaws
yaw_difference = np.array(compare_stage_yaws) - np.array(base_stage_yaws)
indices = np.where(yaw_difference< 0)[0]
print(indices)
# 绘制差值柱状图
plt.figure(figsize=(8, 5))
plt.bar(range(1, num_total_stages + 1), yaw_difference, label='yaw_difference', width=0.5, align='center', alpha=0.7, color='blue')
# 添加参考线 y=0
plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
plt.xlabel('Stage')
plt.ylabel('Yaw Difference (Compare - Base) (rads)')
plt.title('Yaw Difference Per Trail (Compare - Base)')
plt.grid(True)
# plt.savefig(f'{save_dir}/yaw_difference_each_trail.png')
plt.show()
# plt.close()


# 绘制 Delta Yaw 比较图（添加连线）
plt.figure(figsize=(8, 5))
step_yaw_difference = np.array(compare_stage_delta_yaws) - np.array(base_stage_delta_yaws)
indices = np.where(step_yaw_difference < 0)[0]
print(indices)
plt.bar(range(1, num_total_stages + 1), step_yaw_difference, label='step_yaw_difference', width=0.5, align='center', alpha=0.7, color='blue')
# 添加参考线 y=0
plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
plt.xlabel('Stage')
plt.ylabel('Yaw Difference (Compare - Base) (rads)')
plt.title('Yaw Difference Per Step (Compare - Base)')
plt.grid(True)
plt.legend()
# plt.savefig(f'{save_dir}/yaw_difference_each_step.png')
plt.show()
# plt.close()


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

    # 转换数据单位为mm
    base_sdf_array *= 100
    compare_sdf_array *= 100

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
    plt.ylabel('SDF/(mm)')
    plt.title(f'SDF Comparison for {finger.capitalize()} Finger')
    plt.legend()
    plt.grid(True)
    # plt.savefig(f'{save_dir}/{finger.capitalize()}_sdf_comparison.png')
    plt.show()
    # plt.close()

