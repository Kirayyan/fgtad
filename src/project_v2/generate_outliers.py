import datetime
import math
import os
from datetime import timedelta

import numpy as np

from config import args
from experiment_utils import get_outlier_data_filename, get_outlier_idx_filename
from path_utils import outlier_data_path, outlier_idx_path, test_data_path


def perturb_point(point, level, offset=None):
    point, times = point[0], point[1]
    x, y = int(point // map_size[1]), int(point % map_size[1])
    if offset is None:
        offset = [[0, 1], [1, 0], [-1, 0], [0, -1], [1, 1], [-1, -1], [-1, 1], [1, -1]]
        x_offset, y_offset = offset[np.random.randint(0, len(offset))]
    else:
        x_offset, y_offset = offset
    if 0 <= x + x_offset * level < map_size[0] and 0 <= y + y_offset * level < map_size[1]:
        x += x_offset * level
        y += y_offset * level
    return [int(x * map_size[1] + y), times]


def convert(point):
    return [int(point // map_size[1]), int(point % map_size[1])]


def distance(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def time_calcuate(vec, s):
    a = datetime.datetime(vec[3], vec[4], vec[5], vec[0], vec[1], vec[2])
    t = a + timedelta(seconds=s)
    return [t.hour, t.minute, t.second, t.year, t.month, t.day]


def perturb_time(traj, st_loc, end_loc, time_offset, interval):
    for i in range(st_loc, end_loc):
        traj[i][1] = time_calcuate(traj[i][1], int((i - st_loc + 1) * time_offset * interval))
    for i in range(end_loc, len(traj)):
        traj[i][1] = time_calcuate(traj[i][1], int((end_loc - st_loc) * time_offset * interval))
    return traj


def get_interval():
    return 15 if args.dataset == 'porto' else 10


def detour_batch(batch_x, level, prob):
    noisy_batch_x = []
    interval = get_interval()
    for traj in batch_x:
        if len(traj) < 4:
            noisy_batch_x.append(traj[:int(len(traj) * args.obeserved_ratio)])
            continue
        anomaly_len = max(2, int(len(traj) * prob))
        if len(traj) - anomaly_len <= 1:
            noisy_batch_x.append(traj[:int(len(traj) * args.obeserved_ratio)])
            continue
        anomaly_st_loc = np.random.randint(1, len(traj) - anomaly_len - 1)
        anomaly_ed_loc = anomaly_st_loc + anomaly_len
        p_traj = traj[:anomaly_st_loc] + [perturb_point(p, level) for p in traj[anomaly_st_loc:anomaly_ed_loc]] + traj[anomaly_ed_loc:]
        dis = max(distance(convert(traj[anomaly_st_loc][0]), convert(traj[anomaly_ed_loc][0])), 1)
        time_offset = (level * 2) / dis
        p_traj = perturb_time(p_traj, anomaly_st_loc, anomaly_ed_loc, time_offset, interval)
        noisy_batch_x.append(p_traj[:int(len(p_traj) * args.obeserved_ratio)])
    return noisy_batch_x


def time_batch(batch_x, level, prob=0.1):
    noisy_batch_x = []
    for traj in batch_x:
        if len(traj) == 0:
            noisy_batch_x.append(traj)
            continue
        num_points_to_process = min(len(traj), 64)
        num_selected = max(1, int(prob * num_points_to_process))
        anomaly_start_idx_list = sorted(np.random.choice(num_points_to_process, size=num_selected, replace=False))
        noisy_traj = []
        for i, (point_id, timestamp) in enumerate(traj):
            num_greater = sum(i > idx for idx in anomaly_start_idx_list)
            if num_greater > 0:
                dt = datetime.datetime(timestamp[3], timestamp[4], timestamp[5], timestamp[0], timestamp[1], timestamp[2])
                dt = dt + datetime.timedelta(minutes=int(level * num_greater))
                noisy_traj.append([point_id, [dt.hour, dt.minute, dt.second, dt.year, dt.month, dt.day]])
            else:
                noisy_traj.append([point_id, timestamp])
        noisy_batch_x.append(noisy_traj[:int(len(noisy_traj) * args.obeserved_ratio)])
    return noisy_batch_x


def interpolate_point(p1, p2, ratio):
    x1, y1 = int(p1[0] // map_size[1]), int(p1[0] % map_size[1])
    x2, y2 = int(p2[0] // map_size[1]), int(p2[0] % map_size[1])
    new_x = int(x1 + (x2 - x1) * ratio)
    new_y = int(y1 + (y2 - y1) * ratio)
    new_point_id = int(new_x * map_size[1] + new_y)
    t1 = datetime.datetime(p1[1][3], p1[1][4], p1[1][5], p1[1][0], p1[1][1], p1[1][2])
    t2 = datetime.datetime(p2[1][3], p2[1][4], p2[1][5], p2[1][0], p2[1][1], p2[1][2])
    delta = t2 - t1
    new_t = t1 + datetime.timedelta(seconds=int(delta.total_seconds() * ratio))
    return [new_point_id, [new_t.hour, new_t.minute, new_t.second, new_t.year, new_t.month, new_t.day]]


def speed_batch(batch_x, mode='decelerate', factor=2.0, prob=0.3):
    noisy_batch_x = []
    for traj in batch_x:
        if len(traj) < 3:
            noisy_batch_x.append(traj[:int(len(traj) * args.obeserved_ratio)])
            continue
        anomaly_len = max(2, int(len(traj) * prob))
        if len(traj) - anomaly_len <= 0:
            noisy_batch_x.append(traj[:int(len(traj) * args.obeserved_ratio)])
            continue
        anomaly_st_loc = np.random.randint(0, len(traj) - anomaly_len)
        anomaly_ed_loc = anomaly_st_loc + anomaly_len
        if mode == 'accelerate':
            new_traj = list(traj[:anomaly_st_loc])
            step = max(2, int(factor))
            for i in range(anomaly_st_loc, anomaly_ed_loc, step):
                if i < len(traj):
                    new_traj.append(traj[i])
            new_traj.extend(traj[anomaly_ed_loc:])
        else:
            new_traj = list(traj[:anomaly_st_loc])
            num_insert = max(1, int(factor))
            for i in range(anomaly_st_loc, anomaly_ed_loc - 1):
                new_traj.append(traj[i])
                for j in range(1, num_insert + 1):
                    ratio = j / (num_insert + 1)
                    new_traj.append(interpolate_point(traj[i], traj[i + 1], ratio))
            if anomaly_ed_loc > anomaly_st_loc:
                new_traj.append(traj[anomaly_ed_loc - 1])
            new_traj.extend(traj[anomaly_ed_loc:])
        noisy_batch_x.append(new_traj[:int(len(new_traj) * args.obeserved_ratio)])
    return noisy_batch_x


def stay_batch(batch_x, prob=0.3):
    noisy_batch_x = []
    interval = get_interval()
    for traj in batch_x:
        if len(traj) < 3:
            noisy_batch_x.append(traj[:int(len(traj) * args.obeserved_ratio)])
            continue
        p_traj = [[p[0], list(p[1])] for p in traj]
        anomaly_len = max(2, int(len(traj) * prob))
        if len(traj) - anomaly_len <= 1:
            noisy_batch_x.append(p_traj[:int(len(p_traj) * args.obeserved_ratio)])
            continue
        anomaly_st_loc = np.random.randint(1, len(traj) - anomaly_len)
        anomaly_ed_loc = anomaly_st_loc + anomaly_len
        stay_loc = traj[anomaly_st_loc][0]
        base_time = traj[anomaly_st_loc][1]
        extra_per_step = max(1, int(args.time_distance))
        stay_interval = interval + extra_per_step
        for i in range(anomaly_st_loc, anomaly_ed_loc):
            p_traj[i][0] = stay_loc
            offset_sec = (i - anomaly_st_loc) * stay_interval
            p_traj[i][1] = time_calcuate(base_time, offset_sec)
        original_seg_duration = (anomaly_len - 1) * interval
        new_seg_duration = (anomaly_len - 1) * stay_interval
        extra_delay = new_seg_duration - original_seg_duration
        for i in range(anomaly_ed_loc, len(p_traj)):
            p_traj[i][1] = time_calcuate(traj[i][1], extra_delay)
        noisy_batch_x.append(p_traj[:int(len(p_traj) * args.obeserved_ratio)])
    return noisy_batch_x


def choose_mixed_type():
    types = ['detour', 'time', 'speed', 'stay']
    return np.random.choice(types, p=np.array(args.mixed_weights, dtype=float))


def perturb_single(traj, anomaly_type: str):
    if anomaly_type == 'detour':
        return detour_batch([traj], args.spatial_distance, args.spatial_fraction)[0]
    if anomaly_type == 'time':
        return time_batch([traj], args.time_distance, args.time_fraction)[0]
    if anomaly_type == 'speed':
        return speed_batch([traj], args.speed_mode, args.speed_factor, args.speed_fraction)[0]
    if anomaly_type == 'stay':
        return stay_batch([traj], args.stay_fraction)[0]
    raise ValueError(f'Unknown anomaly_type: {anomaly_type}')


def perturb_batch(batch_x, selected_idx, anomaly_type='detour'):
    noisy_batch_x = []
    selected_idx = set(selected_idx.tolist() if hasattr(selected_idx, 'tolist') else selected_idx)
    for idx, traj in enumerate(batch_x):
        if idx not in selected_idx:
            noisy_batch_x.append(traj[:int(len(traj) * args.obeserved_ratio)])
            continue
        current_type = choose_mixed_type() if anomaly_type == 'mixed' else anomaly_type
        noisy_batch_x.append(perturb_single(traj, current_type))
    return noisy_batch_x


# IMPORTANT: keep detour generation bit-for-bit compatible with the original baseline.
# The baseline samples anomaly_st_loc for every trajectory before checking whether the
# trajectory is selected; changing that random-call order changes the whole test set.
def perturb_batch_detour_baseline(batch_x, level, prob, selected_idx):
    noisy_batch_x = []
    interval = get_interval()

    for idx, traj in enumerate(batch_x):
        anomaly_len = int(len(traj) * prob)
        anomaly_st_loc = np.random.randint(1, len(traj) - anomaly_len - 1)

        if idx in selected_idx:
            anomaly_ed_loc = anomaly_st_loc + anomaly_len
            p_traj = traj[:anomaly_st_loc] + [perturb_point(p, level) for p in
                                              traj[anomaly_st_loc:anomaly_ed_loc]] + traj[anomaly_ed_loc:]

            dis = max(distance(convert(traj[anomaly_st_loc][0]), convert(traj[anomaly_ed_loc][0])), 1)
            time_offset = (level * 2) / dis
            p_traj = perturb_time(p_traj, anomaly_st_loc, anomaly_ed_loc, time_offset, interval)
        else:
            p_traj = traj

        p_traj = p_traj[:int(len(p_traj) * args.obeserved_ratio)]
        noisy_batch_x.append(p_traj)

    return noisy_batch_x


def generate_outliers(trajs, ratio=args.ratio, anomaly_type='detour'):
    traj_num = len(trajs)
    selected_size = 0 if traj_num == 0 or ratio <= 0 else max(1, int(traj_num * ratio))
    selected_idx = np.random.randint(0, traj_num, size=selected_size) if selected_size > 0 else np.array([], dtype=int)

    if anomaly_type == 'detour':
        new_trajs = perturb_batch_detour_baseline(trajs, args.spatial_distance, args.spatial_fraction, selected_idx)
    else:
        new_trajs = perturb_batch(trajs, selected_idx, anomaly_type)
    return new_trajs, selected_idx


def process_split(split='init'):
    test_file = test_data_path(args, split)
    if not os.path.exists(test_file):
        print(f'Warning: {test_file} not found, skipping {split}...')
        return
    data = np.load(test_file, allow_pickle=True)
    outliers_trajs, outliers_idx = generate_outliers(data, anomaly_type=args.anomaly_type)
    np.save(outlier_data_path(args, get_outlier_data_filename(args, split)), np.array(outliers_trajs, dtype=object))
    np.save(outlier_idx_path(args, get_outlier_idx_filename(args, split)), np.array(outliers_idx))
    print(f'Saved split={split}: {len(outliers_idx)} outliers')


if __name__ == '__main__':
    np.random.seed(int(args.outlier_seed))
    print('=========================')
    print('Dataset:', args.dataset)
    print('Anomaly Type:', args.anomaly_type)
    print('Observed Ratio:', args.obeserved_ratio)
    print('Exp Name:', args.exp_name)
    if args.anomaly_type == 'mixed':
        print('Mixed weights [detour,time,speed,stay]:', args.mixed_weights)

    if args.dataset == 'porto':
        map_size = (51, 119)
        process_split('init')
        for i in range(1, 11):
            process_split(str(i))
    elif args.dataset == 'cd':
        map_size = (167, 154)
        process_split('init')
    else:
        raise ValueError(f'Unsupported dataset: {args.dataset}')
