import copy
import numpy as np

from config import args
from experiment_utils import get_outlier_data_filename, get_outlier_idx_filename
import generate_outliers as gout
from path_utils import outlier_data_path, outlier_idx_path, test_data_path


def save_one(anomaly_type: str):
    local_args = copy.deepcopy(args)
    local_args.anomaly_type = anomaly_type
    if anomaly_type == 'mixed':
        local_args.mixed_weights = [0.25, 0.25, 0.25, 0.25]

    np.random.seed(int(local_args.outlier_seed))
    gout.args = local_args
    gout.map_size = (51, 119) if local_args.dataset == 'porto' else (167, 154)

    data = np.load(test_data_path(local_args, 'init'), allow_pickle=True)
    outliers_trajs, outliers_idx = gout.generate_outliers(data, anomaly_type=anomaly_type)
    data_name = get_outlier_data_filename(local_args, 'init')
    idx_name = get_outlier_idx_filename(local_args, 'init')
    np.save(outlier_data_path(local_args, data_name), np.array(outliers_trajs, dtype=object))
    np.save(outlier_idx_path(local_args, idx_name), np.array(outliers_idx))
    print(anomaly_type, len(outliers_idx), data_name)


if __name__ == '__main__':
    anomalies = ['detour', 'time', 'speed', 'stay', 'mixed'] if args.prepare_all_anomalies else [args.anomaly_type]
    for anomaly in anomalies:
        save_one(anomaly)
