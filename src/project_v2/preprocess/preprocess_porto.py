import datetime
import json
import random
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sparse
from sklearn.model_selection import train_test_split

from preprocess_utils import *

PROJECT_V2_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_V2_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_V2_ROOT))

from path_utils import dataset_data_dir, normalize_runtime_paths, raw_dataset_dir


def time_convert(timestamp):
    return datetime.datetime.fromtimestamp(timestamp)


def preprocess(trajectories, traj_num, point_num):
    trajs = []  # Preprocessed trajectories

    for traj in trajectories.itertuples():

        traj_seq = []
        valid = True  # Flag to determine whether a trajectory is in boundary

        polyline = json.loads(traj.POLYLINE)
        timestamp = traj.TIMESTAMP

        if len(polyline) >= shortest:
            for lng, lat in polyline:

                if in_boundary(lat, lng, boundary):
                    grid_i = int((lat - boundary['min_lat']) / lat_size)
                    grid_j = int((lng - boundary['min_lng']) / lng_size)

                    t = datetime.datetime.fromtimestamp(timestamp)
                    t = [t.hour, t.minute, t.second, t.year, t.month, t.day]  # Time vector

                    traj_seq.append([int(grid_i * lng_grid_num + grid_j), t])
                    timestamp += 15  # In porto dataset, the sampling rate is 15

                else:
                    valid = False
                    break

            # Randomly delete 30% trajectory points to make the sampling rate not fixed
            to_delete = set(random.sample(range(len(traj_seq)), int(len(traj_seq) * 0.3)))
            traj_seq = [item for index, item in enumerate(traj_seq) if index not in to_delete]

            # Lengths are limited from 20 to 50
            if valid:
                if len(traj_seq) <= longest:
                    trajs.append(traj_seq)
                else:
                    trajs += cutting_trajs(traj_seq, longest, shortest)

    traj_num += len(trajs)

    for traj in trajs:
        point_num += len(traj)

    return trajs, traj_num, point_num


def main(cli_args):
    # Read csv file
    trajectories = pd.read_csv("{}/{}.csv".format(data_dir, data_name), header=0, usecols=['POLYLINE', 'TIMESTAMP'])
    trajectories['datetime'] = trajectories['TIMESTAMP'].apply(time_convert)

    # Inititial dataset
    start_time = datetime.datetime(2013, 7, 1, 0, 0, 0)
    end_time = datetime.datetime(2013, 9, 1, 0, 0, 0)

    traj_num, point_num = 0, 0

    # Select trajectories from start time to end time
    trajs = trajectories[(trajectories['datetime'] >= start_time) & (trajectories['datetime'] < end_time)]
    preprocessed_trajs, traj_num, point_num = preprocess(trajs, traj_num, point_num)
    train_data, test_data = train_test_split(preprocessed_trajs, test_size=0.2, random_state=42)

    output_dir = dataset_data_dir(cli_args)
    np.save(output_dir / "train_data_init.npy", np.array(train_data, dtype=object))
    np.save(output_dir / "test_data_init.npy", np.array(test_data, dtype=object))

    start_time = datetime.datetime(2013, 9, 1, 0, 0, 0)

    # Evolving dataset
    for month in range(1, 11):
        end_time = start_time + datetime.timedelta(days=30)
        trajs = trajectories[(trajectories['datetime'] >= start_time) & (trajectories['datetime'] < end_time)]

        preprocessed_trajs, traj_num, point_num = preprocess(trajs, traj_num, point_num)
        train_data, test_data = train_test_split(preprocessed_trajs, test_size=0.2, random_state=42)

        np.save(output_dir / "train_data_{}.npy".format(month), np.array(train_data, dtype=object))
        np.save(output_dir / "test_data_{}.npy".format(month), np.array(test_data, dtype=object))

        start_time = end_time

    # Dataset statistics
    print("Total trajectory num:", traj_num)
    print("Total point num:", point_num)

    print('Fnished!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='porto')
    parser.add_argument('--project_root', default='')
    parser.add_argument('--data_root', default='')
    parser.add_argument('--dataset_data_dir', default='')
    parser.add_argument('--raw_dataset_root', default='')
    parser.add_argument('--raw_dataset_dir', default='')
    cli_args = normalize_runtime_paths(parser.parse_args())

    random.seed(1234)
    np.random.seed(1234)

    data_dir = str(raw_dataset_dir(cli_args))
    data_name = "porto"

    boundary = {'min_lat': 41.140092, 'max_lat': 41.185969, 'min_lng': -8.690261, 'max_lng': -8.549155}
    grid_size = 0.1
    shortest, longest = 20, 50

    lat_size, lng_size, lat_grid_num, lng_grid_num = grid_mapping(boundary, grid_size)
    A, D = generate_matrix(lat_grid_num, lng_grid_num)

    output_dir = dataset_data_dir(cli_args)
    sparse.save_npz(output_dir / 'adj.npz', A)
    sparse.save_npz(output_dir / 'd_norm.npz', D)

    print('Grid size:', (lat_grid_num, lng_grid_num))
    print('----------Preprocessing----------')
    main(cli_args)
