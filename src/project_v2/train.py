# ========== Environment Variable Settings (must be set before all imports) =========
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import args
from experiment_utils import get_outlier_data_filename, get_outlier_idx_filename
from fgtad_trainer import train_fg_tad, MyDataset, seed_torch, collate_fn
from path_utils import outlier_data_path, outlier_idx_path, train_data_path


def _require(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return path


def _truncate(arr, max_samples: int):
    if max_samples and max_samples > 0:
        return arr[:max_samples]
    return arr


def _build_labels(n_items, outliers_idx):
    labels = np.zeros(n_items)
    valid_idx = [int(i) for i in outliers_idx if 0 <= int(i) < n_items]
    for i in valid_idx:
        labels[i] = 1
    return labels


def _eval_current_checkpoint(fg_tad):
    checkpoint = torch.load(fg_tad.path_checkpoint, map_location=args.device, weights_only=False)
    fg_tad.FGTAD_S.load_state_dict(checkpoint['model_state_dict_s'], strict=False)
    fg_tad.FGTAD_T.load_state_dict(checkpoint['model_state_dict_t'], strict=False)
    selected_score = fg_tad.detection()
    fg_tad.logger.info("SELECTED_METRIC(%s): %.4f", args.metric, selected_score)
    if args.metric == "pr_auc":
        fg_tad.logger.info("PR_AUC: {:.4f}".format(selected_score))
    return selected_score


def main():
    train_path = _require(str(train_data_path(args, "init")))
    current_outlier_path = _require(str(outlier_data_path(args, get_outlier_data_filename(args))))
    current_outlier_idx_path = _require(str(outlier_idx_path(args, get_outlier_idx_filename(args))))

    train_trajs = _truncate(np.load(train_path, allow_pickle=True), args.max_train_samples)
    test_trajs = _truncate(np.load(current_outlier_path, allow_pickle=True), args.max_test_samples)
    outliers_idx = np.load(current_outlier_idx_path, allow_pickle=True)

    train_data = MyDataset(train_trajs)
    test_data = MyDataset(test_trajs)
    labels = _build_labels(len(test_trajs), outliers_idx)

    num_workers = min(max(0, args.num_workers), os.cpu_count() or 1)
    train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=num_workers, pin_memory=True)
    outliers_loader = DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=False,
                                 collate_fn=collate_fn, num_workers=num_workers, pin_memory=True)

    fg_tad = train_fg_tad(s_token_size, t_token_size, labels, train_loader, outliers_loader, args)
    fg_tad.logger.info(f'Experiment: {args.exp_name}')
    fg_tad.logger.info(f'max_train_samples={args.max_train_samples}, max_test_samples={args.max_test_samples}, num_workers={num_workers}')

    if args.task == 'train':
        fg_tad.logger.info("Start pretraining!")
        for epoch in range(args.pretrain_epochs):
            fg_tad.pretrain(epoch)

        fg_tad.train_gmm()
        fg_tad.save_weights_for_fgtad()

        fg_tad.logger.info("Start training!")
        fg_tad.load_fg_tad()
        for epoch in range(args.epochs):
            fg_tad.train(epoch)

        fg_tad.train_gmm_update()
        z = fg_tad.get_hidden()
        fg_tad.get_prob(z.cpu())

        if args.eval_after_train:
            fg_tad.logger.info('Start eval_after_train!')
            _eval_current_checkpoint(fg_tad)

    elif args.task == 'test':
        fg_tad.logger.info('Start testing!')
        fg_tad.logger.info(f'Anomaly Type: {args.anomaly_type}, observed_ratio={args.obeserved_ratio}')
        _eval_current_checkpoint(fg_tad)
    else:
        raise ValueError(f'Unsupported task: {args.task}')


if __name__ == "__main__":
    seed_torch(args.seed)
    print('PyTorch version:', torch.__version__)
    print('CUDA available:', torch.cuda.is_available())
    if torch.cuda.is_available():
        print('Device name:', torch.cuda.get_device_name(0))
        print('Capability:', torch.cuda.get_device_capability(0))

    if args.dataset == 'porto':
        s_token_size = 51 * 119
        t_token_size = 5760
    elif args.dataset == 'cd':
        s_token_size = 167 * 154
        t_token_size = 8640
    else:
        raise ValueError(f'Unsupported dataset: {args.dataset}')

    main()
