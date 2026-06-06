

import os
import random

import numpy as np
import torch
from sklearn.mixture import GaussianMixture
from torch.utils.data import DataLoader

from config import args
from fgtad_trainer import train_fg_tad, MyDataset, seed_torch, collate_fn
from train_labels import Linear_Model


def get_z(trajs):
    """
    获取轨迹的潜在表示（latent representation）
    
    参数:
        trajs: 轨迹数据数组
    
    返回:
        z: 轨迹的潜在向量表示（CPU上的tensor）
    """
    data = MyDataset(trajs)
    loader = DataLoader(dataset=data, batch_size=args.batch_size, shuffle=False,
                        collate_fn=collate_fn, num_workers=4)
    FG_TAD_U.train_loader = loader
    return FG_TAD_U.get_hidden().cpu()


def load_gmm():

    checkpoint = torch.load('./models/gmm_update_{}.pt'.format(args.dataset), map_location=args.device, weights_only=False)
    gmm = GaussianMixture(n_components=args.n_cluster, covariance_type='diag')
    gmm.weights_ = checkpoint['gmm_update_weights']
    gmm.means_ = checkpoint['gmm_update_means']
    gmm.covariances_ = checkpoint['gmm_update_covariances']
    gmm.precisions_cholesky_ = checkpoint['gmm_update_precisions_cholesky']
    return gmm


def get_index(trajs, cats_sample):

    z = get_z(trajs)
    cats = gmm.predict(z) 
    index = [cat in cats_sample for cat in cats]  
    trajs = trajs[index]
    return index, z[index], trajs


def get_score(z):

    probs = gmm.predict_proba(z) 
    idx = []
    linear = Linear_Model()
    for label in range(args.n_cluster):
        data = -probs[:, label]  
        rank = linear.test(label, torch.Tensor(data).to(args.device))
        idx.append(rank)
    idx = np.array(idx).T  
    idxs = np.argsort(idx, axis=1)  

    return idxs


def update_data(origin_trajs, train_trajs, cats_sample):

    _, z, train_trajs = get_index(train_trajs, cats_sample) 
    idxs = get_score(z)  

    max_idxs = idxs[:, 0]  
    for i, traj in enumerate(train_trajs):
        max_idx = max_idxs[i]
     
        origin_trajs[max_idx].append(traj)

     
        min_c = args.n_cluster - 1
        min_idx = idxs[:, min_c][i]

      
        while not origin_trajs[min_idx]:
            min_c -= 1
            min_idx = idxs[:, min_c][i]
        origin_trajs[min_idx].pop(0)  

    return np.array(sum(origin_trajs, []), dtype=object)  


def train_update(train_trajs, test_trajs, labels, i):

    train_data = MyDataset(train_trajs)
    test_data = MyDataset(test_trajs)

    train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=8)
    outliers_loader = DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=False,
                                 collate_fn=collate_fn, num_workers=8)

    FG_TAD.train_loader = train_loader
    FG_TAD.outliers_loader = outliers_loader
    FG_TAD.labels = labels

    pr_auc = []

    for epoch in range(args.epochs):
        FG_TAD.train(epoch)
        results = FG_TAD.detection() 
        pr_auc.append(results)
    results = "%.4f" % max(pr_auc)
    print("File {} PR_AUC:".format(i), results)
    return max(pr_auc)


def test_update(test_trajs, labels, i):

    test_data = MyDataset(test_trajs)
    outliers_loader = DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=False,
                                 collate_fn=collate_fn, num_workers=8)

    FG_TAD_U.outliers_loader = outliers_loader
    FG_TAD_U.labels = labels

    pr_auc = FG_TAD_U.detection() 
    results = "%.4f" % pr_auc
    print("File {} PR_AUC:".format(i), results)
    return pr_auc


def get_category(trajs):

    z = get_z(trajs)  
    c_labels = gmm.predict(z) 

    origin_trajs = []

    for label in range(args.n_cluster):
        index = c_labels == label
        origin_trajs.append(trajs[index].tolist())
    return origin_trajs


def main():

    random.seed(1234)
   
    train_trajs = np.load('./data/{}/train_data_init.npy'.format(args.dataset),
                          allow_pickle=True)[-args.train_num:]


    if args.update_mode == 'rank':
        random_train_trajs = train_trajs

    all_pr_auc = []

    if args.dataset == 'porto':

        for i in range(1, 11):
    
            train_trajs_new = np.load('./data/{}/train_data_{}.npy'.format(args.dataset, i),
                                      allow_pickle=True)

            test_trajs = np.load(
                './data/{}/outliers_data_{}_{}_{}_{}.npy'.format(args.dataset, i, args.distance, args.fraction,
                                                                 args.obeserved_ratio),
                allow_pickle=True)

            outliers_idx = np.load(
                "./data/{}/outliers_idx_{}_{}_{}_{}.npy".format(args.dataset, i, args.distance, args.fraction,
                                                                args.obeserved_ratio),
                allow_pickle=True)

   
            labels = np.zeros(len(test_trajs))
            for idx in outliers_idx:
                labels[idx] = 1

            cats = list(range(0, args.n_cluster))
            cats_sample = random.sample(cats, args.n_cluster // 4) 
            test_index, _, _ = get_index(test_trajs, cats_sample)

            if args.update_mode == 'temporal':
  
                train_index, _, _ = get_index(train_trajs_new, cats_sample)

                train_trajs = np.concatenate((train_trajs, train_trajs_new[train_index]))[-len(train_trajs):]
                pr_auc = train_update(train_trajs, test_trajs[test_index], labels[test_index], i)

            elif args.update_mode == 'rank':

                trajs = get_category(random_train_trajs) 

                train_trajs = update_data(trajs, train_trajs_new, cats_sample)

                random_train_trajs = np.concatenate((random_train_trajs, train_trajs_new))[-len(train_trajs):]
                pr_auc = train_update(train_trajs, test_trajs[test_index], labels[test_index], i)

            elif args.update_mode == 'pretrain':

                pr_auc = test_update(test_trajs[test_index], labels[test_index], i)

            all_pr_auc.append(pr_auc)

    if args.dataset == 'cd':

        traj_path = "../dataset/chengdu/cd"
        path_list = os.listdir(traj_path)
        path_list.sort(key=lambda x: x.split('.'))
        path_list = path_list[3: 10] 

        for i in range(len(path_list)):

            train_trajs_new = np.load('./data/{}/train_data_{}.npy'.format(args.dataset, path_list[i][:8]),
                                      allow_pickle=True)
            print(len(train_trajs_new))
            test_trajs = np.load(
                './data/{}/outliers_data_{}_{}_{}_{}.npy'.format(args.dataset, path_list[i][:8], args.distance,
                                                                 args.fraction, args.obeserved_ratio),
                allow_pickle=True)
            outliers_idx = np.load(
                "./data/{}/outliers_idx_{}_{}_{}_{}.npy".format(args.dataset, path_list[i][:8], args.distance,
                                                                args.fraction, args.obeserved_ratio),
                allow_pickle=True)

            labels = np.zeros(len(test_trajs))
            for idx in outliers_idx:
                labels[idx] = 1

            cats = list(range(0, args.n_cluster))
            cats_sample = random.sample(cats, args.n_cluster // 4)
            test_index, _, _ = get_index(test_trajs, cats_sample)

            test_trajs = test_trajs[test_index]
            labels = labels[test_index]

            if args.update_mode == 'temporal':

                train_index, _, _ = get_index(train_trajs_new, cats_sample)
                train_trajs_new = train_trajs_new[train_index]

                train_trajs = train_trajs[-(len(train_trajs) - len(train_trajs_new)):]
                train_trajs = np.concatenate((train_trajs, train_trajs_new))
                print('Trajecotory num:', len(train_trajs))
                pr_auc = train_update(train_trajs, test_trajs, labels, i)

            elif args.update_mode == 'rank':
   
                trajs = get_category(random_train_trajs)
                train_trajs = update_data(trajs, train_trajs_new, cats_sample)

                random_train_trajs = random_train_trajs[-(len(random_train_trajs) - len(train_trajs_new)):]
                random_train_trajs = np.concatenate((random_train_trajs, train_trajs_new))
                print('Trajecotory num:', len(train_trajs))
                pr_auc = train_update(train_trajs, test_trajs, labels, i)

            elif args.update_mode == 'pretrain':

                pr_auc = test_update(test_trajs, labels, i)

            all_pr_auc.append(pr_auc)
    

    print('------------------------')
    results = "%.4f" % (sum(all_pr_auc) / len(all_pr_auc))
    print('Average PR_AUC:', results)
    print('------------------------')


if __name__ == "__main__":

    seed_torch(1234) 
    print("===========================")
    print("Dataset:", args.dataset)
    print("Mode:", args.update_mode)

    if args.dataset == 'porto':
        s_token_size = 51 * 119 
        t_token_size = 5760    
    elif args.dataset == 'cd':
        s_token_size = 167 * 154
        t_token_size = 8640

    gmm = load_gmm()


    FG_TAD = train_fg_tad(s_token_size, t_token_size, None, None, None, args)
    FG_TAD.mode = 'update'

    checkpoint = torch.load(FG_TAD.path_checkpoint, map_location=args.device, weights_only=False)
    FG_TAD.FGTAD_S.load_state_dict(checkpoint['model_state_dict_s'])
    FG_TAD.FGTAD_T.load_state_dict(checkpoint['model_state_dict_t'])

    FG_TAD_U = train_fg_tad(s_token_size, t_token_size, None, None, None, args)
    checkpoint_U = torch.load(FG_TAD_U.path_checkpoint, map_location=args.device, weights_only=False)
    FG_TAD_U.mode = 'update'

    FG_TAD_U.FGTAD_S.load_state_dict(checkpoint_U['model_state_dict_s'])
    FG_TAD_U.FGTAD_T.load_state_dict(checkpoint_U['model_state_dict_t'])

    main()
