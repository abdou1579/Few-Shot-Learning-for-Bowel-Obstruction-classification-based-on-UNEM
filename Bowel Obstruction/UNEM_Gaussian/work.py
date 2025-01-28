import numpy as np
from src.methods.U_Paddle import U_Paddle
from src.methods.U_Paddle_embt import U_Paddle_EMBT
from src.utils import compute_confidence_interval, load_checkpoint, Logger, extract_mean_features, extract_features
from UNEM_Gaussian.src.methods.tim import ALPHA_TIM, TIM_GD
from UNEM_Gaussian.src.methods.paddle import PADDLE
from UNEM_Gaussian.src.methods.soft_km import SOFT_KM
from UNEM_Gaussian.src.methods.laplacianshot import LaplacianShot
from UNEM_Gaussian.src.methods.paddle_gd import PADDLE_GD
from UNEM_Gaussian.src.methods.ici import ICI
from UNEM_Gaussian.src.methods.bdcspn import BDCSPN
from UNEM_Gaussian.src.methods.baseline import Baseline
from UNEM_Gaussian.src.methods.pt_map import PT_MAP
from UNEM_Gaussian.src.datasets import Tasks_Generator, get_dataset, get_dataloader, SamplerSupport, SamplerQuery, CategoriesSampler
import torch
import os
from src.utils import load_pickle, save_pickle
import random
import matplotlib.pyplot as plt

import pandas as pd

# Default args
class Args:
    shots = [1, 5]
    train_samples = 10
    k_eff = 5
    n_ways = 5
    n_query = 5
    sampling = 'random'
    used_set_support = 'support'
    alpha_dirichlet = 0.5
    train_lr = 0.001
    train_iter = 100
    dataset = 'dummy_dataset'
    arch = 'dummy_arch'
    train_n_layers = 3
    train_method = 'U_PADDLE'
    save_results = True


def main():
    args = Args()
    logger = Logger()
    df1 = pd.read_pickle('Bride_fshot/train_softmax_RN50_T30.plk')
    df2 = pd.read_pickle('Bride_fshot/train_visual_RN50.plk')
    
    
    features_support, features_query, labels_support, labels_query = train_test_split(df1['concat_features']. df1['concat_labels'], test_size = 0.5)

    # Load pre-extracted features from pandas pickles
    all_features_support = torch.tensor(features_support.values)
    all_labels_support = torch.tensor(labels_support.values, dtype=torch.long)
    all_features_query = torch.tensor(features_query.values)
    all_labels_query = torch.tensor(labels_query.values, dtype=torch.long)

    results = []
    loss_histories = []
    for shot in args.shots:
        u_paddle = U_Paddle(args.train_n_layers, device='cpu', log_file='log.txt')
        u_paddle.to('cpu')

        sampler = CategoriesSampler(all_labels_support, all_labels_query, args.train_samples, args.k_eff, args.n_ways, shot, args.n_query, args.sampling, args.used_set_support, args.alpha_dirichlet)
        sampler.create_list_classes(all_labels_support, all_labels_query)
        sampler_support = SamplerSupport(sampler)
        sampler_query = SamplerQuery(sampler)

        test_loader_query = [(all_features_query[indices, :], all_labels_query[indices]) for indices in sampler_query]
        test_loader_support = [(all_features_support[indices, :], all_labels_support[indices]) for indices in sampler_support]

        task_generator = Tasks_Generator(k_eff=args.k_eff, n_ways=args.n_ways, shot=shot, n_query=args.n_query, loader_support=test_loader_support, loader_query=test_loader_query, train_mean=torch.mean(all_features_support, dim=0), log_file='log.txt')
        tasks = task_generator.generate_tasks()

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(u_paddle.parameters(), lr=args.train_lr)
        scheduler = StepLR(optimizer, step_size=50, gamma=0.5)

        u_paddle, loss_history, params_history = training2(u_paddle, tasks, criterion, optimizer, scheduler, args.train_iter, 'cpu')

        results.append(u_paddle)
        loss_histories.append(loss_history)

        if args.save_results:
            path = f'results/u_params/{args.dataset}/{args.arch}/{args.train_n_layers}'
            os.makedirs(path, exist_ok=True)
            with open(f'{path}/{args.train_method}.txt', 'w') as f:
                f.write(f'{args.k_eff}\tlayers: {args.train_n_layers}\n')
                for i, shot in enumerate(args.shots):
                    f.write(f'{shot}-shot -- sampling: {args.sampling}\n')
                    torch.save(results[i].state_dict(), f'{path}/{args.train_method}_{shot}shots.pt')

            for i, loss_history in enumerate(loss_histories):
                plt.plot(loss_history)
                plt.savefig(f'{path}/{args.train_method}_{args.shots[i]}shots_loss_history.png')
                plt.close()

# Training function
def training2(u_padnet, task_dic, criterion, optimizer, scheduler, epochs, device):
    support, query = task_dic['x_s'], task_dic['x_q']
    y_support, y_query = task_dic['y_s'], task_dic['y_q']
    loss_history = []

    for epoch in tqdm(range(epochs)):
        optimizer.zero_grad()
        logits = u_padnet(support, y_support, query, device)
        logits = logits.view(-1, logits.size(-1))
        labels = y_query.view(-1)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        loss_history.append(loss.item())

    return u_padnet, loss_history, None

if __name__ == "__main__":
    main()
