import numpy as np
from src.methods.U_Paddle import U_Paddle
from src.methods.U_Paddle_embt import U_Paddle_EMBT
from src.utils import compute_confidence_interval, load_checkpoint, Logger, extract_mean_features, extract_features
from src.methods.tim import ALPHA_TIM, TIM_GD
from src.methods.paddle import PADDLE
from src.methods.soft_km import SOFT_KM
from src.methods.paddle_gd import PADDLE_GD
from src.methods.ici import ICI
from src.methods.laplacianshot import LaplacianShot
from src.methods.bdcspn import BDCSPN
from src.methods.baseline import Baseline
from src.methods.pt_map import PT_MAP
from src.datasets import Tasks_Generator, get_dataset, get_dataloader, SamplerSupport, SamplerQuery, CategoriesSampler
import torch
import os
from src.utils import load_pickle, save_pickle
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_selection
import pandas as pd

shots = 5
df1 = pd.read_pickle('Bride_fshot/train_softmax_RN50_T30.plk')
df2 = pd.read_pickle('Bride_fshot/train_visual_RN50.plk')


features_support, features_query, labels_support, labels_query = train_test_split(df1['concat_features']. df1['concat_labels'], test_size = 0.5)


all_features_support = features_support
all_labels_support = labels_support.long()
all_features_query = features_query
all_labels_query = labels_query.long()


results = []
loss_histories = []
for shot in shots:
    u_paddle = self.get_model_new()
    u_paddle = u_paddle.to(self.device)

    #for i in range(int(self.args.train_tasks/self.args.train_batch_size)):
    sampler = CategoriesSampler(all_labels_support, all_labels_query, self.args.train_samples,
                            3, 1, shot, 5, 
                            2, 2, 0.2)
    sampler.create_list_classes(all_labels_support, all_labels_query)
    sampler_support = SamplerSupport(sampler)
    sampler_query = SamplerQuery(sampler)

    test_loader_query = []
    for indices in sampler_query :
        test_loader_query.append((all_features_query[indices,:], all_labels_query[indices]))

    test_loader_support = []
    for indices in sampler_support :
        test_loader_support.append((all_features_support[indices,:], all_labels_support[indices]))

    task_generator = Tasks_Generator(k_eff=3, n_ways=self.args.n_ways, shot=shot, n_query=self.args.n_query, loader_support=test_loader_support, loader_query=test_loader_query, train_mean=train_mean, log_file=self.log_file)
   tasks = task_generator.generate_tasks()
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(u_paddle.parameters(), lr=self.args.train_lr)
    scheduler = StepLR(optimizer, step_size=50, gamma=0.5)
    
    u_paddle, loss_history, params_history = training2(u_paddle, tasks, criterion, optimizer, scheduler, self.args.train_iter, self.device)
    #torch.cuda.empty_cache()
    
    results.append(u_paddle)
    loss_histories.append(loss_history)



