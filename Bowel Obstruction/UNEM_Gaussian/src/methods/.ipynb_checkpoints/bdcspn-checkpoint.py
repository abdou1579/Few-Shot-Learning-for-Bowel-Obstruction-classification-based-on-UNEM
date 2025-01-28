import torch.nn.functional as F
from tqdm import tqdm
import torch
import time
from numpy import linalg as LA
import numpy as np
from scipy.stats import mode
from ..utils import get_metric, Logger, get_one_hot
from sklearn.metrics import accuracy_score, f1_score
from scipy.optimize import linear_sum_assignment

class BDCSPN(object):
    def __init__(self, model, device, log_file, args):
        self.device = 'cpu'
        self.norm_type = args.norm_type
        self.k_eff = args.k_eff
        self.temp = args.temp
        self.num_NN = args.num_NN
        self.number_tasks = args.batch_size
        self.model = model
        self.log_file = log_file
        self.n_ways = args.n_ways
        self.logger = Logger(__name__, self.log_file)
        self.init_info_lists()
        self.dataset = args.dataset
        self.used_set_support = args.used_set_support

    def __del__(self):
        self.logger.del_logger()

    def init_info_lists(self):
        self.timestamps = []
        self.test_acc = []
        self.test_F1 = []


    def record_info(self, y_q, preds_q):
        """
        inputs:
            y_q : torch.Tensor of shape [n_tasks, q_shot]
            q_pred : torch.Tensor of shape [n_tasks, q_shot]:
        """
        n_tasks, q_shot = preds_q.shape
        preds_q = torch.from_numpy(preds_q)
        y_q = torch.from_numpy(y_q)
        accuracy = (preds_q == y_q).float().mean(1, keepdim=True)
        print('y_q', y_q)
        print("preds_q", preds_q)
        self.test_acc.append(accuracy)
        union = list(range(self.n_ways))
        for i in range(n_tasks):
            ground_truth = list(y_q[i].reshape(q_shot).cpu().numpy())
            preds = list(preds_q[i].reshape(q_shot).cpu().numpy())
            #union = set.union(set(ground_truth),set(preds))
            f1 = f1_score(ground_truth, preds, average='weighted', labels=union, zero_division=1)
            self.test_F1.append(f1)
        pass

    def get_logs(self):
        self.test_F1 = np.array([self.test_F1])
        self.test_acc = torch.cat(self.test_acc, dim=1).cpu().numpy()
        return {'timestamps': self.timestamps, 'F1': self.test_F1,
                'acc': self.test_acc}

    def normalization(self, z_s, z_q, train_mean):
        """
            inputs:
                z_s : np.Array of shape [n_task, s_shot, feature_dim]
                z_q : np.Array of shape [n_task, q_shot, feature_dim]
                train_mean: np.Array of shape [feature_dim]
        """
        z_s = z_s.cpu()
        z_q = z_q.cpu()
        # CL2N Normalization
        if self.norm_type == 'CL2N':
            z_s = z_s - train_mean
            z_s = z_s / LA.norm(z_s, 2, 2)[:, :, None]
            z_q = z_q - train_mean
            z_q = z_q / LA.norm(z_q, 2, 2)[:, :, None]
        # L2 Normalization
        elif self.norm_type == 'L2N':
            z_s = z_s / LA.norm(z_s, 2, 2)[:, :, None]
            z_q = z_q / LA.norm(z_q, 2, 2)[:, :, None]
        return z_s, z_q

    def proto_rectification(self, y_s, support, query, shot):
        """
            inputs:
                support : np.Array of shape [n_task, s_shot, feature_dim]
                query : np.Array of shape [n_task, q_shot, feature_dim]
                shot: Shot

            ouput:
                proto_weights: prototype of each class
        """
        eta = support.mean(1) - query.mean(1)  # Shifting term
        query = query + eta[:, np.newaxis, :]  # Adding shifting term to each normalized query feature
        if self.dataset == 'inatural' and self.used_set_support == 'repr':
            query_aug = torch.cat((support, query), axis=1)  # Augmented set S' (X')
            one_hot = get_one_hot(y_s)
            counts = one_hot.sum(1).view(support.size()[0], -1, 1)
            weights = one_hot.transpose(1, 2).matmul(support)
            print("weights", weights.size())
            print("counts", counts.size())
            support_ = weights / counts
        else:
            query_aug = np.concatenate((support, query), axis=1)  # Augmented set S' (X')
            support_ = support.reshape(support.shape[0], shot, self.n_ways, support.shape[-1]).mean(1)  # Init basic prototypes Pn
            support_ = torch.from_numpy(support_)
            query_aug = torch.from_numpy(query_aug)

        proto_weights = []
        for j in tqdm(range(self.number_tasks)):
            distance = get_metric('cosine')(support_[j], query_aug[j])
            predict = torch.argmin(distance, dim=1)
            cos_sim = F.cosine_similarity(query_aug[j][:, None, :], support_[j][None, :, :], dim=2)  # Cosine similarity between X' and Pn
            cos_sim = self.temp * cos_sim
            W = F.softmax(cos_sim, dim=1)
            support_list = [(W[predict == i, i].unsqueeze(1) * query_aug[j][predict == i]).mean(0, keepdim=True) for i in predict.unique()]
            proto = torch.cat(support_list, dim=0)  # Rectified prototypes P'n
            
            if proto.shape[0] != len(torch.unique(y_s)):
                support_2 = torch.from_numpy(support)
                query_2 = torch.from_numpy(query)
                query_aug = torch.cat((support_2, query_2), axis=1)  # Augmented set S' (X')
                one_hot = get_one_hot(y_s.squeeze(2))
                counts = one_hot.sum(1).view(support_2.size()[0], -1, 1)
                weights = one_hot.transpose(1, 2).matmul(support_2)
                proto = (weights / counts)[j]

            proto_weights.append(proto)
        proto_weights = np.stack(proto_weights, axis=0)
        return proto_weights

    def run_task(self, task_dic, shot):
        # Extract support and query
        y_s, y_q = task_dic['y_s'], task_dic['y_q']
        x_s, x_q = task_dic['x_s'], task_dic['x_q']
        train_mean = task_dic['train_mean']

        
        if self.dataset == 'inatural' and self.used_set_support == 'repr':
            # Extract features
            # support, query = extract_features(self.model, x_s, x_q)
            # support = torch.load('features_support.pt').to('cpu')
            # support = support.unsqueeze(0)
            # y_s = torch.load('labels_support.pt').to('cpu')
            # y_s = y_s.unsqueeze(0)
            support, query = self.normalization(z_s=x_s, z_q=x_q, train_mean=train_mean)
            support = x_s.to(self.device)
            query = x_q.to(self.device)
            y_s = y_s.long().squeeze(2)
            y_q = y_q.long().squeeze(2)
            query = query.to('cpu')
            self.logger.info(" ==> Executing proto-rectification ...")
            support = self.proto_rectification(y_s=y_s, support=support, query=query, shot=shot)
            query = query.numpy()
            y_q = y_q.numpy()
            print("support shape", support.shape)
        
        else:
            # Extract features
            #z_s, z_q = extract_features(model=self.model, support=x_s, query=x_q)

            # Perform normalizations required
            support, query = self.normalization(z_s=x_s, z_q=x_q, train_mean=train_mean)
            support = support.numpy()
            query = query.numpy()
            # y_s = y_s.numpy().squeeze(2)[:,::shot][0]
            # y_s = y_s.squeeze(2)[:, :self.n_ways][0]
            y_q = y_q.long().squeeze(2).numpy()
            support = self.proto_rectification(y_s=y_s, support=support, query=query, shot=shot)
            
        # support = torch.from_numpy(support)
        # query = torch.from_numpy(query)
        # y_s = torch.from_numpy(y_s)
        # y_q = torch.from_numpy(y_q)
        # Run adaptation
        self.run_prediction(support=support, query=query, y_s=y_s, y_q=y_q, shot=shot)

        # Extract adaptation logs
        logs = self.get_logs()

        return logs

    def run_prediction(self, support, query, y_s, y_q, shot):
        """
        Corresponds to the BD-CSPN inference
        inputs:
            support : torch.Tensor of shape [n_task, s_shot, feature_dim]
            query : torch.Tensor of shape [n_task, q_shot, feature_dim]
            y_s : torch.Tensor of shape [n_task, s_shot]
            y_q : torch.Tensor of shape [n_task, q_shot]
        """
        t0 = time.time()
        self.logger.info(" ==> Executing predictions on {} shot tasks ...".format(shot))
        out_list = []
        for i in tqdm(range(self.number_tasks)):
            if self.dataset == 'inatural' and self.used_set_support == 'repr':
                y_s_i = np.unique(y_s[i])
            else:
                y_s_i = y_s.numpy().squeeze(2)[i, :self.n_ways]
                print(y_s_i)
            substract = support[i][:, None, :] - query[i]
            print("substract", substract.shape)
            distance = LA.norm(substract, 2, axis=-1)
            print("dist", distance)
            idx = np.argpartition(distance, self.num_NN, axis=0)[:self.num_NN]
            print("idx", idx)
            nearest_samples = np.take(y_s_i, idx)
            out = mode(nearest_samples, axis=0)[0]
            out_list.append(out)
        n_tasks, q_shot, feature_dim = query.shape
        out = np.stack(out_list, axis=0).reshape((n_tasks, q_shot))
        self.record_info(y_q=y_q, preds_q=out)