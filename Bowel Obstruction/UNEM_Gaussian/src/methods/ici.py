from typing import Tuple, List
import torch
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import math
import numpy as np
from sklearn.linear_model import ElasticNet
from tqdm import tqdm
import time


class ICI(object):
    def __init__(self, model, device, log_file, args):
        super().__init__()
        self.model = model
        self.device = device
        self.step = args.step
        self.max_iter = args.max_iter
        self.reduce = args.reduce
        self.d = args.d
        self.C = args.C
        self.n_ways = args.n_ways
        self.initial_embed(args.reduce, args.d)
        self.initial_classifier(args.classifier)
        self.init_info_lists()
        self.elasticnet = ElasticNet(alpha=1.0, l1_ratio=1.0, fit_intercept=True, normalize=True, warm_start=True, selection="cyclic")
        self.dataset = args.dataset
        self.used_set_support = args.used_set_support

    def init_info_lists(self):
        self.test_acc = []

    def record_info(self, probs_q, y_q):
        """
        inputs:
            y_q : torch.Tensor of shape [n_task, q_shot] :
        """
        preds_q = probs_q.argmax(2)
        accuracy = (preds_q.cpu() == y_q.cpu()).float().mean(1, keepdim=True)
        self.test_acc.append(accuracy)

    def get_logs(self):
        self.test_acc = torch.cat(self.test_acc, dim=1).cpu().numpy()
        return {'acc': self.test_acc}

    def run_task(self, task_dic, shot):
        """
        inputs:
            task_dic : dictionnary with n_tasks few-shot tasks
            shot : scalar, number of shots
        """

        # Extract support and query
        y_s = task_dic['y_s']               # [n_task, shot]
        y_q = task_dic['y_q']               # [n_task, n_query]
        support = task_dic['x_s']           # [n_task, shot, feature_dim]
        query = task_dic['x_q']             # [n_task, n_query, feature_dim]

        # Perform normalizations required
        y_s = y_s.long().squeeze(2).to(self.device)
        y_q = y_q.long().squeeze(2).to(self.device)
        support = F.normalize(support, dim=2).to(self.device)
        query = F.normalize(query, dim=2).to(self.device)

        # Run methof
        self.run_method(support, query, y_s, y_q)

        # Extract adaptation logs
        logs = self.get_logs()

        return logs

    def run_method(self, support_features, query_features, support_labels, query_labels, **kwargs):

        probs_q = []
        n_tasks, _, _ = support_features.size()

        for i in tqdm(range(n_tasks)):

            support_X, support_y = support_features.cpu().numpy()[i], support_labels.cpu().numpy()[i]
            way, num_support = support_labels.unique().size(0), len(support_X)

            query_X = query_features.cpu().numpy()[i]
            unlabel_X = query_X
            num_unlabel = unlabel_X.shape[0]
            embeddings = np.concatenate([support_X, unlabel_X])
            X = self.embed(embeddings)
            H = np.dot(np.dot(X, np.linalg.inv(np.dot(X.T, X))), X.T)
            X_hat = np.eye(H.shape[0]) - H

            if self.max_iter == "auto":
                # set a big number
                self.max_iter = num_support + num_unlabel
            elif self.max_iter == "fix":
                self.max_iter = math.ceil(num_unlabel / self.step)
            else:
                assert float(self.max_iter).is_integer()

            support_set = np.arange(num_support).tolist()

            # Train classifier
            self.classifier.fit(support_X, support_y)

            for _ in range(self.max_iter):
                # Get pseudo labels
                pseudo_y = self.classifier.predict(unlabel_X)
                y = np.concatenate([support_y, pseudo_y])
                Y = self.label2onehot(y, way)
                y_hat = np.dot(X_hat, Y)

                # Expand based on credibility of pseudo labels
                support_set = self.expand(support_set, X_hat, y_hat, way, num_support, pseudo_y, embeddings, y)
                y = np.argmax(Y, axis=1)

                # Re-train classifier
                self.classifier.fit(embeddings[support_set], y[support_set])
                if len(support_set) == len(embeddings):
                    break

            prob_q = self.classifier.predict_proba(query_X)
            probs_q.append(prob_q)
            
        probs_q = torch.from_numpy(np.array(probs_q))

        self.record_info(probs_q=probs_q, y_q=query_labels)


    def expand(self, support_set, X_hat, y_hat, way, num_support, pseudo_y, embeddings, targets):

        # Get the path (i.e the evolution of |gamma_i| as a function of lambda increasing)
        _, coefs, _ = self.elasticnet.path(X_hat, y_hat, l1_ratio=1.0)
        coefs = np.sum(np.abs(coefs.transpose(2, 1, 0)[::-1, num_support:, :]), axis=2)
        selected = np.zeros(way)
        for gamma in coefs:
            for i, g in enumerate(gamma):
                if (
                    g == 0.0
                    and (i + num_support not in support_set)
                    and (selected[pseudo_y[i]] < self.step)
                ):
                    support_set.append(i + num_support)
                    selected[pseudo_y[i]] += 1
            if np.sum(selected >= self.step) == way:
                break
        return support_set

    def initial_embed(self, reduce, d):
        reduce = reduce.lower()
        assert reduce in ["isomap", "ltsa", "mds", "lle", "se", "pca", "none"]
        if reduce == "isomap":
            from sklearn.manifold import Isomap

            embed = Isomap(n_components=d)
        elif reduce == "ltsa":
            from sklearn.manifold import LocallyLinearEmbedding

            embed = LocallyLinearEmbedding(n_components=d, n_neighbors=5, method="ltsa")
        elif reduce == "mds":
            from sklearn.manifold import MDS

            embed = MDS(n_components=d, metric=False)
        elif reduce == "lle":
            from sklearn.manifold import LocallyLinearEmbedding

            embed = LocallyLinearEmbedding(
                n_components=d, n_neighbors=5, eigen_solver="dense"
            )
        elif reduce == "se":
            from sklearn.manifold import SpectralEmbedding

            embed = SpectralEmbedding(n_components=d)
        elif reduce == "pca":
            from sklearn.decomposition import PCA
            embed = PCA(n_components=d)

        if reduce == "none":
            self.embed = lambda x: x
        else:
            self.embed = lambda x: embed.fit_transform(x)

    def initial_classifier(self, classifier):
        assert classifier in ["lr", "svm"]
        if classifier == "svm":
            self.classifier = SVC(C=self.C, gamma="auto", kernel="linear", probability=True)
        elif classifier == "lr":
            self.classifier = LogisticRegression(
                C=self.C, multi_class="auto", solver="lbfgs", max_iter=1000
            )

    def label2onehot(self, label, num_class):
        result = np.zeros((label.shape[0], num_class))
        for ind, num in enumerate(label):
            result[ind, num] = 1.0
        return result
