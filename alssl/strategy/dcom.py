import numpy as np
import pandas as pd
import torch
from scipy.special import softmax
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from tqdm import tqdm

from ..data.base import ALDataModule
from ..model.base import BaseALModel
from .base import BaseStrategy
from .utils import predict


class DCoMStrategy(BaseStrategy):
    K_LOGISTIC=50
    A_LOGISTIC=0.8
    DELTA_RESOLUTION=0.05
    MAX_DELTA=1.1
    INITIAL_DELTA=0.75

    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.lSet_deltas = None

    def construct_graph_excluding_lSet(self, features, delta=None, batch_size=700):
        """
        Creates a directed graph where:
        x -> y if l2(x, y) < delta.
        Considers all images, but does not reference or delete edges in lSet.

        The graph is represented by a list of edges (a sparse matrix) and stored in a DataFrame.
        """
        if delta is None:
            delta = self.delta_avg

        xs, ys, ds = [], [], []
        print(f'Start constructing graph using delta={delta}')
        # distance computations are done in GPU
        cuda_feats = torch.tensor(features).cuda()
        for i in range(len(features) // batch_size):
            # distance comparisons are done in batches to reduce memory consumption
            cur_feats = cuda_feats[i * batch_size: (i + 1) * batch_size]
            dist = torch.cdist(cur_feats, cuda_feats)
            mask = dist < delta
            # saving edges using indices list - saves memory.
            x, y = mask.nonzero().T
            xs.append(x.cpu() + batch_size * i)
            ys.append(y.cpu())
            ds.append(dist[mask].cpu())

        xs = torch.cat(xs).numpy()
        ys = torch.cat(ys).numpy()
        ds = torch.cat(ds).numpy()

        df = pd.DataFrame({'x': xs, 'y': ys, 'd': ds})
        print(f'Before delete lSet neighbors: Graph contains {len(df)} edges.')
        return df
    
    def construct_graph(self, features, train_ids, delta=None, batch_size=700):
        """
         Creates a directed graph where:
         x -> y if l2(x, y) < delta, and deletes the covered points using lSet_deltas.

         Deletes edges to the covered samples (samples that are covered by lSet balls)
         and deletes all the edges from lSet.

         The graph is represented by a list of edges (a sparse matrix) and stored in a DataFrame.
         """
        if delta is None:
            delta = self.delta_avg

        df = self.construct_graph_excluding_lSet(features, delta, batch_size)

        # removing incoming edges to all cover from the existing labeled set
        edges_from_lSet_in_ball = np.isin(df.x, np.arange(len(train_ids))) & (df.d < df.x.map(self.lSet_deltas_dict))
        covered_samples = df.y[edges_from_lSet_in_ball].unique()

        edges_to_covered_samples = np.isin(df.y, covered_samples)
        all_edges_from_lSet = np.isin(df.x, np.arange(len(train_ids)))

        mask = all_edges_from_lSet | edges_to_covered_samples  # all the points inside the balls
        df_filtered = df[~mask]

        print(f'Finished constructing graph using delta={delta}')
        print(f'Graph contains {len(df_filtered)} edges.')
        return df_filtered, covered_samples
    
    def select_ids(self, model: nn.Module, dataset: ALDataModule, budget: int, almodel: BaseALModel, *args) -> list:
        all_ids = np.concatenate([dataset.train_ids, np.array(dataset.get_unlabeled_ids())])
        existing_indices = np.arange(len(dataset.train_ids))

        m = almodel.get_lightning_module()(**almodel.get_hyperparameters())
        _, _, features_unlabeled = predict(
            m,
            dataset.unlabeled_dataloader(), 
            scoring="none", desc="DCoM strategy (unlabeled)")
        
        _, _, features_train = predict(
            m,
            dataset.train_dataloader(), 
            scoring="none", desc="DCoM strategy (train)")
        
        features = np.concatenate([features_train, features_unlabeled])

        def get_competence_score(coverage):
            """
            Implementation of the logistic function weighting.
            """
            k = self.K_LOGISTIC  # the logistic growth rate or steepness of the curve
            a = self.A_LOGISTIC  # the logistic function center
            p = (1 + np.exp(-k * (1 - a)))
            competence_score = p / (1 + np.exp(-k * (coverage - a)))
            print(f'a = {a}, k = {k}, p = {round(p, 4)}')
            print("The coverage over the graph is: ", coverage)
            print("The competence_score is: ", round(competence_score, 3))
            return round(competence_score, 3)

        print(f"\n==================== Start DCoM Active Sampling ====================")
        # Calculate the current coverage
        selected = []
        
        margin, y_gt_train, y_preds_all = self.calculate_margin(model, dataset)
        margin[0: len(dataset.train_ids)] = 0 # We define the margin score to be 1-margin (as described in our paper)

        if self.lSet_deltas is None:
            self.lSet_deltas = [self.INITIAL_DELTA,] * len(dataset.train_ids)
            self.lSet_deltas_dict = dict(zip(np.arange(len(dataset.train_ids)), self.lSet_deltas))
        
        all_labels = np.take(dataset.full_train_dataset.targets, all_ids)
        self.lSet_deltas[-budget:] = self.new_centroids_deltas(
            features, y_gt_train, y_preds_all, budget, all_ids, dataset.train_ids, all_labels=all_labels)   
        
        self.delta_avg = np.average(self.lSet_deltas)
        
        fully_graph = self.construct_graph_excluding_lSet(features, self.MAX_DELTA)
        current_coverage = DCoMStrategy.calculate_coverage(fully_graph, dataset.train_ids, self.lSet_deltas_dict, len(all_ids))
        del fully_graph

        competence_score = get_competence_score(current_coverage)

        cur_df, covered_samples = self.construct_graph(features, dataset.train_ids, self.delta_avg)

        for i in range(budget): # The active selection
            coverage = len(covered_samples) / len(all_ids)

            if len(cur_df) == 0:
                ranks = np.zeros(len(dataset.train_ids))
            else:
                # calculate density for each point
                ranks = self.calculate_density(cur_df, all_ids)

            cur_selection = DCoMStrategy.normalize_and_maximize(ranks, margin, 1, lambda r, e: competence_score * e + (1 - competence_score) * r)[0]
            print(f'Iteration is {i}.\tGraph has {len(cur_df)} edges.\tCoverage is {coverage:.3f}. \tCurr choice is {cur_selection}. \tcompetence_score={competence_score}')

            new_covered_samples = cur_df.y[(cur_df.x == cur_selection)].values
            assert len(np.intersect1d(covered_samples, new_covered_samples)) == 0, 'all samples should be new'

            cur_df = cur_df[(~np.isin(cur_df.y, new_covered_samples))]  # Delete all the edges to the covered samples
            covered_samples = np.concatenate([covered_samples, new_covered_samples])
            margin[cur_selection] = 0
            selected.append(cur_selection)
        assert len(selected) == budget, 'added a different number of samples'
        assert len(np.intersect1d(selected, existing_indices)) == 0, 'should be new samples'
        next_initial_deltas = [np.average(self.lSet_deltas),] * budget
        self.lSet_deltas.extend(next_initial_deltas) 
        return all_ids[selected]
    
    def calculate_density(self, df, all_ids):
        rank_mapping = pd.DataFrame(df.groupby('x')['y'].count())
        all_indices_df = pd.DataFrame(index=np.arange(len(all_ids)))
        result_df = pd.merge(all_indices_df, rank_mapping, left_index=True, right_index=True, how='left').fillna(0)
        return np.array(result_df)
    
    def calculate_margin(self, model: nn.Module, dataset: ALDataModule):
        _, y_preds_unlabeled, _ = predict(
            model,
            dataset.unlabeled_dataloader(), 
            scoring="none", desc="DCoM strategy (unlabeled)")
        
        y_gt_train, y_preds_train, _ = predict(
            model,
            dataset.train_dataloader(), 
            scoring="none", desc="DCoM strategy (train)")

        proba_unlabeled = softmax(y_preds_unlabeled, 1)
        proba_train = softmax(y_preds_train, 1)

        proba = np.concatenate([proba_train, proba_unlabeled])
        sorted_proba = np.sort(proba, axis=1)[:, ::-1]  # Descending order
        difference = sorted_proba[:, 0] - sorted_proba[:, 1]
        # for code consistency across uncertainty, entropy methods i.e., picking datapoints with max value
        ranks = -1 * difference
        # ranks = difference

        margin_result = np.array(ranks).reshape(-1, 1)
        scaler = MinMaxScaler()
        normalized_margin_result = scaler.fit_transform(margin_result)
        final_margin_result = np.array(normalized_margin_result.flatten().tolist())
        return final_margin_result, y_gt_train, np.concatenate([y_preds_train, y_preds_unlabeled]).argmax(-1)

    @staticmethod
    def normalize_and_maximize(param1_list, param2_list, amount, target_func):
        """
        Perform pre-processing on each list and apply the target function on them.
        """
        param1_arr = np.array(param1_list).reshape(-1, 1)
        param2_arr = np.array(param2_list).reshape(-1, 1)

        # Min-Max normalization using scikit-learn's MinMaxScaler
        scaler = MinMaxScaler()
        param1_normalized = scaler.fit_transform(param1_arr)
        param2_normalized = scaler.fit_transform(param2_arr)

        # Calculate the product using the provided target_func
        product_array = target_func(param1_normalized.flatten(), param2_normalized.flatten())

        sorted_indices = np.argsort(product_array)[::-1]

        return sorted_indices[:amount]

    @staticmethod
    def calculate_coverage(fully_df, lSet, lSet_deltas_dict, total_data_len):
        """
        Return the current probability coverage.
        """
        covered_samples = fully_df.y[np.isin(fully_df.x, np.arange(len(lSet))) & (
                fully_df.d < fully_df.x.map(lSet_deltas_dict))].unique()  # lSet send arrow to them
        return len(covered_samples) / total_data_len
    
    def new_centroids_deltas(self, features, lSet_labels, pseudo_labels, budget, all_ids, train_ids, batch_size=500, all_labels=[]):
        """
        Performs binary search of the next delta values.
        """
        def calc_threshold(coverage):
            assert 0 <= coverage <= 1, f'coverage is not between 0 to 1: {coverage}'
            return 0.2 * coverage + 0.4

        def check_purity(df, cent_label, delta):
            # find all the neighbors
            edges_from_lSet = (df.x == cent_idx) & (df.d < delta)
            neighbors_idx = df.y[edges_from_lSet]

            # take their neighbors and compute the ball purity (Are the labels the same as the chosen point?)
            neighbors_pseudo_labels = list(map(pseudo_labels.__getitem__, neighbors_idx))
            neighbors_real_labels = list(map(all_labels.__getitem__, neighbors_idx))

            if len(neighbors_idx):
                pseudo_purity = sum(np.array(neighbors_pseudo_labels == cent_label)) / len(neighbors_idx)
                real_purity = sum(np.array(neighbors_real_labels == cent_label)) / len(neighbors_idx)
                print(f'real_purity: {real_purity}, pseudo_purity: {pseudo_purity}')
                return pseudo_purity
            return 0

        new_deltas = []
        df = self.construct_graph_excluding_lSet(features, self.MAX_DELTA, batch_size)

        fully_df = self.construct_graph_excluding_lSet(features, self.MAX_DELTA)
        covered_samples = fully_df.y[np.isin(fully_df.x, np.arange(len(train_ids))) & (
                fully_df.d < fully_df.x.map(self.lSet_deltas_dict))].unique()
        coverage = len(covered_samples) / len(all_ids)

        purity_threshold = calc_threshold(coverage)
        print("Current threshold: ", purity_threshold)

        for cent_idx, centroid in enumerate(train_ids):
            if cent_idx < len(train_ids) - budget:  # Not new points
                continue

            print(f'start calculation for cent_idx: {cent_idx}')
            low_del_val = 0
            max_del_val = self.MAX_DELTA
            mid_del_val = (low_del_val + max_del_val) / 2
            last_purity = 0
            last_delta = mid_del_val

            while abs(low_del_val - max_del_val) > self.DELTA_RESOLUTION:
                curr_purity = check_purity(df, cent_label=lSet_labels[cent_idx], delta=mid_del_val)
                print("centroid: ", centroid, ", idx: ", cent_idx, ". delta = ", mid_del_val, " and purity = ", curr_purity)

                if last_delta < mid_del_val and last_purity == purity_threshold and curr_purity < purity_threshold:
                    mid_del_val = last_delta
                    break

                if curr_purity < purity_threshold:
                    # if smaller than the threshold - try smaller delta
                    max_del_val = mid_del_val
                elif curr_purity >= purity_threshold:  # if bigger than threshold -> try bigger delta
                    low_del_val = mid_del_val

                last_purity = curr_purity
                last_delta = mid_del_val
                mid_del_val = (low_del_val + max_del_val) / 2

            curr_purity = check_purity(df, cent_label=lSet_labels[cent_idx], delta=mid_del_val)
            print("the chosen delta: ", mid_del_val, "and its purity: ", curr_purity)
            print("---------------------------------------------------------------------------")
            new_deltas.append(mid_del_val)

        self.lSet_deltas = [np.float32(delta) for delta in new_deltas]
        self.lSet_deltas_dict = dict(zip(np.arange(len(train_ids)), self.lSet_deltas))
        print("All new deltas: ", new_deltas, '\n')
        return new_deltas