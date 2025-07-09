"""
A module of sampling algorithms.
"""
#  Copyright (c) 2025.7.4, BM4Ckit.
#  Authors: Pu Pengxin, Song Xin
#  Version: 0.9a
#  File: sampling.py
#  Environment: Python 3.12

import numpy as np
import time
import copy


class SimpRandSamp(object):
    """
    Simple random sampling.

    Method:
        run: run sampling.

    Attributes:
        seed: int, the random seed.
        sampling_args: 1darray, the list of sampling points' indexes in X_all.
        sampling_points: 2darray, the sampling points.
    """

    def __init__(self, size, seed=None):
        """
        Parameters:
            size: int, the number of samples to take.
            seed: int, the random seed.
        """
        self.size = size
        self.seed = seed

    def run(self, X_all, ):
        """
        Run sampling.

        parameters:
            X_all: ndarray, the population i.e., the set of all possible points, each point responds to a row of X_all.
        """
        n_samp = len(X_all)
        np.random.seed(self.seed)
        self.sampling_args = np.random.choice(n_samp, self.size, replace=False)
        self.sampling_points = X_all[self.sampling_args]


class FPS(object):
    """
    The Furtherest Points Sampling.

    Method:
        distace: calculating distance matrix
        run: run sampling.

    Attributes:
        distance_matrix: 2darray, the distacne matrix of X_all.
        sampling_args: 1darray, the list of sampling points' indexes in X_all.
        sampling_points: 2darray, the sampling points.
    """

    def __init__(self, size, norm=2):
        """
        Parameters:
            size: int, the number of samples to take.
            norm: positive int|np.inf, the type of calculating distance.
        """
        self.sampling_points = None
        self.size = size
        self.norm = norm

    def distance(self, X_all):
        """
        Define the distance matrix of X_all.
        Returns: Distance Matrix
        """
        n_samp, n_feat = np.shape(X_all)
        self.distance_matrix = np.linalg.norm(X_all[:, None] - X_all[None, :], ord=self.norm, axis=-1)
        return self.distance_matrix

    def run(self, X_all: np.ndarray, X_init=None, init_type='points', verbose=2) -> np.ndarray:
        """
        Run sampling.

        Args:
            X_all: ndarray[N, D], the population i.e., the set of all feasible points, each point responds to a row of X_all.
            X_init: None|ndarray|1darray, the set of initial points (for init_type='points'), or the index set of initial points in X_all (for init_type='index'). If X_init is None, X_init is set to the point in X_all closest to center-of-X_all.
            init_type: 'points' or 'index', assign the type of input X_init.
            verbose: int, control print content. 0 for silence.

        Returns:
            the sampling points array
        """
        # Define initial varibles
        if X_all.ndim == 1:
            X_all = X_all[:, None]
        elif X_all.ndim != 2:
            raise ValueError(f'Expected X_all is 1 or 2 dimension, but got {X_all.ndim}-D array.')
        n_samp, n_feat = np.shape(X_all)
        all_args = set(range(n_samp))
        self.sampling_points = np.zeros((self.size, n_feat))
        self.sampling_args = np.zeros(self.size, dtype=np.int64)
        L = np.inf
        init_args = 0

        # If X_init is None, X_init is set to the point in X_all closest to center-of-X_all
        if X_init is None:
            if verbose > 0:
                print('X_init is not read, generating the X_init...\n')
            Xcenter = np.mean(X_all, axis=0)
            for i, feat in enumerate(X_all):
                L_temp = np.linalg.norm(feat - Xcenter, ord=self.norm)
                if L_temp < L:
                    L = L_temp
                    X_init = copy.deepcopy(feat)
                    init_args = i

        elif init_type == 'index':  # X_init is the indices in X_all
            # check input
            X_init = np.asarray(X_init, dtype=np.int64)
            if np.any((X_init < 0) + (X_init > n_samp)):
                raise ValueError('X_init is out of range')

            init_args = set(X_init.tolist())
            if verbose > 0:
                print('read the initial index in X_all successfully.\n')

        elif init_type == 'points':  # X_init is directly the coords
            # check input
            if len(np.shape(X_init)) == 1:
                X_init = np.array([X_init])
            elif len(np.shape(X_init)) != 2:
                raise ValueError('X_init must be 1D or 2D array')
            if len(X_init[0, :]) != n_feat:
                raise ValueError('Columns number of X_init and X_all does not match')
            #
            init_args = set()
            __init_indx = 0
            # print(X_init,'\n',init_args)
            # find the element args of X_init in X_all, by finding the element in X_init that have minimal distance of element in X_all
            for j, featInit in enumerate(X_init):
                L_temp = np.linalg.norm(X_all - featInit, ord=self.norm, axis=-1)
                __init_indx = np.where(L_temp < 1e-5)[0][0]
                init_args.add(__init_indx)
            if verbose > 0:
                print('read the initial index in X_all successfully.\n')

        else:
            raise ValueError('Invalid value of init_type. Please set init_type to \'index\' or \'points\'.')

        if len(init_args) + self.size > n_samp:
            raise ValueError('Sum of initial and added points is greater than X_all')

        if verbose > 0: print('Variables initialized, starting main loop...\n')

        # main loop
        # Calculate the distance matrix
        if verbose > 0: print('Calculating the distance matrix...\n')
        distMat = self.distance(X_all)

        # Find the furthest points
        if verbose > 0: print('Search the furthest points...')
        dist1 = np.inf
        dist2 = -1.
        samp_args = 0
        for k in range(self.size):
            rest_args = all_args - init_args
            for i in rest_args:
                for j in init_args:
                    i, j = int(i), int(j)
                    # Find the min distance for j
                    if distMat[i, j] <= dist1:
                        dist1 = distMat[i, j]
                # Find the max distance for i
                if dist1 >= dist2:
                    dist2 = dist1
                    samp_args = i
                # initialize
                dist1 = np.inf
            self.sampling_args[k] = int(samp_args)
            # initialize
            dist2 = -1.
            init_args.add(int(samp_args))

        if verbose > 0: print('*****Completed*****')
        self.sampling_points = X_all[self.sampling_args]
        return self.sampling_points


class Uncertainty_Samp(object):
    """
    Run sampling by uncertainty of models. The uncertainty is calculated by

    Method:
        run: run sampling.

    Attribute:
        train_mean_MSE: float, the mean value of training-MSE of all split-train-test loops
        valid_mean_MSE: float, the mean value of validation-MSE of all split-train-test loops
        predict_mean: 1darray, the mean values of predictions of X_all.
        uncertain_list = 1darray, the list of each samples' uncertainties given by predicted values' standard deviation of all split-train-test loops.
        uncertain_sort_args: 1darray, the list of indexes that sorted predictions from min to max of samples in X_all.
        sampling_args: 1darray, the list of sampling points' indexes in X_all.
    """

    def __init__(self, size, model, n_iter=100, seed=None):
        """
        Parameters:
        size: integer, number of features to sampling,
        model: The model (particularly in sklearn) with method `fit` and `predict` that to fit and predict data.
        n_iter: integer, number of iterations.
        seed: integer, rand seed.
        """

        self.size = size
        self.model = model
        self.n_iter = n_iter
        self.seed = seed

    def run(self, X, y, X_all, split_size=0.1, n_group=1, verbose=2):
        """
        X: np.ndarray, input features.
        y: np.ndarray, labels of dataset with the same order of y.
        X_all: ����Ԥ���������
        split_size: split size.
        n_group: ϵͳ����ʱƽ�����������
        verbose: Control print content. 0 for scilence.
        """
        # Check var
        n_samp, n_feat = np.shape(X)
        if n_samp != len(y):
            raise ValueError('Sample numbers of X and y must match!!!')
        elif self.size >= len(X):
            raise ValueError('Number of sampling points must less than total samples')

        if type(n_group) != int:
            raise TypeError('n_group must be an integer!!!')
        elif len(X) % n_group != 0:
            raise ValueError('Number of X cannot be divisible by n_group')
        elif 1 > split_size > 0:
            n_split = round(split_size * n_samp) // n_group
        elif split_size <= 0:
            raise ValueError('split_size must be an integer greater than 1, or a float between 0 and 1!!!')
        elif split_size <= n_group:
            raise ValueError('split_size must greater than n_group!!!')
        elif split_size >= 1:
            n_split = split_size // n_group

        # Initialize var
        X_pred = X_all
        self.sampling_points = np.zeros((self.size, n_feat))
        self.sampling_args = np.zeros(self.size, dtype=np.int64)
        self.uncertainty = np.zeros(self.size, dtype=np.int64)
        n_samples_in = len(X) // n_group
        n_feat = len(X[0, :])
        train_MSE_list = np.empty((self.n_iter,))
        valid_MSE_list = np.empty((self.n_iter,))
        pred_matrix = np.empty((len(X_pred), self.n_iter))
        if verbose > 0: print('Varibles initialized, entering main loop...\n')

        # MAIN LOOP
        group = np.reshape(X, (n_group, n_samples_in, n_feat))  # ����X_raw
        y_group = np.reshape(y, (n_group, n_samples_in))  # ����y_raw
        np.random.seed(self.seed)
        seed_list = np.random.randint(10000000, size=self.n_iter)
        for i in range(self.n_iter):
            np.random.seed(seed_list[i])
            arg_va = np.random.choice(n_samples_in, size=n_split, replace=False)
            arg_tr = np.delete(np.array(range(n_samples_in)), arg_va)
            # print(arg_va, '\n', arg_tr)
            X_train = np.empty((0, n_feat))
            X_valid = np.empty((0, n_feat))
            y_train = np.empty((0))
            y_valid = np.empty((0))
            for j in range(n_group):
                X_train = np.r_[X_train, group[j, arg_tr, :]]
                X_valid = np.r_[X_valid, group[j, arg_va, :]]
                y_train = np.r_[y_train, y_group[j, arg_tr]]
                y_valid = np.r_[y_valid, y_group[j, arg_va]]

            self.model.fit(X_train, y_train)  # Train the model
            train_MSE = np.sqrt(np.mean((self.model.predict(X_train) - y_train)**2))
            valid_MSE = np.sqrt(np.mean((self.model.predict(X_valid) - y_valid)**2))
            train_MSE_list[i] = copy.deepcopy(train_MSE)
            valid_MSE_list[i] = copy.deepcopy(valid_MSE)
            if verbose > 0: print('***iteration %i  train_MSE = %s  val_MSE = %s' % (i + 1, train_MSE, valid_MSE))
            # predict
            pred_matrix[:, i] = self.model.predict(X_pred)
        self.train_mean_MSE = np.mean(train_MSE_list)
        self.valid_mean_MSE = np.mean(valid_MSE_list)
        self.predict_mean = np.mean(pred_matrix, axis=1)
        self.uncertain_list = np.std(pred_matrix, axis=1)
        # sort
        self.uncertain_sort_args = np.argsort(self.uncertain_list)
        self.sampling_args = self.uncertain_sort_args[-self.size:]


