#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
Hidden Markov Chain Building (HMCB)
adapted from https://github.com/maximtrp/mchmm
"""

import pandas as pd
import numpy as np
import scipy as sp
from itertools import product, chain, permutations

class HMCB:

    def __init__(self):
        """
        INPUT:
            N/A

        OUTPUT:
            N/A
        """


    def transition_matrix(self, seq_array):
        """
        Computes the transition frequency matrix of the observed data

        INPUT:
            seq_array (numpy.ndarray): array of sequences of events observed in
                                       the data.

        OUTPUT:
            freq_matrix (numpy.ndarray): transition frequency matrix
                                         of the observed data
        """

        signs = self.signals
        freq_matrix = np.zeros((len(signs), len(signs)))
        for row in seq_array:
            row_array = np.array(row)
            for x, y in product(range(len(signs)), repeat=2):
                xid = np.argwhere(row_array==signs[x]).flatten()
                yid = xid + 1
                yid = yid[yid < len(row_array)]
                s = np.count_nonzero(row_array[yid] == signs[y])
                freq_matrix[x, y] += s

        return freq_matrix


    def from_pandas(self, df, df_col):
        """
        Build a first-order Markov Chain from observations in the data. Defines, signals,
        transition probabilites.
        Takes data ONLY from a pandas dataframe

        INPUT:
            df (pandas df): pandas dataframe containing one unique session (or user)
                            per row. The sequence of events should be one continuous
                            string of event in one of the columns of the dataframe.

            df_col (string): column from the pandas df in which the series of events
                             is.

        OUTPUT:
            HMCB class instance
        """

        # signals list
        seq = []
        for i in range(df.shape[0]):
            try:
                seq.append(list(df[df_col].values[i]))
            except:
                print(i)
                print(df[df_col].values[i])
                continue
        self.seq_array = np.array(seq, dtype=object)

        self.signals = np.unique(list(chain.from_iterable(self.seq_array)))

        # observed transitions frequency matrix
        self.observed_freq_matrix = self.transition_matrix(self.seq_array)
        self._obs_row_totals = np.sum(self.observed_freq_matrix, axis=1)

        # observed transition probability matrix
        # (or alternatively, ormalized observed transition frequency matrix)
        self.observed_p_matrix = np.nan_to_num(
            self.observed_freq_matrix / self._obs_row_totals[:, None]
        )

        return self


    def n_order_matrix(self, transition_matrix, order):
        """
        Computes the Nth order transition probability matrix

        INPUT:
            transition_matrix (numpy.ndarray): first order transition probability
                                               matrix from transition_matrix function

            order (int): target order of the transition probability matrix

        OUTPUT:
            n_order_mat (numpy.ndarray): Nth order transition probability matrix
        """

        return np.linalg.matrix_power(transition_matrix, order)


class HOHMCB:
    def __init__(self, order):
        """
        INPUT:
            order (int): order of the Markov Chain to build

        OUTPUT:
            N/A

        NOTE: HOHMCB class assumes that the length of the data is at least of order + 1,
              otherwise, the class will return an error.
        """

        self.order = order


    def transition_matrix(self, seq_array):
        """
        Computes the transition frequency matrix of the observed data

        INPUT:
            seq_array (numpy.ndarray): array of sequences of events observed in
                                       the data.

        OUTPUT:
            freq_matrix (numpy.ndarray): transition frequency matrix
                                         of the observed data
        """

        signs = self.signals
        freq_matrix = np.zeros((len(signs), len(signs)))
        for row in seq_array:
            row_array = np.array(row)
            for x, y in product(range(len(signs)), repeat=2):
                xid = np.argwhere(row_array==signs[x]).flatten()
                yid = xid + 1
                yid = yid[yid < len(row_array)]
                s = np.count_nonzero(row_array[yid] == signs[y])
                freq_matrix[x, y] += s

        return freq_matrix


    def from_pandas(self, df, df_col):
        """
        Build a higher-order Markov Chain from observations in the data. Defines, signals,
        transition probabilites.
        Takes data ONLY from a pandas dataframe

        INPUT:
            df (pandas df): pandas dataframe containing one unique session (or user)
                            per row. The sequence of events should be one continuous
                            string of event in one of the columns of the dataframe.

            df_col (string): column from the pandas df in which the series of events
                             is.

        OUTPUT:
            HOHMCB class instance
        """

        # signals list
        seq = []
        for i in range(df.shape[0]):
            try:
                seq.append(list(df[df_col].values[i]))
            except:
                print(i)
                print(df[df_col].values[i])
                continue
        self.seq_array = np.array(seq, dtype=object)
        seq_flat = []
        for i in seq:
            seq_flat.append(i)
        self.seq_flat = seq_flat

        self.signals = list(permutations(np.unique(list(chain.from_iterable(self.seq_array))), self.order))

        # observed transitions frequency matrix
        self.observed_freq_matrix = self.transition_matrix(self.seq_array)
        self._obs_row_totals = np.sum(self.observed_freq_matrix, axis=1)

        # observed transition probability matrix
        # (or alternatively, ormalized observed transition frequency matrix)
        self.observed_p_matrix = np.nan_to_num(
            self.observed_freq_matrix / self._obs_row_totals[:, None]
        )

        return self


    def n_order_matrix(self, transition_matrix, order):
        """
        Computes the Nth order transition probability matrix

        INPUT:
            transition_matrix (numpy.ndarray): first order transition probability
                                               matrix from transition_matrix function

            order (int): target order of the transition probability matrix

        OUTPUT:
            n_order_mat (numpy.ndarray): Nth order transition probability matrix
        """

        return np.linalg.matrix_power(transition_matrix, order)
