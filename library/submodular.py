# coding: utf-8
"""
劣モジュラ最適化によって文書要約をするためのクラス
"""
from filer2.filer2 import Filer
from igraph import *
import collections
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.spatial.distance import cosine, sqeuclidean
import itertools

class SubModular(object):
    def __init__(self, list_bag, directed=True):
        # 入力のリスト(n, 2)のリスト形式
        # 有向グラフで計算するか、無向グラフで計算するか
        self._directed = directed
        # inputfileからnode, edge, weightを計算
        self._list_node, self._list_edge, self._list_weight = self._cal_node_edge_weight(list_bag)
        # 単語のword_idのdictを作成する
        self._dict_word_id = {word: i for i, word in enumerate(self._list_node)}
    
    # 入力されたエッジリストから、list_node, list_edge, list_weightを計算する
    def _cal_node_edge_weight(self, list_bag):
        """
        list_bag: bag_of_words
        list_node: nodeのリスト
        list_edge: edgeのリスト
        list_weight: weightのリスト
        """
        # 有向グラフの場合
        if self._directed == True:
            list_edgelist = self._cal_bag_edgelist(list_bag)
            list_edge = [tuple(row) for row in list_edgelist]
            # ノードリスト
            list_node = list(set([word for row in list_edgelist for word in row]))
            # エッジリストとそのweightを作成
            tuple_edge, tuple_weight = zip(*collections.Counter(list_edge).items())

            return list_node, list(tuple_edge), list(tuple_weight)
        
        else:
            list_edgelist = self._cal_bag_edgelist(list_bag)
            # 有向エッジリストを無向エッジリストに変換する
            list_edge = [tuple(sort(row)) for row in list_edgelist]
            # ノードリスト
            list_node = list(set([word for row in list_edgelist for word in row]))
            # エッジリストとそのweightを作成
            tuple_edge, tuple_weight = zip(*collections.Counter(list_edge).items())

            return list_node, list(tuple_edge), list(tuple_weight)
        
    def _cal_bag_edgelist(self, list_bag):
        """
        bag_of_wordsをedgelistに変換する
        list_bag: bag_of_words
        """
        # 有向グラフの場合
        if self._directed == True:
            list_edgelist = []
            for row in list_bag:
                list_tmp = [[row[i], row[i+1]] for i in range(len(row)-1)]
                list_edgelist.extend(list_tmp)
            return list_edgelist
        # 無向グラフの場合
        else:
            list_edgelist = []
            for row in list_bag:
                list_edgelist.extend(list(itertools.combinations(tuple(row),2)))
            return list_edgelist
        
    # ノード間の距離を計算する
    def _cal_matrix_path_out(self, inverse_flag=True, weight=5):
        """
        list_node: nodeのリスト
        list_edge: edgeのリスト
        list_weight: 重みのリスト
        weight: flag=Falseの場合、重みをいくつ履かせるか
        inverse_flag: ノード間の距離を計算するときに、inverseにするか、マイナスにするかのflag
        matrix: 計算後のノード間の距離が記録されたmatrix, (i,j)成分はノードiからノードjへの距離を表している。
        """
        
        g = Graph(directed=self._directed)
        g.add_vertices(self._list_node)
        g.add_edges(self._list_edge)
        # pathをマイナスにするか、inverseにするか
        if inverse_flag == True:
            list_out_weight = np.float(1)/np.array(self._list_weight)
        else:
            list_out_weight = np.max(self._list_weight)+weight - np.array(self._list_weight)
        # sparse_matrixの定義
        matrix = lil_matrix((len(self._list_node), len(self._list_node)))
        # ノード間の距離を計算する
        for i, node in enumerate(self._list_node):
            list_path = g.get_shortest_paths(node, self._list_node,
                                             mode=OUT, weights=self._list_weight,
                                             output='epath')
            for j, row in enumerate(list_path):
                if len(row) > 0:
                    matrix[i, j] = np.sum(list_out_weight[row])
        return matrix.tocsr()


# ベクトル操作をするためのクラス
class Vector(object):
    
    @staticmethod
    def cal_matrix_cos(list_word, dict_word_vec):
        """
        cosine類似度で単語間の距離を計算
        list_word: 単語のリスト
        dict_word_vec: 単語がkey, 分散表現がvalueのdict
        return: cos_matrix: 各単語間の類似度を計算したmatrix
        """
        cos_matrix = np.zeros((len(list_word), len(list_word)))
        for i, word1 in enumerate(list_word):
            for j, word2 in enumerate(list_word):
                cos_matrix[i][j] = cosine(dict_word_vec[word1], dict_word_vec[word2])
                
        return cos_matrix
    
    @staticmethod
    def cal_matrix_euc(list_word, dict_word_vec):
        """
        sqe_euclid類似度で単語間の距離を計算
        list_word: 単語のリスト
        dict_word_vec: 単語がkey, 分散表現がvalueのdict
        return: cos_matrix: 各単語間の類似度を計算したmatrix
        """
        euc_matrix = np.zeros((len(list_word), len(list_word)))
        for i, word1 in enumerate(list_word):
            for j, word2 in enumerate(list_word):
                euc_matrix[i][j] = sqeuclidean(dict_word_vec[word1], dict_word_vec[word2])
                
        return euc_matrix