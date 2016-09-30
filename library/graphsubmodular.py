# coding: utf-8
"""
劣モジュラ最適化によって文書要約をするためのクラス
"""
from igraph import *
import collections
import numpy as np
from scipy.sparse import lil_matrix
import itertools
import copy
from filer2.filer2 import Filer


class GraphSubModular(object):
    def __init__(self, list_sep_all, list_sep, list_edgelist=None, directed=True, inverse_flag=True):
        # 有向グラフで計算するか、無向グラフで計算するか
        self._directed = directed
        # 距離を計算するときに、inverseにするのか、マイナスにするのか
        self._inverse_flag = inverse_flag

        # bag_of_words
        self._list_bag = list_sep
        self._list_bag_all = list_sep_all
        # inputfileからnode, edge, weightを計算
        if list_edgelist == None:
            self._list_node, self._list_edge, self._list_weight = self._cal_node_edge_weight(list_sep)
        else:
            list_node = list(set([word for row in self._list_bag for word in row]))
            self._list_node = list_node
            list_edge = [tuple(row) for row in list_edgelist]
            tuple_edge, tuple_weight = zip(*collections.Counter(list_edge).items())
            self._list_edge = list(tuple_edge)
            self._list_weight = list(tuple_weight)

        # 単語のword_idのdictを作成する
        self._dict_word_id = {word: i for i, word in enumerate(self._list_node)}
        # 距離行列の作成
        self._matrix = self._cal_matrix_path_out(inverse_flag=self._inverse_flag)
        # 全単語のリスト
        self._list_all_word = [word for row in self._list_bag for word in row]
        # 抽出した文章の集合
        self._list_C = []

    @property
    def dict_word_id(self):
        print 'property: dict_word_id'
        return self._dict_word_id

    @dict_word_id.setter
    def dict_word_id(self, value):
        print 'setter: dict_word_id'
        self._dict_word_id = value

    @dict_word_id.deleter
    def dict_word_id(self):
        print 'deleter: dict_word_id'
        del self._dict_word_id

    @property
    def matrix(self):
        print 'property: matrix'
        return self._matrix

    @matrix.setter
    def matrix(self, value):
        print 'setter: matrix'
        self._matrix = value

    @matrix.deleter
    def matrix(self):
        print 'deleter: matrix'
        del self._matrix

    @property
    def list_all_word(self):
        print 'property: list_all_word'
        return self._list_all_word

    @list_all_word.setter
    def list_all_word(self, value):
        print 'setter: list_all_word'
        self._list_all_word = value

    @list_all_word.deleter
    def list_all_word(self):
        print 'deleter: list_all_word'
        del self._list_all_word
        self._list_all_word = []

    @property
    def list_C(self):
        print 'property: list_C'
        return self._list_C

    @list_C.setter
    def list_C(self, value):
        print 'setter: list_C'
        self._list_C = value

    @list_C.deleter
    def list_C(self):
        print 'deleter: list_C'
        del self._list_C
        self._list_C = []

    # 入力されたエッジリストから、list_node, list_edge, list_weightを計算する
    def _cal_node_edge_weight(self, list_bag):
        """
        list_bag: bag_of_words
        list_node: nodeのリスト
        list_edge: edgeのリスト
        list_weight: weightのリスト
        """
        list_edgelist = self._cal_bag_edgelist(list_bag)
        # 有向エッジリストを無向エッジリストに変換する
        list_edge = [tuple(sorted(row)) for row in list_edgelist]
        # ノードリスト
        list_node = list(set([word for row in list_bag for word in row]))
        # エッジリストとそのweightを作成
        tuple_edge, tuple_weight = zip(*collections.Counter(list_edge).items())

        return list_node, list(tuple_edge), list(tuple_weight)

    def _cal_bag_edgelist(self, list_bag):
        """
        bag_of_wordsをedgelistに変換する
        list_bag: bag_of_words
        return: list_edgelist
        """
        list_edgelist = []
        for row in list_bag:
            list_edgelist.extend(list(itertools.combinations(tuple(row),2)))
        return list_edgelist

    # ノード間の距離を計算する
    def _cal_matrix_path_out(self, inverse_flag=True, weight=5, fill='max'):
        """
        list_node: nodeのリスト
        list_edge: edgeのリスト
        list_weight: 重みのリスト
        weight: flag=Falseの場合、重みをいくつ履かせるか
        inverse_flag: ノード間の距離を計算するときに、inverseにするか、マイナスにするかのflag
        fill: パスが存在しない場合に、なんの値を入れるか。
        matrix: 計算後のノード間の距離が記録されたmatrix, (i,j)成分はノードiからノードjへの距離を表している。
        """

        g = Graph(directed=self._directed)
        g.add_vertices(self._list_node)
        g.add_edges(self._list_edge)
        # pathをマイナスにするか、inverseにするか
        if inverse_flag == True:
            list_out_weight = np.float(1)/np.array(self._list_weight)
        else:
            list_out_weight = (np.max(self._list_weight)+weight - np.array(self._list_weight))/np.max(self._list_weight)
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

        # matrixをarray型に変換
        matrix = matrix.toarray()
        # maxの値を取得
        max_value = np.amax(matrix)
        # 0の要素にmax_valueを代入する
        matrix[matrix==0] = max_value
        # 対角成分は0にする（自分から自分への距離は0)
        for i, row in enumerate(matrix):
            matrix[i][i] = 0
        return matrix

    def _cal_cost(self, list_c_word, scale):
        """
        コストの計算
        :param list_c_word: 現在採用している文の中に含まれる単語
        :param distance_matrix: W * Vの距離行列
        :param scale: スケール関数に何を使うか, 0: e^x, 1: x, 2: ln_x
        :return: f_C (計算したコスト)
        """
        # 単語をidに変換
        list_c_id = sorted([self._dict_word_id[word] for word in list_c_word])
        f_C = 0.0
        # すべての単語を検索
        for word in self._list_all_word:
            # 行番号
            row_id = self._dict_word_id[word]
            # 対象の行の抜き出し
            row = self._matrix[row_id][list_c_id]
            # スケーリング関数: e^x
            if scale == 0:
                f_C -= np.exp(np.amin(row))
            # スケーリング関数: x
            elif scale == 1:
                f_C -= np.amin(row)
            # スケーリング関数: ln_x
            else:
                f_C -= np.log(np.amin(row))
        return f_C

    def _m_greedy_1(self, list_C, list_id_sep_sepall, r=1, scale=0):
        """
        修正貪欲法の一周分
        :param list_C: 現在採用している文のbag_of_words
        :param list_id_document: それ以外の採用候補
        :param distance_matrix: 距離行列, W * V
        :param r: 文字数に対するコストをどれだけかけるか
        :param scale: スケーリング関数、0: e^x, 1: x, 2: ln_x
        :return: doc: idとその単語のリスト
        """
        # list_Cが空の時
        if len(list_C) == 0:
            # 計算したスコアを記録するためのリスト
            list_id_score = []
            for doc_id, sep, sepall in list_id_sep_sepall:
                # documentに含まれる単語のリスト
                list_c_word = sorted(list(set([word for word in sep])))
                # スコアの計算
                f_C = self._cal_cost(list_c_word=list_c_word,
                                     scale=scale)
                f_C = f_C/(np.power(len(sepall), r))
                # リストにidとbagとスコアを記録
                list_id_score.append([doc_id, sep, sepall, f_C])
            # スコアが最大になるものを取得
            doc_id, sep, sepall, _ = sorted(list_id_score, key=lambda x: x[3], reverse=True)[0]
            return [doc_id, sep, sepall]
        # list_Cが空ではないとき
        else:
            # 現在のlist_Cに含まれるユニークな単語のリストを作成
            list_c_word = sorted(list(set([word for row in list_C for word in row[1]])))
            # f_C: 現在のコストの計算
            f_C = self._cal_cost(list_c_word=list_c_word,
                                 scale=scale)
            # 文書を一つずつ追加した時のスコアの増分を計算する
            list_id_score = []
            for doc_id, sep, sepall in list_id_sep_sepall:
                # 文章の追加
                list_c_word_s = list(set(list_c_word + sep))
                # コストの計算
                f_C_s = self._cal_cost(list_c_word=list_c_word_s,
                                       scale=scale)
                # スコアの増分を計算
                delta = (f_C_s - f_C) / np.power(len(sepall), r)
                # スコアの増分を記録
                list_id_score.append([doc_id, sep, sepall, delta])
            # スコアの増分が一番大きかったdocを返す
            doc_id, sep, sepall, _ = sorted(list_id_score, key=lambda x: x[3], reverse=True)[0]
            return [doc_id, sep, sepall]


    def m_greedy(self, num_w = 100, r=1, scale=0):
        """
        修正貪欲法による文章の抽出
        :param num_w: 単語数の制約
        :param r: 単語数に対するコストのパラメータ
        :param scale: スケーリング関数, 0: e^x, 1: x, 2: ln_x
        :return: 抽出した文章のidとそのbag_of_wordsのリスト
        """
        # list_id_documentの作成
        list_id_sep_sepall = [[i, row[0], row[1]] for i, row in enumerate(zip(self._list_bag, self._list_bag_all)) if len(row[0]) > 0]
        list_id_sep_sepall_copy = copy.deepcopy(list_id_sep_sepall)
        # 要約文書のリスト
        list_C = []
        # num_sで指定した文章数を抜き出すまで繰り返す
        C_word = 0
        while len(list_id_sep_sepall):
            # コストが一番高くなる組み合わせを計算
            doc_id, sep, sepall = self._m_greedy_1(list_C=list_C,
                                                   list_id_sep_sepall=list_id_sep_sepall,
                                                   r=r, scale=scale)
            if C_word + len(sepall) <= num_w:
                # 採用したリストをappend
                list_C.append([doc_id, sep, sepall])
                C_word += len(sepall)
            # 元の集合からremove
            list_id_sep_sepall.remove([doc_id, sep, sepall])

        list_id_score = []
        for doc_id, sep, sepall in list_id_sep_sepall_copy:
            # documentに含まれる単語のリスト
            list_c_word = sorted(list(set([word for word in sep])))
            # スコアの計算
            f_C = self._cal_cost(list_c_word=list_c_word,
                                 scale=scale)
            # リストにidとbagとスコアを記録
            list_id_score.append([doc_id, sep, sepall, f_C])
        # スコアが最大になるものを取得
        doc_id, sep, sepall, max_f = sorted(list_id_score, key=lambda x: x[3], reverse=True)[0]

        list_c_word = sorted(list(set([word for row in list_C for word in row[1]])))
        f_C = self._cal_cost(list_c_word=list_c_word, scale=scale)

        if f_C >= max_f:
            self._list_C = list_C
        else:
            self._list_C = [[doc_id, sep, sepall]]

        print '計算が終了しました'


class Sentence_GraphSubModular(GraphSubModular):
    """
    グラフ間の構造を用いて類似度を測定して、劣モジュラ最適化を行う
    """
    def __init__(self, list_sep_all, list_sep, list_edgelist=None, directed=True, inverse_flag=True):
        super(Sentence_GraphSubModular, self).__init__(list_sep_all, list_sep, list_edgelist, directed, inverse_flag)
        self._d_matrix = self._cal_distance_matrix().toarray()

    def _cal_distance_matrix(self):
        """
        文書iから文書jへの距離を計算するためのメソッド
        :return:
        """
        dict_word_id = self._dict_word_id
        matrix = self.matrix
        d_matrix = lil_matrix((len(self._list_bag), len(self._list_bag)))
        for i, row1 in enumerate(self._list_bag):
            list_id1 = [dict_word_id[word] for word in row1]
            for j, row2 in enumerate(self._list_bag):
                list_id2 = [dict_word_id[word] for word in row2]
                matrix_rev = matrix[list_id1][:, list_id2]
                d_matrix[i, j] = np.average(np.amin(matrix_rev, axis=1))

        return d_matrix

    def _cal_cost(self, list_C_id, scale):

        f_C = 0.0
        # すべての単語を検索
        for row_id in range(len(self._list_bag)):
            # 対象の行の抜き出し
            row = self._d_matrix[row_id][list_C_id]
            # スケーリング関数: e^x
            if scale == 0:
                f_C -= np.exp(np.amin(row))
            # スケーリング関数: x
            elif scale == 1:
                f_C -= np.amin(row)
            # スケーリング関数: ln_x
            else:
                f_C -= np.log(np.amin(row))
        return f_C

    def _m_greedy_1(self, list_C_id, list_id_sep_sepall, r=1, scale=0):
        """
        修正貪欲法の一周分
        :param list_C: 現在採用している文のbag_of_words
        :param list_id_document: それ以外の採用候補
        :param distance_matrix: 距離行列, W * V
        :param r: 文字数に対するコストをどれだけかけるか
        :param scale: スケーリング関数、0: e^x, 1: x, 2: ln_x
        :return: doc: idとその単語のリスト
        """
        # list_Cが空の時
        if len(list_C_id) == 0:
            # 計算したスコアを記録するためのリスト
            list_id_score = []
            for doc_id, sep, sepall in list_id_sep_sepall:
                # スコアの計算
                f_C = self._cal_cost(list_C_id=[doc_id],
                                     scale=scale)
                f_C = f_C/(np.power(len(sepall), r))
                # リストにidとbagとスコアを記録
                list_id_score.append([doc_id, sep, sepall, f_C])
            # スコアが最大になるものを取得
            doc_id, sep, sepall, _ = sorted(list_id_score, key=lambda x: x[3], reverse=True)[0]
            return [doc_id, sep, sepall]
        # list_Cが空ではないとき
        else:
            # f_C: 現在のコストの計算
            f_C = self._cal_cost(list_C_id=list_C_id,
                                 scale=scale)
            # 文書を一つずつ追加した時のスコアの増分を計算する
            list_id_score = []
            for doc_id, sep, sepall in list_id_sep_sepall:
                # コストの計算
                f_C_s = self._cal_cost(list_C_id=list_C_id+[doc_id],
                                       scale=scale)
                # スコアの増分を計算
                delta = (f_C_s - f_C) / np.power(len(sepall), r)
                # スコアの増分を記録
                list_id_score.append([doc_id, sep, sepall, delta])
            # スコアの増分が一番大きかったdocを返す
            doc_id, sep, sepall, _ = sorted(list_id_score, key=lambda x: x[3], reverse=True)[0]
            return [doc_id, sep, sepall]


    def m_greedy(self, num_w = 100, r=1, scale=0):
        """
        修正貪欲法による文章の抽出
        :param num_w: 単語数の制約
        :param r: 単語数に対するコストのパラメータ
        :param scale: スケーリング関数, 0: e^x, 1: x, 2: ln_x
        :return: 抽出した文章のidとそのbag_of_wordsのリスト
        """
        # list_id_documentの作成
        list_id_sep_sepall = [[i, row[0], row[1]] for i, row in enumerate(zip(self._list_bag, self._list_bag_all)) if len(row[0]) > 0]
        list_id_sep_sepall_copy = copy.deepcopy(list_id_sep_sepall)
        # 要約文書のリスト
        list_C = []
        list_C_id = []
        # num_sで指定した文章数を抜き出すまで繰り返す
        C_word = 0
        while len(list_id_sep_sepall):
            # コストが一番高くなる組み合わせを計算
            doc_id, sep, sepall = self._m_greedy_1(list_C_id=list_C_id,
                                                   list_id_sep_sepall=list_id_sep_sepall,
                                                   r=r, scale=scale)
            if C_word + len(sepall) <= num_w:
                # 採用したリストをappend
                list_C.append([doc_id, sep, sepall])
                list_C_id.append(doc_id)
                C_word += len(sepall)
            # 元の集合からremove
            list_id_sep_sepall.remove([doc_id, sep, sepall])

        list_id_score = []
        for doc_id, sep, sepall in list_id_sep_sepall_copy:
            # スコアの計算
            f_C = self._cal_cost(list_C_id=[doc_id],
                                 scale=scale)
            # リストにidとbagとスコアを記録
            list_id_score.append([doc_id, sep, sepall, f_C])
        # スコアが最大になるものを取得
        doc_id, sep, sepall, max_f = sorted(list_id_score, key=lambda x: x[3], reverse=True)[0]

        f_C = self._cal_cost(list_C_id=list_C_id, scale=scale)

        if f_C >= max_f:
            self._list_C = list_C
        else:
            self._list_C = [[doc_id, sep, sepall]]

        print '計算が終了しました'


class Modified_GraphSubModular(object):
    """
    劣モジュラ関数を正の関数に置き換える
    """
    def __init__(self, list_sep_all, list_sep, list_edgelist=None, directed=True, simrank_flag=True, weighted=False,  log_flag=True):
        """

        :param list_sep_all:
        :param list_sep:
        :param list_edgelist:
        :param directed:
        :param simrank_flag:
        :param log_flag:
        """
        # 有向グラフで計算するか、無向グラフで計算するか
        self._directed = directed
        # 距離を計算するときに、simrankで計算するかそれとも平均ノード次数で計算するか
        self._simrank_flag = simrank_flag
        # simrankを計算する時にweightをかけるかどうか
        self._weighted = weighted
        # 平均ノード次数を計算する時に、logを使うか、使わないか
        self._log_flag = log_flag

        # bag_of_words
        self._list_bag = list_sep
        self._list_bag_all = list_sep_all
        # inputfileからnode, edge, weightを計算
        if list_edgelist == None:
            self._list_node, self._list_edge, self._list_weight = self._cal_node_edge_weight(list_sep)
        else:
            list_node = list(set([word for row in self._list_bag for word in row]))
            self._list_node = list_node
            list_edge = [tuple(row) for row in list_edgelist]
            tuple_edge, tuple_weight = zip(*collections.Counter(list_edge).items())
            self._list_edge = list(tuple_edge)
            self._list_weight = list(tuple_weight)

        # 単語のword_idのdictを作成する
        self._dict_word_id = {word: i for i, word in enumerate(self._list_node)}

        # 距離行列の作成
        if self._simrank_flag == True:
            if self._weighted == True:
                self._matrix = self._cal_simrank(list_edgelist=list_edgelist,
                                                 C=0.8, iteration=10)
            else:
                self._matrix = self._cal_simrank(list_edgelist=self._list_edge,
                                                 C=0.8, iteration=10)

        else:
            self._matrix = self._cal_matrix_path_out(fill='max', log_flag=log_flag)

        # 全単語のリスト
        self._list_all_word = [word for row in self._list_bag for word in row]
        self._d_matrix = np.array([self._matrix[self._dict_word_id[word]] for word in self._list_all_word])
        # 抽出した文章の集合
        self._list_C = []
        
    @property
    def dict_word_id(self):
        print 'property: dict_word_id'
        return self._dict_word_id

    @dict_word_id.setter
    def dict_word_id(self, value):
        print 'setter: dict_word_id'
        self._dict_word_id = value

    @dict_word_id.deleter
    def dict_word_id(self):
        print 'deleter: dict_word_id'
        del self._dict_word_id

    @property
    def matrix(self):
        print 'property: matrix'
        return self._matrix

    @matrix.setter
    def matrix(self, value):
        print 'setter: matrix'
        self._matrix = value

    @matrix.deleter
    def matrix(self):
        print 'deleter: matrix'
        del self._matrix

    @property
    def list_all_word(self):
        print 'property: list_all_word'
        return self._list_all_word

    @list_all_word.setter
    def list_all_word(self, value):
        print 'setter: list_all_word'
        self._list_all_word = value

    @list_all_word.deleter
    def list_all_word(self):
        print 'deleter: list_all_word'
        del self._list_all_word
        self._list_all_word = []

    @property
    def list_C(self):
        print 'property: list_C'
        return self._list_C

    @list_C.setter
    def list_C(self, value):
        print 'setter: list_C'
        self._list_C = value

    @list_C.deleter
    def list_C(self):
        print 'deleter: list_C'
        del self._list_C
        self._list_C = []

    # 入力されたエッジリストから、list_node, list_edge, list_weightを計算する
    def _cal_node_edge_weight(self, list_bag):
        """
        list_bag: bag_of_words
        list_node: nodeのリスト
        list_edge: edgeのリスト
        list_weight: weightのリスト
        """
        list_edgelist = self._cal_bag_edgelist(list_bag)
        # 有向エッジリストを無向エッジリストに変換する
        list_edge = [tuple(sorted(row)) for row in list_edgelist]
        # ノードリスト
        list_node = list(set([word for row in list_bag for word in row]))
        # エッジリストとそのweightを作成
        tuple_edge, tuple_weight = zip(*collections.Counter(list_edge).items())

        return list_node, list(tuple_edge), list(tuple_weight)

    def _cal_bag_edgelist(self, list_bag):
        """
        bag_of_wordsをedgelistに変換する
        list_bag: bag_of_words
        return: list_edgelist
        """
        list_edgelist = []
        for row in list_bag:
            list_edgelist.extend(list(itertools.combinations(tuple(row),2)))
        return list_edgelist

    # ノード間の距離を計算する
    def _cal_matrix_path_out(self, fill='max', log_flag=True):
        """
        list_node: nodeのリスト
        list_edge: edgeのリスト
        list_weight: 重みのリスト
        weight: flag=Falseの場合、重みをいくつ履かせるか
        inverse_flag: ノード間の距離を計算するときに、inverseにするか、マイナスにするかのflag
        fill: パスが存在しない場合に、なんの値を入れるか。
        matrix: 計算後のノード間の距離が記録されたmatrix, 平均距離
        """

        g = Graph(directed=self._directed)
        g.add_vertices(self._list_node)
        g.add_edges(self._list_edge)
        # logの平均距離にするか、普通の平均距離にするか
        if log_flag == True:
            list_out_weight = np.log(np.array(self._list_weight))
        else:
            list_out_weight = np.array(self._list_weight)

        # sparse_matrixの定義
        matrix = lil_matrix((len(self._list_node), len(self._list_node)))
        # ノード間の距離を計算する
        for i, node in enumerate(self._list_node):
            list_path = g.get_shortest_paths(node, self._list_node,
                                             mode=OUT, weights=self._list_weight,
                                             output='epath')
            for j, row in enumerate(list_path):
                if len(row) > 0:
                    matrix[i, j] = np.average(list_out_weight[row])

        # matrixをarray型に変換
        matrix = matrix.toarray()
        # maxの値を取得
        max_value = np.amax(matrix)
        # 0の要素にmax_valueを代入する
        matrix[matrix==0] = max_value
        # 対角成分は0にする（自分から自分への距離は0)
        for i, row in enumerate(matrix):
            matrix[i][i] = 0
        return matrix

    def _cal_simrank(self, list_edgelist, C=0.8, iteration=10):
        g = Graph(directed=self._directed)
        g.add_vertices(self._list_node)
        g.add_edges(list_edgelist)
        G = np.array(g.get_adjacency()._data)
        G = G / np.sum(G, axis=1, dtype=float)[:,np.newaxis]
        G[np.isinf(G)] = 0.0
        G[np.isnan(G)] = 0.0
        G = G.T
        S = np.identity(len(self._list_node))
        for iter in range(iteration):
            S = C * np.dot(np.dot(G.T, S), G)
            for i in range(len(self._list_node)):
                S[i][i] = 1.0

        return S


    # コストの計算
    def _cal_cost(self, list_c_word, scale):
        """
        コストの計算
        :param list_c_word: 現在採用している文の中に含まれる単語
        :param distance_matrix: W * Vの距離行列
        :param scale: スケール関数に何を使うか, 0: e^x, 1: x, 2: ln_x
        :return: f_C (計算したコスト)
        """
        # 単語が入ってなければ0を返す
        if len(list_c_word) == 0:
            return 0.0
        # 単語をidに変換
        list_c_id = sorted([self._dict_word_id[word] for word in list_c_word])
        f_C = 0.0
        # すべての単語を検索
        matrix_tmp = self._d_matrix[:,list_c_id]
        # スケーリング関数: e^x
        if scale == 0:
            f_C = np.sum(np.exp(np.amax(matrix_tmp, axis=1)))
        # スケーリング関数: x
        elif scale == 1:
            f_C = np.sum(np.amax(matrix_tmp, axis=1))
        # スケーリング関数: ln_x
        else:
            f_C = np.sum(np.log(np.amax(matrix_tmp, axis=1)))

        return f_C

    def _m_greedy_1(self, list_C, list_id_sep_sepall, r=1, scale=0):
        """
        修正貪欲法の一周分
        :param list_C: 現在採用している文のbag_of_words
        :param list_id_document: それ以外の採用候補
        :param distance_matrix: 距離行列, W * V
        :param r: 文字数に対するコストをどれだけかけるか
        :param scale: スケーリング関数、0: e^x, 1: x, 2: ln_x
        :return: doc: idとその単語のリスト
        """
        # 現在のlist_Cに含まれるユニークな単語のリストを作成
        list_c_word = sorted(list(set([word for row in list_C for word in row[1]])))
        # f_C: 現在のコストの計算
        f_C = self._cal_cost(list_c_word=list_c_word,
                             scale=scale)
        
        # 文書を一つずつ追加した時のスコアの増分を計算する
        list_return = list_id_sep_sepall[0]
        delta_max = 0.0
        for doc_id, sep, sepall in list_id_sep_sepall:
            # 文章の追加
            list_c_word_s = list(set(list_c_word + sep))
            # コストの計算
            f_C_s = self._cal_cost(list_c_word=list_c_word_s,
                                   scale=scale)
            # スコアの増分を計算
            delta = (f_C_s - f_C) / np.power(len(sepall), r)
            if delta > delta_max:
                delta_max = delta
                list_return = [doc_id, sep, sepall]
        return list_return


    def m_greedy(self, num_w = 100, r=1, scale=0):
        """
        修正貪欲法による文章の抽出
        :param num_w: 単語数の制約
        :param r: 単語数に対するコストのパラメータ
        :param scale: スケーリング関数, 0: e^x, 1: x, 2: ln_x
        :return: 抽出した文章のidとそのbag_of_wordsのリスト
        """
        # list_id_documentの作成
        list_id_sep_sepall = [[i, row[0], row[1]] for i, row in enumerate(zip(self._list_bag, self._list_bag_all)) if len(row[0]) > 0]
        list_id_sep_sepall_copy = copy.deepcopy(list_id_sep_sepall)
        # 要約文書のリスト
        list_C = []
        # num_sで指定した文章数を抜き出すまで繰り返す
        C_word = 0
        while len(list_id_sep_sepall):
            # コストが一番高くなる組み合わせを計算
            doc_id, sep, sepall = self._m_greedy_1(list_C=list_C,
                                                   list_id_sep_sepall=list_id_sep_sepall,
                                                   r=r, scale=scale)
            if C_word + len(sepall) <= num_w:
                # 採用したリストをappend
                list_C.append([doc_id, sep, sepall])
                C_word += len(sepall)
            # 元の集合からremove
            list_id_sep_sepall.remove([doc_id, sep, sepall])

        f_max = 0.0
        list_return = list_id_sep_sepall_copy[0]
        for doc_id, sep, sepall in list_id_sep_sepall_copy:
            # documentに含まれる単語のリスト
            list_c_word = sorted(list(set([word for word in sep])))
            # スコアの計算
            f_C = self._cal_cost(list_c_word=list_c_word,
                                 scale=scale)
            if f_C > f_max:
                f_max = f_C
                list_return = [doc_id, sep, sepall]

        list_c_word = sorted(list(set([word for row in list_C for word in row[1]])))
        f_C = self._cal_cost(list_c_word=list_c_word, scale=scale)

        if f_C >= f_max:
            self._list_C = list_C
        else:
            self._list_C = [list_return]

        print '計算が終了しました'


# 品詞ごとにバイアスをかける
class Partition_GraphSubModular(Modified_GraphSubModular):

    def __init__(self, list_sep_all, list_sep, list_pos, list_edgelist=None, directed=True, simrank_flag=True, weighted=False, log_flag=True):
        super(Partition_GraphSubModular, self).__init__(list_sep_all, list_sep, list_edgelist, directed, simrank_flag, weighted, log_flag)
        self._list_pos = list_pos
        # 品詞ごとの距離行列を作成する
        self._d_matrix_n, self._d_matrix_j, self._d_matrix_v, self._d_matrix_o = self._cal_partition_matrix()


    # 品詞ごとの距離行列を作成する
    def _cal_partition_matrix(self):
        dict_pos_word = {'n': [],
                         'j': [],
                         'v': [],
                         '-': []}

        for bags, poses in zip(self._list_bag, self._list_pos):
            for word, pos in zip(bags, poses):
                dict_pos_word[pos].append(self._dict_word_id[word])

        unique_n = list(set(dict_pos_word['n']))
        unique_j = list(set(dict_pos_word['j']))
        unique_v = list(set(dict_pos_word['v']))
        unique_o = list(set(dict_pos_word['-']))

        # 名詞のmatrix
        d_matrix_n = np.array([self._matrix[word_id] for word_id in dict_pos_word['n']])

        # 形容詞のmatrix
        d_matrix_j = np.array([self._matrix[word_id] for word_id in dict_pos_word['j']])

        # 動詞のmatrix
        d_matrix_v = np.array([self._matrix[word_id] for word_id in dict_pos_word['v']])

        # それ以外のmatrix
        d_matrix_o = np.array([self._matrix[word_id] for word_id in dict_pos_word['-']])

        return d_matrix_n, d_matrix_j, d_matrix_v, d_matrix_o

    # コストの計算
    def _cal_cost(self, list_c_word, list_c_pos, scale):
        """
        コストの計算
        :param list_c_word: 現在採用している文の中に含まれる単語
        :param distance_matrix: W * Vの距離行列
        :param scale: スケール関数に何を使うか, 0: e^x, 1: x, 2: ln_x
        :return: f_C (計算したコスト)
        """
        # 単語が入ってなければ0を返す
        if len(list_c_word) == 0:
            return 0.0
        else:
            # 品詞ごとに単語をidに変換
            list_n_id = sorted([self._dict_word_id[word]
                                for word, pos in zip(list_c_word, list_c_pos)
                                if pos == 'n'])
            list_n_id = list(set(list_n_id))
            list_j_id = sorted([self._dict_word_id[word]
                                for word, pos in zip(list_c_word, list_c_pos)
                                if pos == 'j'])
            list_j_id = list(set(list_j_id))
            list_v_id = sorted([self._dict_word_id[word]
                                for word, pos in zip(list_c_word, list_c_pos)
                                if pos == 'v'])
            list_v_id = list(set(list_v_id))
            list_o_id = sorted([self._dict_word_id[word]
                                for word, pos in zip(list_c_word, list_c_pos)
                                if pos == '-'])
            list_o_id = list(set(list_o_id))
            # すべての単語を検索
            matrix_n = self._d_matrix_n[:,list_n_id]
            matrix_j = self._d_matrix_n[:,list_j_id]
            matrix_v = self._d_matrix_n[:,list_v_id]
            matrix_o = self._d_matrix_n[:,list_o_id]
            list_matrix = [matrix_n, matrix_j, matrix_v, matrix_o]
            list_bias = [self._n, self._j, self._v, self._o]
            
            # スケーリング関数: e^x
            if scale == 0:
                f_C = 0.0
                for matrix, bias in zip(list_matrix, list_bias):
                    if matrix.shape[1] != 0:
                        f_C += bias * np.sum(np.exp(np.amax(matrix, axis=1)))

            # スケーリング関数: x
            elif scale == 1:
                f_C = 0.0
                for matrix, bias in zip(list_matrix, list_bias):
                    if matrix.shape[1] != 0:
                        f_C += bias * np.sum(np.amax(matrix, axis=1))

            # スケーリング関数: ln_x
            else:
                f_C = 0.0
                for matrix, bias in zip(list_matrix, list_bias):
                    if matrix.shape[1] != 0:
                        f_C += bias * np.sum(np.log(np.amax(matrix, axis=1)))

        return f_C

    def _m_greedy_1(self, list_C, list_id_sep_pos_sepall, r=1, scale=0):
        """
        修正貪欲法の一周分
        :param list_C: 現在採用している文のbag_of_words
        :param list_id_document: それ以外の採用候補
        :param distance_matrix: 距離行列, W * V
        :param r: 文字数に対するコストをどれだけかけるか
        :param scale: スケーリング関数、0: e^x, 1: x, 2: ln_x
        :return: doc: idとその単語のリスト
        """
        # 現在のlist_Cに含まれるユニークな単語のリストを作成
        list_c_word_pos = sorted(list(set([word_pos for row in list_C for word_pos in row[1]])))
        if len(list_c_word_pos) == 0:
            list_c_word = []
            list_c_pos = []
        else:
            list_c_word, list_c_pos = zip(*list_c_word_pos)
            list_c_word = list(list_c_word)
            list_c_pos = list(list_c_pos)
        # f_C: 現在のコストの計算
        f_C = self._cal_cost(list_c_word=list_c_word,
                             list_c_pos=list_c_pos,
                             scale=scale)

        # 文書を一つずつ追加した時のスコアの増分を計算する
        list_return = list_id_sep_pos_sepall[0]
        delta_max = 0.0
        for doc_id, sep_pos, sepall in list_id_sep_pos_sepall:
            # 文章の追加
            list_c_word_pos_s = list(set(list_c_word_pos + sep_pos))
            list_c_word_s, list_c_pos_s = zip(*list_c_word_pos_s)

            list_c_word_s = list(list_c_word_s)
            list_c_pos_s = list(list_c_pos_s)

            # コストの計算
            f_C_s = self._cal_cost(list_c_word=list_c_word_s,
                                   list_c_pos=list_c_pos_s,
                                   scale=scale)
            # スコアの増分を計算
            delta = (f_C_s - f_C) / np.power(len(sepall), r)
            if delta > delta_max:
                delta_max = delta
                list_return = [doc_id, sep_pos, sepall]
        return list_return


    def m_greedy(self, num_w = 100, r=1, scale=0, n=0.2, j=0.2, v=0.2, o=0.2):
        """
        修正貪欲法による文章の抽出
        :param num_w: 単語数の制約
        :param r: 単語数に対するコストのパラメータ
        :param scale: スケーリング関数, 0: e^x, 1: x, 2: ln_x
        :return: 抽出した文章のidとそのbag_of_wordsのリスト
        """
        # 係数をセット
        self._n = n
        self._j = j
        self._v = v
        self._o = o
        # list_id_documentの作成
        list_id_sep_pos_sepall = [[i, [(word, pos) for word, pos in zip(row[0], row[1])], row[2]]
                                  for i, row in enumerate(zip(self._list_bag,
                                                              self._list_pos,
                                                              self._list_bag_all))
                                  if len(row[0]) > 0]
        list_id_sep_pos_sepall_copy = copy.deepcopy(list_id_sep_pos_sepall)
        # 要約文書のリスト
        list_C = []
        # num_sで指定した文章数を抜き出すまで繰り返す
        C_word = 0
        while len(list_id_sep_pos_sepall):
            # コストが一番高くなる組み合わせを計算
            doc_id, sep_pos, sepall = self._m_greedy_1(list_C=list_C,
                                                       list_id_sep_pos_sepall=list_id_sep_pos_sepall,
                                                       r=r, scale=scale)
            if C_word + len(sepall) <= num_w:
                # 採用したリストをappend
                list_C.append([doc_id, sep_pos, sepall])
                C_word += len(sepall)
            # 元の集合からremove
            list_id_sep_pos_sepall.remove([doc_id, sep_pos, sepall])

        f_max = 0.0
        list_return = list_id_sep_pos_sepall_copy[0]
        for doc_id, sep_pos, sepall in list_id_sep_pos_sepall_copy:
            # documentに含まれる単語のリスト
            list_sep_pos = sorted(list(set([word for word in sep_pos])))
            list_c_word, list_c_pos = zip(*list_sep_pos)
            list_c_word = list(list_c_word)
            list_c_pos = list(list_c_pos)
            # スコアの計算
            f_C = self._cal_cost(list_c_word=list_c_word,
                                 list_c_pos=list_c_pos,
                                 scale=scale)
            if f_C > f_max:
                f_max = f_C
                list_return = [doc_id, sep_pos, sepall]

        list_sep_pos = sorted(list(set([sep_pos for row in list_C for sep_pos in row[1]])))
        list_c_word, list_c_pos = zip(*list_sep_pos)
        list_c_word = list(list_c_word)
        list_c_pos = list(list_c_pos)
        f_C = self._cal_cost(list_c_word=list_c_word,
                             list_c_pos=list_c_pos,
                             scale=scale)

        if f_C >= f_max:
            self._list_C = list_C
        else:
            self._list_C = [list_return]

        print '計算が終了しました'


class Multiplex_GraphSubModular(object):
    """
    BigramとCorの両方を用いて最適化
    """

    def __init__(self, list_sep_all, list_sep, weighted=False):
        """

        :param list_sep_all:
        :param list_sep:
        :param list_edgelist:
        :param directed:
        :param simrank_flag:
        :param log_flag:
        """
        # simrankを計算する時にweightをかけるかどうか
        self._weighted = weighted

        # bag_of_words
        self._list_bag = list_sep
        self._list_bag_all = list_sep_all
        # inputfileからnode, edge, weightを計算
        self._list_node, self._list_edge_c, self._list_weight_c, self._list_edge_all_c = self._cal_node_edge_weight(list_sep,
                                                                                                                    flag='cor')
        _, self._list_edge_b, self._list_weight_b, self._list_edge_all_b = self._cal_node_edge_weight(list_sep,
                                                                                                      flag='bigram')

        # 単語のword_idのdictを作成する
        self._dict_word_id = {word: i for i, word in enumerate(self._list_node)}

        # 距離行列の作成
        if self._weighted == True:
            self._matrix_c = self._cal_simrank(list_edgelist=self._list_edge_all_c,
                                               C=0.8, iteration=20)
            self._matrix_b = self._cal_simrank(list_edgelist=self._list_edge_all_b,
                                               C=0.8, iteration=20)
        else:
            self._matrix_c = self._cal_simrank(list_edgelist=self._list_edge_c,
                                               C=0.8, iteration=20)
            self._matrix_b = self._cal_simrank(list_edgelist=self._list_edge_b,
                                               C=0.8, iteration=20)

        # 全単語のリスト
        self._list_all_word = [word for row in self._list_bag for word in row]
        self._d_matrix_c = np.array([self._matrix_c[self._dict_word_id[word]] for word in self._list_all_word])
        self._d_matrix_b = np.array([self._matrix_b[self._dict_word_id[word]] for word in self._list_all_word])
        # 抽出した文章の集合
        self._list_C = []

    @property
    def dict_word_id(self):
        print 'property: dict_word_id'
        return self._dict_word_id

    @dict_word_id.setter
    def dict_word_id(self, value):
        print 'setter: dict_word_id'
        self._dict_word_id = value

    @dict_word_id.deleter
    def dict_word_id(self):
        print 'deleter: dict_word_id'
        del self._dict_word_id

    @property
    def matrix_c(self):
        print 'property: matrix_c'
        return self._matrix_c

    @matrix_c.setter
    def matrix_c(self, value):
        print 'setter: matrix_c'
        self._matrix_c = value

    @matrix_c.deleter
    def matrix_c(self):
        print 'deleter: matrix_c'
        del self._matrix_c

    @property
    def matrix_b(self):
        print 'property: matrix_b'
        return self._matrix_b

    @matrix_b.setter
    def matrix_b(self, value):
        print 'setter: matrix_b'
        self._matrix_b = value

    @matrix_b.deleter
    def matrix_b(self):
        print 'deleter: matrix_b'
        del self._matrix_b

    @property
    def list_all_word(self):
        print 'property: list_all_word'
        return self._list_all_word

    @list_all_word.setter
    def list_all_word(self, value):
        print 'setter: list_all_word'
        self._list_all_word = value

    @list_all_word.deleter
    def list_all_word(self):
        print 'deleter: list_all_word'
        del self._list_all_word
        self._list_all_word = []

    @property
    def list_C(self):
        print 'property: list_C'
        return self._list_C

    @list_C.setter
    def list_C(self, value):
        print 'setter: list_C'
        self._list_C = value

    @list_C.deleter
    def list_C(self):
        print 'deleter: list_C'
        del self._list_C
        self._list_C = []

    # 入力されたエッジリストから、list_node, list_edge, list_weightを計算する
    def _cal_node_edge_weight(self, list_bag, flag='cor'):
        """
        list_bag: bag_of_words
        list_node: nodeのリスト
        list_edge: edgeのリスト
        list_weight: weightのリスト
        """
        if flag == 'cor':
            list_edgelist = self._cal_bag_edgelist_c(list_bag)
        else:
            list_edgelist = self._cal_bag_edgelist_b(list_bag)
        # 有向エッジリストを無向エッジリストに変換する
        list_edge = [tuple(sorted(row)) for row in list_edgelist]
        # ノードリスト
        list_node = list(set([word for row in list_bag for word in row]))
        # エッジリストとそのweightを作成
        tuple_edge, tuple_weight = zip(*collections.Counter(list_edge).items())

        return list_node, list(tuple_edge), list(tuple_weight), list_edge

    def _cal_bag_edgelist_c(self, list_bag):
        """
        bag_of_wordsをedgelistに変換する
        list_bag: bag_of_words
        return: list_edgelist
        """
        list_edgelist = []
        for row in list_bag:
            list_edgelist.extend(list(itertools.combinations(tuple(row),2)))
        return list_edgelist

    def _cal_bag_edgelist_b(self, list_bag):
        """
        bag_of_wordsをedgelistに変換する
        list_bag: bag_of_words
        return: list_edgelist
        """
        list_edgelist = []
        for row in list_bag:
            list_edgelist.extend([[row[i], row[i+1]] for i in range(len(row)-1)])
        return list_edgelist

    def _cal_simrank(self, list_edgelist, C=0.8, iteration=20):
        g = Graph(directed=False)
        g.add_vertices(self._list_node)
        g.add_edges(list_edgelist)
        G = np.array(g.get_adjacency()._data)
        G = G / np.sum(G, axis=1, dtype=float)[:,np.newaxis]
        G[np.isinf(G)] = 0.0
        G[np.isnan(G)] = 0.0
        G = G.T
        S = np.identity(len(self._list_node))
        for iter in range(iteration):
            S = C * np.dot(np.dot(G.T, S), G)
            for i in range(len(self._list_node)):
                S[i][i] = 1.0
        return S

    # コストの計算
    def _cal_cost(self, list_c_word, d_matrix, scale):
        """
        コストの計算
        :param list_c_word: 現在採用している文の中に含まれる単語
        :param distance_matrix: W * Vの距離行列
        :param scale: スケール関数に何を使うか, 0: e^x, 1: x, 2: ln_x
        :return: f_C (計算したコスト)
        """
        # 単語が入ってなければ0を返す
        if len(list_c_word) == 0:
            return 0.0
        # 単語をidに変換
        list_c_id = sorted([self._dict_word_id[word] for word in list_c_word])
        f_C = 0.0
        # すべての単語を検索
        matrix_tmp = d_matrix[:,list_c_id]
        # スケーリング関数: e^x
        if scale == 0:
            f_C = np.sum(np.exp(np.amax(matrix_tmp, axis=1)))
        # スケーリング関数: x
        elif scale == 1:
            f_C = np.sum(np.amax(matrix_tmp, axis=1))

        return f_C

    def _m_greedy_1(self, list_C, list_id_sep_sepall, r=1, scale=0):
        """
        修正貪欲法の一周分
        :param list_C: 現在採用している文のbag_of_words
        :param list_id_document: それ以外の採用候補
        :param distance_matrix: 距離行列, W * V
        :param r: 文字数に対するコストをどれだけかけるか
        :param scale: スケーリング関数、0: e^x, 1: x, 2: ln_x
        :return: doc: idとその単語のリスト
        """
        # 現在のlist_Cに含まれるユニークな単語のリストを作成
        list_c_word = sorted(list(set([word for row in list_C for word in row[1]])))
        # f_C: 現在のコストの計算
        f_C_c = self._cal_cost(list_c_word=list_c_word,
                               d_matrix=self._d_matrix_c,
                               scale=scale)
        f_C_b = self._cal_cost(list_c_word=list_c_word,
                               d_matrix=self._d_matrix_b,
                               scale=scale)

        f_C = self._a*f_C_c + self._b*f_C_b

        # 文書を一つずつ追加した時のスコアの増分を計算する
        list_return = list_id_sep_sepall[0]
        delta_max = 0.0
        for doc_id, sep, sepall in list_id_sep_sepall:
            # 文章の追加
            list_c_word_s = list(set(list_c_word + sep))
            # コストの計算
            f_C_c_s = self._cal_cost(list_c_word=list_c_word_s,
                                     d_matrix=self._d_matrix_c,
                                     scale=scale)
            f_C_b_s = self._cal_cost(list_c_word=list_c_word_s,
                                     d_matrix=self._d_matrix_b,
                                     scale=scale)

            f_C_s = self._a*f_C_c_s + self._b*f_C_b_s

            # スコアの増分を計算
            delta = (f_C_s - f_C) / np.power(len(sepall), r)
            if delta > delta_max:
                delta_max = delta
                list_return = [doc_id, sep, sepall]
        return list_return


    def m_greedy(self, num_w = 100, r=1, scale=0, a=0.5, b=0.5):
        """
        修正貪欲法による文章の抽出
        :param num_w: 単語数の制約
        :param r: 単語数に対するコストのパラメータ
        :param scale: スケーリング関数, 0: e^x, 1: x
        :param a:
        :param b:
        :return: 抽出した文章のidとそのbag_of_wordsのリスト
        """
        # パラメータのセット
        self._a = a
        self._b = b
        # list_id_documentの作成
        list_id_sep_sepall = [[i, row[0], row[1]] for i, row in enumerate(zip(self._list_bag, self._list_bag_all)) if len(row[0]) > 0]
        list_id_sep_sepall_copy = copy.deepcopy(list_id_sep_sepall)
        # 要約文書のリスト
        list_C = []
        # num_sで指定した文章数を抜き出すまで繰り返す
        C_word = 0
        while len(list_id_sep_sepall):
            # コストが一番高くなる組み合わせを計算
            doc_id, sep, sepall = self._m_greedy_1(list_C=list_C,
                                                   list_id_sep_sepall=list_id_sep_sepall,
                                                   r=r,
                                                   scale=scale)
            if C_word + len(sepall) <= num_w:
                # 採用したリストをappend
                list_C.append([doc_id, sep, sepall])
                C_word += len(sepall)
            # 元の集合からremove
            list_id_sep_sepall.remove([doc_id, sep, sepall])

        # 一つだけ追加した時のコストの計算
        f_max = 0.0
        list_return = list_id_sep_sepall_copy[0]
        for doc_id, sep, sepall in list_id_sep_sepall_copy:
            # documentに含まれる単語のリスト
            list_c_word = sorted(list(set([word for word in sep])))
            # スコアの計算
            f_C_c = self._cal_cost(list_c_word=list_c_word,
                                   d_matrix=self._d_matrix_c,
                                   scale=scale)
            f_C_b = self._cal_cost(list_c_word=list_c_word,
                                   d_matrix=self._d_matrix_b,
                                   scale=scale)
            f_C = self._a*f_C_c + self._b*f_C_b

            if f_C > f_max:
                f_max = f_C
                list_return = [doc_id, sep, sepall]

        # 現在のコスト計算
        list_c_word = sorted(list(set([word for row in list_C for word in row[1]])))
        f_C_c = self._cal_cost(list_c_word=list_c_word,
                               d_matrix=self._d_matrix_c,
                               scale=scale)
        f_C_b = self._cal_cost(list_c_word=list_c_word,
                               d_matrix=self._d_matrix_b,
                               scale=scale)

        f_C = self._a*f_C_c + self._b*f_C_b

        if f_C >= f_max:
            self._list_C = list_C
        else:
            self._list_C = [list_return]

        print '計算が終了しました'
        

class Polarity_GraphSubModular(object):
    """
    極性辞書を用いて、ポジティブネットワーク、ネガティブネットワークを作成する
    """

    def __init__(self, list_sep, list_sep_lemmas, dict_path, weighted=False, threshold=0.5, identity=0.5, mode='bigram'):
        """

        :param list_sep:
        :param list_sep_lemmas:
        :param dict_path:
        :param weighted:
        :param threshold:
        :param identity:
        """
        # simrankを計算する時にweightをかけるかどうか
        self._weighted = weighted
        # 極性辞書のパス
        self._dict_path = dict_path
        self._dict_word_polarity = Filer.readdump(self._dict_path)
        # simrankのidentityの調整
        self._identity = identity
        # corを使うかbigramを使うか
        self._mode = mode

        # bag_of_words
        self._list_bag = list_sep
        self._list_bag_lemmas = [[(word, lemmas) for word, lemmas in zip(sep, sep_lemmas)] for sep, sep_lemmas in zip(list_sep, list_sep_lemmas)]
        # inputfileからnode, edge, weightを計算
        self._list_node, self._list_edge_p, self._list_weight_p, self._list_edge_all_p = self._cal_node_edge_weight(self._list_bag_lemmas, flag='pos', threshold=threshold)
        _, self._list_edge_n, self._list_weight_n, self._list_edge_all_n = self._cal_node_edge_weight(self._list_bag_lemmas, flag='neg', threshold=-1*threshold)

        # 単語のword_idのdictを作成する
        self._dict_word_id = {word: i for i, word in enumerate(self._list_node)}

        # 距離行列の作成
        if self._weighted == True:
            self._matrix_p = self._cal_simrank(list_edgelist=self._list_edge_all_p,
                                               C=0.8, iteration=20)
            self._matrix_n = self._cal_simrank(list_edgelist=self._list_edge_all_n,
                                               C=0.8, iteration=20)
        else:
            self._matrix_p = self._cal_simrank(list_edgelist=self._list_edge_p,
                                               C=0.8, iteration=20)
            self._matrix_n = self._cal_simrank(list_edgelist=self._list_edge_n,
                                               C=0.8, iteration=20)

        # 全単語のリスト
        self._list_all_word = [word for row in self._list_bag for word in row]
        self._d_matrix_p = np.array([self._matrix_p[self._dict_word_id[word]] for word in self._list_all_word])
        self._d_matrix_n = np.array([self._matrix_n[self._dict_word_id[word]] for word in self._list_all_word])
        # 抽出した文章の集合
        self._list_C = []

    @property
    def dict_word_id(self):
        print 'property: dict_word_id'
        return self._dict_word_id

    @dict_word_id.setter
    def dict_word_id(self, value):
        print 'setter: dict_word_id'
        self._dict_word_id = value

    @dict_word_id.deleter
    def dict_word_id(self):
        print 'deleter: dict_word_id'
        del self._dict_word_id

    @property
    def matrix_p(self):
        print 'property: matrix_p'
        return self._matrix_p

    @matrix_p.setter
    def matrix_p(self, value):
        print 'setter: matrix_p'
        self._matrix_p = value

    @matrix_p.deleter
    def matrix_p(self):
        print 'deleter: matrix_p'
        del self._matrix_p

    @property
    def matrix_n(self):
        print 'property: matrix_n'
        return self._matrix_n

    @matrix_n.setter
    def matrix_n(self, value):
        print 'setter: matrix_n'
        self._matrix_n = value

    @matrix_n.deleter
    def matrix_n(self):
        print 'deleter: matrix_n'
        del self._matrix_n

    @property
    def list_all_word(self):
        print 'property: list_all_word'
        return self._list_all_word

    @list_all_word.setter
    def list_all_word(self, value):
        print 'setter: list_all_word'
        self._list_all_word = value

    @list_all_word.deleter
    def list_all_word(self):
        print 'deleter: list_all_word'
        del self._list_all_word
        self._list_all_word = []

    @property
    def list_C(self):
        print 'property: list_C'
        return self._list_C

    @list_C.setter
    def list_C(self, value):
        print 'setter: list_C'
        self._list_C = value

    @list_C.deleter
    def list_C(self):
        print 'deleter: list_C'
        del self._list_C
        self._list_C = []

    # 入力されたエッジリストから、list_node, list_edge, list_weightを計算する
    def _cal_node_edge_weight(self, list_bag, flag='pos', threshold=0.5):
        """
        list_bag: bag_of_words
        flag: 'pos' or 'neg'
        """
        if self._mode == 'cor':
            if flag == 'pos':
                list_edgelist = self._cal_bag_edgelist_p(list_bag, threshold=threshold)
            else:
                list_edgelist = self._cal_bag_edgelist_n(list_bag, threshold=threshold)
        elif self._mode == 'bigram':
            if flag == 'pos':
                list_edgelist = self._cal_bag_edgelist_p_b(list_bag, threshold=threshold)
            else:
                list_edgelist = self._cal_bag_edgelist_n_b(list_bag, threshold=threshold)
        # 有向エッジリストを無向エッジリストに変換する
        list_edge = [tuple(sorted(row)) for row in list_edgelist]
        # ノードリスト
        list_node = list(set([word[0] for row in list_bag for word in row]))
        # エッジリストとそのweightを作成
        tuple_edge, tuple_weight = zip(*collections.Counter(list_edge).items())

        return list_node, list(tuple_edge), list(tuple_weight), list_edge

    def _cal_bag_edgelist_p(self, list_bag, threshold):
        """
        bag_of_wordsをedgelistに変換する
        list_bag: bag_of_words
        return: list_edgelist
        """
        list_edgelist = []
        for row in list_bag:
            list_edgelist.extend(list(itertools.combinations(tuple(row),2)))
        list_edgelist_p = []
        for row in list_edgelist:
            word1 = row[0][0]
            word1_lemmas = row[0][1]
            word2 = row[1][0]
            word2_lemmas = row[1][1]
            if word1_lemmas in self._dict_word_polarity and word2_lemmas in self._dict_word_polarity:
                if self._dict_word_polarity[word1_lemmas][0] >= threshold and self._dict_word_polarity[word2_lemmas][0] >= threshold:
                    list_edgelist_p.append((word1, word2))
            elif word1_lemmas in self._dict_word_polarity:
                if self._dict_word_polarity[word1_lemmas][0] >= threshold:
                    list_edgelist_p.append((word1, word2))
            elif word2_lemmas in self._dict_word_polarity:
                if self._dict_word_polarity[word2_lemmas][0] >= threshold:
                    list_edgelist_p.append((word1, word2))

        return list_edgelist_p
    
    def _cal_bag_edgelist_p_b(self, list_bag, threshold):
        """
        bag_of_wordsをedgelistに変換する
        list_bag: bag_of_words
        return: list_edgelist
        """
        list_edgelist = []
        for row in list_bag:
            list_edgelist.extend([(row[i], row[i+1]) for i in range(len(row)-1)])

        list_edgelist_p = []
        for row in list_edgelist:
            word1 = row[0][0]
            word1_lemmas = row[0][1]
            word2 = row[1][0]
            word2_lemmas = row[1][1]
            if word2_lemmas in self._dict_word_polarity:
                if self._dict_word_polarity[word2_lemmas][0] >= threshold:
                    list_edgelist_p.append((word1, word2))
            elif word1_lemmas in self._dict_word_polarity:
                if self._dict_word_polarity[word1_lemmas][0] >= threshold:
                    list_edgelist_p.append((word1, word2))

        return list_edgelist_p
    
    def _cal_bag_edgelist_n(self, list_bag, threshold):
        """
        bag_of_wordsをedgelistに変換する
        list_bag: bag_of_words
        return: list_edgelist
        """
        list_edgelist = []
        for row in list_bag:
            list_edgelist.extend(list(itertools.combinations(tuple(row),2)))
        list_edgelist_n = []
        for row in list_edgelist:
            word1 = row[0][0]
            word1_lemmas = row[0][1]
            word2 = row[1][0]
            word2_lemmas = row[1][1]
            if word1_lemmas in self._dict_word_polarity and word2_lemmas in self._dict_word_polarity:
                if self._dict_word_polarity[word1_lemmas][1] <= threshold and self._dict_word_polarity[word2_lemmas][1] <= threshold:
                    list_edgelist_n.append((word1, word2))
            elif word1_lemmas in self._dict_word_polarity:
                if self._dict_word_polarity[word1_lemmas][1] <= threshold:
                    list_edgelist_n.append((word1, word2))
            elif word2_lemmas in self._dict_word_polarity:
                if self._dict_word_polarity[word2_lemmas][1] <= threshold:
                    list_edgelist_n.append((word1, word2))

        return list_edgelist_n
    
    def _cal_bag_edgelist_n_b(self, list_bag, threshold=0.0):
        """
        bag_of_wordsをedgelistに変換する
        list_bag: bag_of_words
        return: list_edgelist
        """
        list_edgelist = []
        for row in list_bag:
            list_edgelist.extend([(row[i], row[i+1]) for i in range(len(row)-1)])
        list_edgelist_n = []
        for row in list_edgelist:
            word1 = row[0][0]
            word1_lemmas = row[0][1]
            word2 = row[1][0]
            word2_lemmas = row[1][1]
            if word2_lemmas in self._dict_word_polarity:
                if self._dict_word_polarity[word2_lemmas][1] <= threshold:
                    list_edgelist_n.append((word1, word2))
            elif word1_lemmas in self._dict_word_polarity:
                if self._dict_word_polarity[word1_lemmas][1] <= threshold:
                    list_edgelist_n.append((word1, word2))

        return list_edgelist_n

    def _cal_simrank(self, list_edgelist, C=0.8, iteration=20):
        g = Graph(directed=False)
        g.add_vertices(self._list_node)
        g.add_edges(list_edgelist)
        G = np.array(g.get_adjacency()._data)
        G = G / np.sum(G, axis=1, dtype=float)[:,np.newaxis]
        G[np.isinf(G)] = 0.0
        G[np.isnan(G)] = 0.0
        G = G.T
        S = np.identity(len(self._list_node))
        for iter in range(iteration):
            S = C * np.dot(np.dot(G.T, S), G)
            for i in range(len(self._list_node)):
                S[i][i] = 1.0
        for i in range(len(self._list_node)):
            S[i][i] = self._identity
        return S

    # コストの計算
    def _cal_cost(self, list_c_word, d_matrix, scale):
        """
        コストの計算
        :param list_c_word: 現在採用している文の中に含まれる単語
        :param distance_matrix: W * Vの距離行列
        :param scale: スケール関数に何を使うか, 0: e^x, 1: x, 2: ln_x
        :return: f_C (計算したコスト)
        """
        # 単語が入ってなければ0を返す
        if len(list_c_word) == 0:
            return 0.0
        # 単語をidに変換
        list_c_id = sorted([self._dict_word_id[word] for word in list_c_word])
        f_C = 0.0
        # すべての単語を検索
        matrix_tmp = d_matrix[:,list_c_id]
        # スケーリング関数: e^x
        if scale == 0:
            f_C = np.sum(np.exp(np.amax(matrix_tmp, axis=1)))
        # スケーリング関数: x
        elif scale == 1:
            f_C = np.sum(np.amax(matrix_tmp, axis=1))

        return f_C

    def _m_greedy_1(self, list_C, list_id_sep, r=1, scale=0):
        """
        修正貪欲法の一周分
        :param list_C: 現在採用している文のbag_of_words
        :param list_id_document: それ以外の採用候補
        :param distance_matrix: 距離行列, W * V
        :param r: 文字数に対するコストをどれだけかけるか
        :param scale: スケーリング関数、0: e^x, 1: x, 2: ln_x
        :return: doc: idとその単語のリスト
        """
        # 現在のlist_Cに含まれるユニークな単語のリストを作成
        list_c_word = sorted(list(set([word for row in list_C for word in row[1]])))
        # f_C: 現在のコストの計算
        f_C_c = self._cal_cost(list_c_word=list_c_word,
                               d_matrix=self._d_matrix_p,
                               scale=scale)
        f_C_b = self._cal_cost(list_c_word=list_c_word,
                               d_matrix=self._d_matrix_n,
                               scale=scale)

        f_C = self._a*f_C_c + self._b*f_C_b

        # 文書を一つずつ追加した時のスコアの増分を計算する
        list_return = list_id_sep[0]
        delta_max = 0.0
        for doc_id, sep in list_id_sep:
            # 文章の追加
            list_c_word_s = list(set(list_c_word + sep))
            # コストの計算
            f_C_c_s = self._cal_cost(list_c_word=list_c_word_s,
                                     d_matrix=self._d_matrix_p,
                                     scale=scale)
            f_C_b_s = self._cal_cost(list_c_word=list_c_word_s,
                                     d_matrix=self._d_matrix_n,
                                     scale=scale)

            f_C_s = self._a*f_C_c_s + self._b*f_C_b_s

            # スコアの増分を計算
            delta = (f_C_s - f_C) / np.power(len(sep), r)
            if delta > delta_max:
                delta_max = delta
                list_return = [doc_id, sep]
        return list_return


    def m_greedy(self, num_w = 100, r=1, scale=0, a=0.5, b=0.5):
        """
        修正貪欲法による文章の抽出
        :param num_w: 単語数の制約
        :param r: 単語数に対するコストのパラメータ
        :param scale: スケーリング関数, 0: e^x, 1: x
        :param a:
        :param b:
        :return: 抽出した文章のidとそのbag_of_wordsのリスト
        """
        # パラメータのセット
        self._a = a
        self._b = b
        # list_id_documentの作成
        list_id_sep = [[i, row] for i, row in enumerate(self._list_bag) if len(row) > 0]
        list_id_sep_copy = copy.deepcopy(list_id_sep)
        # 要約文書のリスト
        list_C = []
        # num_sで指定した文章数を抜き出すまで繰り返す
        C_word = 0
        while len(list_id_sep):
            # コストが一番高くなる組み合わせを計算
            doc_id, sep = self._m_greedy_1(list_C=list_C,
                                           list_id_sep=list_id_sep,
                                           r=r,
                                           scale=scale)
            if C_word + len(sep) <= num_w:
                # 採用したリストをappend
                list_C.append([doc_id, sep])
                C_word += len(sep)
            # 元の集合からremove
            list_id_sep.remove([doc_id, sep])

        # 一つだけ追加した時のコストの計算
        f_max = 0.0
        list_return = list_id_sep_copy[0]
        for doc_id, sep in list_id_sep_copy:
            # documentに含まれる単語のリスト
            list_c_word = sorted(list(set([word for word in sep])))
            # スコアの計算
            f_C_c = self._cal_cost(list_c_word=list_c_word,
                                   d_matrix=self._d_matrix_p,
                                   scale=scale)
            f_C_b = self._cal_cost(list_c_word=list_c_word,
                                   d_matrix=self._d_matrix_n,
                                   scale=scale)
            f_C = self._a*f_C_c + self._b*f_C_b

            if f_C > f_max:
                f_max = f_C
                list_return = [doc_id, sep]

        # 現在のコスト計算
        list_c_word = sorted(list(set([word for row in list_C for word in row[1]])))
        f_C_c = self._cal_cost(list_c_word=list_c_word,
                               d_matrix=self._d_matrix_p,
                               scale=scale)
        f_C_b = self._cal_cost(list_c_word=list_c_word,
                               d_matrix=self._d_matrix_n,
                               scale=scale)

        f_C = self._a*f_C_c + self._b*f_C_b

        if f_C >= f_max:
            self._list_C = list_C
        else:
            self._list_C = [list_return]

        print '計算が終了しました'
