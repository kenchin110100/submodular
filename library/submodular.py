# coding: utf-8
"""
劣モジュラ最適化によって文書要約をするためのクラス
"""
from filer2.filer2 import Filer
from igraph import *
import collections
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix, hstack
from scipy.spatial.distance import cosine, sqeuclidean, euclidean
import itertools
from sklearn.cluster import KMeans
import copy

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



# ベクトル操作をするためのクラス
class Vector(object):

    def __init__(self, list_sep_all, list_sep, dict_path):
        """
        :param list_bag: bag of words が記録されたリスト
        """
        self._list_bag = list_sep
        self._list_bag_all = list_sep_all
        # すべての単語の列
        self._list_all_word = [word for row in self._list_bag for word in row]
        # ユニークな単語の列
        self._list_unique_word = list(set(self._list_all_word))
        # 単語をid化
        self._dict_word_id = {word: i for i, word in enumerate(self._list_unique_word)}
        # 辞書の読み込み
        list_word_vec = Filer.readtsv(dict_path)
        self._dict_word_vec = {row[0]: np.array(row[1:], dtype=float) for row in list_word_vec}
        # 距離行列の作成
        self._matrix = self._cal_matrix()
        # 最終的に出力するリスト
        self._list_C = []
        # 総単語数の表示
        print 'num word: ', len(self._list_all_word)
        # 語彙数の表示
        print 'num vocabulary: ', len(self._list_unique_word)
        # 辞書に登録されていない単語数の表示
        list_nonword = [word for word in self._list_unique_word if word not in self._dict_word_id]
        print 'Non dictionalized word: ', len(list_nonword)

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

    @property
    def dict_word_vec(self):
        print 'property: dict_word_vec'
        return self._dict_word_vec

    @dict_word_vec.setter
    def dict_word_vec(self, value):
        print 'setter: dict_word_vec'
        self._dict_word_vec = value

    @dict_word_vec.deleter
    def dict_word_vec(self):
        print 'deleter: dict_word_vec'
        del self._dict_word_vec
        self._dict_word_vec = {}
        
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

    def _cal_matrix(self):
        """
        距離行列を作成する
        :return: matrix
        """
        matrix = np.zeros((len(self._list_unique_word), len(self._list_unique_word)))
        for i, word1 in enumerate(self._list_unique_word):
            for j, word2 in enumerate(self._list_unique_word):
                if word1 in self._dict_word_vec and word2 in self._dict_word_vec:
                    matrix[i][j] = euclidean(self._dict_word_vec[word1], self._dict_word_vec[word2])
        # 0の所には最長のパスを入れる
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

    def m_greedy(self, num_w=100, r=1, scale=0):
        """
        修正貪欲法による文章の抽出
        :param num_w: 抽出する単語数の上限
        :param r: 単語数に対するコストのパラメータ
        :param scale: スケーリング関数, 0: e^x, 1: x, 2: ln_x
        :return: 抽出した文章のidとそのbag_of_wordsのリスト
        """
        # list_id_documentの作成
        list_id_sep_sepall = [[i, row[0], row[1]] for i, row in enumerate(zip(self._list_bag, self._list_bag_all)) if len(row[0]) > 0]
        list_id_sep_sepall_copy = copy.deepcopy(list_id_sep_sepall)
        # 要約文書のリスト
        list_C = []
        C_word = 0
        # num_sで指定した文章数を抜き出すまで繰り返す
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


class SubModular(object):
    def __init__(self, list_sep_all, list_sep):
        """
        A Class of Submodular Functions for Document Summarization (Liu, et al, 2011)の実装
        :param list_bag: bag of wordsが記録されたリスト
        """
        # インプットデータの読み込み
        self._list_bag_u = list_sep
        self._list_bag_u_all = list_sep_all

        # パラメータの計算
        self.set_params()

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

    @property
    def list_C_id(self):
        print 'property: list_C_id'
        return self._list_C_id

    @list_C_id.setter
    def list_C_id(self, value):
        print 'setter: list_C_id'
        self._list_C_id = value

    @list_C_id.deleter
    def list_C_id(self):
        print 'deleter: list_C_id'
        del self._list_C_id



    def set_params(self):
        """
        計算に必要なパラメーターの計算を行う
        :return:
        """
        # 全単語のリスト(unigram)
        self._list_all_word_u = [word for row in self._list_bag_u for word in row]
        # ユニークな単語のリスト(unigram)
        self._list_unique_word_u = list(set(self._list_all_word_u))
        # 単語のid変換の辞書(unigram)
        self._dict_word_id_u = {word: i for i, word in enumerate(self._list_unique_word_u)}
        self._dict_id_word_u = {i: word for word, i in self._dict_word_id_u.items()}
        # tf-idfのマトリックスの計算（文書、単語のマトリックス）
        self._matrix_u = self._cal_matrix(self._list_bag_u,
                                          self._dict_word_id_u,
                                          self._dict_id_word_u,
                                          len(self._list_bag_u),
                                          len(self._list_unique_word_u))

        # bigramリストの作成
        self._list_bag_b = [[row[i]+'_'+row[i+1] for i in range(len(row)-1)] for row in self._list_bag_u]
        # 全単語のリスト(bigram)
        self._list_all_word_b = [word for row in self._list_bag_b for word in row]
        # ユニークな単語のリスト(bigram)
        self._list_unique_word_b = list(set(self._list_all_word_b))
        # 単語のid変換の辞書(bigram)
        self._dict_word_id_b = {word: i for i, word in enumerate(self._list_unique_word_b)}
        self._dict_id_word_b = {i: word for word, i in self._dict_word_id_b.items()}
        # tf-idfのマトリックスの計算（文書、単語のマトリックス, bigram）
        self._matrix_b = self._cal_matrix(self._list_bag_b,
                                          self._dict_word_id_b,
                                          self._dict_id_word_b,
                                          len(self._list_bag_b),
                                          len(self._list_unique_word_b))

        # unigramとbigramのmatrixを結合
        self._matrix_ub = hstack([self._matrix_u, self._matrix_b])
        # 距離行列の計算
        self._distance_matrix = self._cal_distance_matrix(self._matrix_ub.toarray())
        # 重要度のリストを作成
        self._arr_r = np.sum(self._distance_matrix, axis=1) / len(self._list_bag_u)
        # C(V)をリスト形式で計算
        self._arr_CV = np.sum(self._distance_matrix, axis=1)

        # 文書数の表示
        print 'num_sentence: ', len(self._list_bag_u)
        # 総単語数の表示
        print 'num word: ', len(self._list_all_word_u)
        # 語彙数の表示
        print 'num vocabulary: ', len(self._list_unique_word_u)

    def _cal_matrix(self, list_bag, dict_word_id, dict_id_word, num_row, num_col):
        """
        bag of wordsからtfidfのmatrixを計算する
        :return: matrix(ただし、スパースマトリックス形式, csr_matrix)
        """
        # sparse_matrixのrow, col, dataを計算
        row_col_data = [[i, dict_word_id[word], float(j)/len(row)]
                        for i, row in enumerate(list_bag)
                        for word, j in collections.Counter(row).items()]

        row, col, data = zip(*row_col_data)
        # csr_matrixの初期化（この段階でtfになっている）
        matrix = csr_matrix((list(data), (list(row), list(col))),
                            shape=(num_row, num_col))
        # idfの辞書を作成
        list_idf = matrix.getnnz(axis=0)
        list_idf = [np.log(float(len(list_bag))/num)+1 for num in list_idf]
        dict_word_idf = {dict_id_word[i]: idf for i, idf in enumerate(list_idf)}

        # idfの辞書をもとにもう一度matrixを作成
        row_col_data = [[i, dict_word_id[word], float(j)/len(row)*dict_word_idf[word]]
                        for i, row in enumerate(list_bag)
                        for word, j in collections.Counter(row).items()]
        row, col, data = zip(*row_col_data)
        # csr_matrixの初期化（この段階でtf-idfになっている）
        matrix = csr_matrix((list(data), (list(row), list(col))),
                            shape=(num_row, num_col))

        return matrix

    def _cal_distance_matrix(self, matrix):
        """
        bag of wordsのmatrixからcosineの距離行列の計算
        :param matrix: numpy.array形式の文書×単語数のmatrix
        :return: numpy.array形式の文書×文書のmatrix
        """
        # まず行列を正規化する
        l_matrix = np.sqrt(np.sum(matrix*matrix, axis=1))
        norm_matrix = matrix / l_matrix[:, np.newaxis]
        # 欠損値を埋める
        norm_matrix[np.isnan(norm_matrix)] = 0
        return norm_matrix.dot(norm_matrix.T)

    def _cal_cluster(self, matrix, n_cluster):
        """
        K-meansでクラスタリングをする
        :param matrix: numpy.array型の文書数×単語数の行列
        :param n_cluster: クラスタリング数
        :return: numpy.arrayのクラスタ番号が記録されたarray
        """
        kmeans = KMeans(n_clusters=n_cluster)
        arr_cluster = kmeans.fit_predict(matrix)
        return arr_cluster

    def _cal_L(self, list_id, alpha):
        """
        与えらえれた集合に対するL(S)を計算する
        :param list_id: id(文書の列番号）が記録されたリスト
        :param alpha: 係数
        :param arr_CV: C(V)が記録されたnumpy.array
        :return: L(S)のコスト
        """
        # C(S)を計算する
        arr_Ci = np.sum(self._distance_matrix[:,list_id], axis=1)
        # min(C(S), alpha*C(V))を計算する
        arr_L = np.minimum(arr_Ci, self._arr_CV*alpha)
        return np.sum(arr_L)

    def _cal_R(self, list_id):
        """
        与えられた集合に対するR(S)を計算する
        :param list_id: id(文書の列番号）が記録されたリスト
        :param arr_cluster: 各文書のクラスタidが記録されたarr
        :return: R(S)のコスト
        """
        # 各クラスタごとにrをまとめたリストを作成
        dict_cluster_r = collections.defaultdict(list)
        for cluster, r in zip(self._arr_cluster[list_id], self._arr_r[list_id]):
            dict_cluster_r[cluster].append(r)

        # R(S)の計算
        R = 0
        for _, value in dict_cluster_r.items():
            R += np.sqrt(np.sum(value))

        return R

    def _m_greedy_1(self, list_rest_id, alpha, lamda, r):
        """
        貪欲法で、損失関数を最大化する、文書idを返す
        :param list_rest_id: まだ集合Sに含まれていないid(文書の列番号)が記録されたリスト
        :param alpha: 係数
        :param lamda: 係数
        :return: 損失関数を最大化する文書id
        """
        # 損失関数を最大化させる文書のidとその時のリスト
        max_delta = 0
        max_id = 0
        if len(self._list_C_id) == 0:
            F_old = 0
        else:
            L_old = self._cal_L(self._list_C_id, alpha)
            R_old = self._cal_R(self._list_C_id)
            F_old = L_old + lamda*R_old
        for rest_id in list_rest_id:
            L_tmp = self._cal_L(self._list_C_id+[rest_id], alpha)
            R_tmp = self._cal_R(self._list_C_id+[rest_id])
            F_tmp = L_tmp + lamda*R_tmp
            delta = (F_tmp - F_old) / np.power(len(self._list_bag_u_all[rest_id]), r)
            if delta > max_delta:
                max_delta = delta
                max_id = rest_id

        return max_id

    def m_greedy(self, num_w=100, alpha=0.1, lamda=6, r=1, n_cluster=100):
        # アトリビュートの初期化
        self._list_C_id = []
        self._list_C = []
        self._arr_cluster = self._cal_cluster(self._matrix_ub.toarray(), n_cluster)
        list_rest_id = [i for i, row in enumerate(self._list_bag_u) if len(row) > 0]
        list_rest_id_copy = copy.deepcopy(list_rest_id)
        C_word = 0
        while len(list_rest_id):
            next_id = self._m_greedy_1(list_rest_id, alpha=alpha, lamda=lamda, r=r)
            if C_word + len(self._list_bag_u_all[next_id]) <= num_w:
                self._list_C_id.append(next_id)
                self._list_C.append([next_id,
                                     self._list_bag_u[next_id],
                                     self._list_bag_u_all[next_id]])
                C_word += len(self._list_bag_u_all[next_id])
            list_rest_id.remove(next_id)
        
        # 全体から一つだけ抽出
        max_delta = 0
        max_id = 0
        for rest_id in list_rest_id_copy:
            L_tmp = self._cal_L([rest_id], alpha)
            R_tmp = self._cal_R([rest_id])
            F_tmp = L_tmp + lamda*R_tmp
            if F_tmp > max_delta:
                max_delta = F_tmp
                max_id = rest_id
                
        # Fを比較
        L_c = self._cal_L(self._list_C_id, alpha)
        R_c = self._cal_R(self._list_C_id)
        F_c = L_c + lamda*R_c

        if max_delta > F_c:
            self._list_C_id = [max_id]
            self._list_C = [max_id,
                            self._list_bag_u[max_id],
                            self._list_bag_u_all[max_id]]

        print '計算が終了しました'


class Rouge_N(object):
    """
    Rouge-Nを計算するためのクラス
    インスタンス化しなくても使える
    形態素解析された後のリストとNを引数にする
    """

    @classmethod
    def _get_ngram(cls, list_sep, N):
        """
        形態素解析されてたファイルからN-gramの辞書を作成する
        :param list_sep: 形態素解析された単語が記録されている２次元の配列
        :param N: 次数(1~4)まで
        :return: N-gramの出現頻度が記録された辞書
        """
        if N > 4 or N < 1:
            print 'N should be set at 1, 2, 3, or 4.'
            return None

        # Ngramの作成
        list_sep_N = []
        for row in list_sep:
            list_tmp = []
            for i in range(len(row)-(N-1)):
                sentence = '_'.join(row[i:i+N])
                list_tmp.append(sentence)
            list_sep_N.append(list_tmp)

        dict_counter = collections.Counter([row1 for row in list_sep_N for row1 in row])

        return dict_counter

    @classmethod
    def rouge(cls, list_sep_test, list_sep_ans, N=1):
        dict_test_count = cls._get_ngram(list_sep_test, N=N)
        dict_ans_count = cls._get_ngram(list_sep_ans, N=N)
        # 分子と分母の初期化
        numer = 0
        denom = 0
        # 分母の計算
        for key, value in dict_ans_count.items():
            denom += dict_ans_count[key]

        # 分子の計算
        for key, value in dict_test_count.items():
            if key in dict_ans_count:
                numer += np.amin([dict_ans_count[key], value])

        return float(numer)/denom


class Modified_GraphSubModular(GraphSubModular):
    """
    グラフ間の構造を用いて類似度を測定して、劣モジュラ最適化を行う
    """
    def __init__(self):
        self._d_matrix = self._cal_distance_matrix()

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
        for row_id in range(self._list_bag):
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


class Modified_Vector(object):

    def __init__(self, list_sep_all, list_sep, dict_path):
        """
        :param list_bag: bag of words が記録されたリスト
        """
        self._list_bag = list_sep
        self._list_bag_all = list_sep_all
        # すべての単語の列
        self._list_all_word = [word for row in self._list_bag for word in row]
        # ユニークな単語の列
        self._list_unique_word = list(set(self._list_all_word))
        # 単語をid化
        self._dict_word_id = {word: i for i, word in enumerate(self._list_unique_word)}
        self._dict_id_word = {i: word for word, i in self._dict_word_id.items()}
        # 辞書の読み込み
        list_word_vec = Filer.readtsv(dict_path)
        self._dict_word_vec = {row[0]: np.array(row[1:], dtype=float) for row in list_word_vec}
        # 単語の分散表現のmatrix
        self._matrix_word_vec = np.array([self._dict_id_word[i] for i, word in enumerate(self._list_unique_word)])
        # tfidfの計算
        self._tfidf = self._cal_tfidf(list_bag=self._list_bag,
                                      dict_word_id=self._dict_word_id,
                                      dict_id_word=self._dict_id_word,
                                      num_row=len(self._list_bag),
                                      num_col=len(self._list_unique_word))
        # 距離行列の作成
        self._d_matrix = self._cal_matrix()
        # 最終的に出力するリスト
        self._list_C = []
        # 総単語数の表示
        print 'num word: ', len(self._list_all_word)
        # 語彙数の表示
        print 'num vocabulary: ', len(self._list_unique_word)
        # 辞書に登録されていない単語数の表示
        list_nonword = [word for word in self._list_unique_word if word not in self._dict_word_id]
        print 'Non dictionalized word: ', len(list_nonword)

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

    @property
    def dict_word_vec(self):
        print 'property: dict_word_vec'
        return self._dict_word_vec

    @dict_word_vec.setter
    def dict_word_vec(self, value):
        print 'setter: dict_word_vec'
        self._dict_word_vec = value

    @dict_word_vec.deleter
    def dict_word_vec(self):
        print 'deleter: dict_word_vec'
        del self._dict_word_vec
        self._dict_word_vec = {}

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

    def _cal_matrix(self):
        """
        距離行列を作成する
        :return: matrix
        """
        d_matrix = np.zeros((len(self._list_bag), len(self._list_bag)))
        # tfidfで重み付けをしたベクトルを作成する
        s_matrix = []
        for i, words in enumerate(self._list_bag):
            weight = self._tfidf.getrow(i).data
            weight = weight/np.float(np.sum(weight))
            indice = self._tfidf.getrow(i).indices
            s_vector = self._matrix_word_vec[indice] * weight[:,np.newaixs]
            s_vector = np.average(s_vector, axis=0)
            s_matrix.append[s_vector]

        for i, row1 in enumerate(s_matrix):
            for j, row2 in enumerate(s_matrix):
                d_matrix[i][j] = euclidean(row1, row2)

        return d_matrix

    def _cal_tfidf(self, list_bag, dict_word_id, dict_id_word, num_row, num_col):
        """
        bag of wordsからtfidfのmatrixを計算する
        :return: matrix(ただし、スパースマトリックス形式, csr_matrix)
        """
        # sparse_matrixのrow, col, dataを計算
        row_col_data = [[i, dict_word_id[word], float(j)/len(row)]
                        for i, row in enumerate(list_bag)
                        for word, j in collections.Counter(row).items()]

        row, col, data = zip(*row_col_data)
        # csr_matrixの初期化（この段階でtfになっている）
        matrix = csr_matrix((list(data), (list(row), list(col))),
                            shape=(num_row, num_col))
        # idfの辞書を作成
        list_idf = matrix.getnnz(axis=0)
        list_idf = [np.log(float(len(list_bag))/num)+1 for num in list_idf]
        dict_word_idf = {dict_id_word[i]: idf for i, idf in enumerate(list_idf)}

        # idfの辞書をもとにもう一度matrixを作成
        row_col_data = [[i, dict_word_id[word], float(j)/len(row)*dict_word_idf[word]]
                        for i, row in enumerate(list_bag)
                        for word, j in collections.Counter(row).items()]
        row, col, data = zip(*row_col_data)
        # csr_matrixの初期化（この段階でtf-idfになっている）
        matrix = csr_matrix((list(data), (list(row), list(col))),
                            shape=(num_row, num_col))

        return matrix


    def _cal_cost(self, list_C_id, scale):
        """
        コストの計算
        :param list_c_word: 現在採用している文の中に含まれる単語
        :param distance_matrix: W * Vの距離行列
        :param scale: スケール関数に何を使うか, 0: e^x, 1: x, 2: ln_x
        :return: f_C (計算したコスト)
        """
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

    def m_greedy(self, num_w=100, r=1, scale=0):
        """
        修正貪欲法による文章の抽出
        :param num_w: 抽出する単語数の上限
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
        C_word = 0
        # num_sで指定した文章数を抜き出すまで繰り返す
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
