# coding: utf-8
"""
劣モジュラ最適化によって文書要約をするためのクラス
"""
from filer2.filer2 import Filer
from igraph import *
import collections
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.spatial.distance import cosine, sqeuclidean, euclidean
import itertools

class SubModular(object):
    def __init__(self, list_bag=None, list_edgelist=None, directed=True, inverse_flag=True):
        # 有向グラフで計算するか、無向グラフで計算するか
        self._directed = directed
        # 距離を計算するときに、inverseにするのか、マイナスにするのか
        self._inverse_flag = inverse_flag

        # エッジリストとして入力するか、bag of wordsとして入力するか
        if list_bag != None and list_edgelist != None:
            # bag_of_words
            self._list_bag = list_bag
            # inputfileからnode, edge, weightを計算
            self._list_node, self._list_edge, self._list_weight = self._cal_node_edge_weight(list_bag)
            # エッジの再計算
            list_edge = [tuple(row) for row in list_edgelist]
            tuple_edge, tuple_weight = zip(*collections.Counter(list_edge).items())
            self._list_edge = list(tuple_edge)
            self._list_weight = list(tuple_weight)
            # エッジリストにはあるがノードリストにはない単語を追加する
            list_node_add = list(set([word for row in self._list_edge for word in row]))
            self._list_node = list(set(self._list_node + list_node_add))

        elif list_bag != None and list_edgelist == None:
            # bag_of_words
            self._list_bag = list_bag
            # inputfileからnode, edge, weightを計算
            self._list_node, self._list_edge, self._list_weight = self._cal_node_edge_weight(list_bag)

        else:
            print '入力ファイルが足りません'
            raise
            
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
            list_edge = [tuple(sorted(row)) for row in list_edgelist]
            # ノードリスト
            list_node = list(set([word for row in list_edgelist for word in row]))
            # エッジリストとそのweightを作成
            tuple_edge, tuple_weight = zip(*collections.Counter(list_edge).items())

            return list_node, list(tuple_edge), list(tuple_weight)
        
    def _cal_bag_edgelist(self, list_bag):
        """
        bag_of_wordsをedgelistに変換する
        list_bag: bag_of_words
        return: list_edgelist
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

    def _m_greedy_1(self, list_C, list_id_document, r=1, scale=0):
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
            for doc_id, document in list_id_document:
                # documentに含まれる単語のリスト
                list_c_word = sorted(list(set([word for word in document])))
                # スコアの計算
                f_C = self._cal_cost(list_c_word=list_c_word,
                                     scale=scale)
                f_C = f_C/(np.power(len(document), r))
                # リストにidとbagとスコアを記録
                list_id_score.append([doc_id, document, f_C])
            # スコアが最大になるものを取得
            doc_id, document, _ = sorted(list_id_score, key=lambda x: x[2], reverse=True)[0]
            return [doc_id, document]
        # list_Cが空ではないとき
        else:
            # 現在のlist_Cに含まれるユニークな単語のリストを作成
            list_c_word = sorted(list(set([word for row in list_C for word in row[1]])))
            # f_C: 現在のコストの計算
            f_C = self._cal_cost(list_c_word=list_c_word,
                                 scale=scale)
            print f_C
            # 文書を一つずつ追加した時のスコアの増分を計算する
            list_id_score = []
            for doc_id, document in list_id_document:
                # 文章の追加
                list_c_word_s = list(set(list_c_word + document))
                # コストの計算
                f_C_s = self._cal_cost(list_c_word=list_c_word_s,
                                       scale=scale)
                # スコアの増分を計算
                delta = (f_C_s - f_C) / np.power(len(document), r)
                # スコアの増分を記録
                list_id_score.append([doc_id, document, delta])
            # スコアの増分が一番大きかったdocを返す
            doc_id, document, _ = sorted(list_id_score, key=lambda x: x[2], reverse=True)[0]
            return [doc_id, document]

    def m_greedy(self, num_s=5, r=1, scale=0):
        """
        修正貪欲法による文章の抽出
        :param num_s: 抽出する文書数
        :param r: 単語数に対するコストのパラメータ
        :param scale: スケーリング関数, 0: e^x, 1: x, 2: ln_x
        :return: 抽出した文章のidとそのbag_of_wordsのリスト
        """
        # list_id_documentの作成
        list_id_document = [[i, row] for i, row in enumerate(self._list_bag)]
        # 要約文書のリスト
        list_C = []
        # num_sで指定した文章数を抜き出すまで繰り返す
        while len(list_C) < num_s:
            # コストが一番高くなる組み合わせを計算
            doc_id, doc = self._m_greedy_1(list_C=list_C,
                                           list_id_document=list_id_document,
                                           r=r, scale=scale)
            # 採用したリストをappend
            list_C.append([doc_id, doc])
            # 元の集合からremove
            list_id_document.remove([doc_id, doc])

        self._list_C = list_C

# ベクトル操作をするためのクラス
class Vector(object):

    def __init__(self, list_bag, dict_path):
        """
        :param list_bag: bag of words が記録されたリスト
        """
        self._list_bag = list_bag
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

    def _m_greedy_1(self, list_C, list_id_document, r=1, scale=0):
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
            for doc_id, document in list_id_document:
                # documentに含まれる単語のリスト
                list_c_word = sorted(list(set([word for word in document])))
                # スコアの計算
                f_C = self._cal_cost(list_c_word=list_c_word,
                                     scale=scale)
                f_C = f_C/(np.power(len(document), r))
                # リストにidとbagとスコアを記録
                list_id_score.append([doc_id, document, f_C])
            # スコアが最大になるものを取得
            doc_id, document, _ = sorted(list_id_score, key=lambda x: x[2], reverse=True)[0]
            return [doc_id, document]
        # list_Cが空ではないとき
        else:
            # 現在のlist_Cに含まれるユニークな単語のリストを作成
            list_c_word = sorted(list(set([word for row in list_C for word in row[1]])))
            # f_C: 現在のコストの計算
            f_C = self._cal_cost(list_c_word=list_c_word,
                                 scale=scale)
            print f_C
            # 文書を一つずつ追加した時のスコアの増分を計算する
            list_id_score = []
            for doc_id, document in list_id_document:
                # 文章の追加
                list_c_word_s = list(set(list_c_word + document))
                # コストの計算
                f_C_s = self._cal_cost(list_c_word=list_c_word_s,
                                       scale=scale)
                # スコアの増分を計算
                delta = (f_C_s - f_C) / np.power(len(document), r)
                # スコアの増分を記録
                list_id_score.append([doc_id, document, delta])
            # スコアの増分が一番大きかったdocを返す
            doc_id, document, _ = sorted(list_id_score, key=lambda x: x[2], reverse=True)[0]
            return [doc_id, document]

    def m_greedy(self, num_s=5, r=1, scale=0):
        """
        修正貪欲法による文章の抽出
        :param num_s: 抽出する文書数
        :param r: 単語数に対するコストのパラメータ
        :param scale: スケーリング関数, 0: e^x, 1: x, 2: ln_x
        :return: 抽出した文章のidとそのbag_of_wordsのリスト
        """
        # list_id_documentの作成
        list_id_document = [[i, row] for i, row in enumerate(self._list_bag)]
        # 要約文書のリスト
        list_C = []
        # num_sで指定した文章数を抜き出すまで繰り返す
        while len(list_C) < num_s:
            # コストが一番高くなる組み合わせを計算
            doc_id, doc = self._m_greedy_1(list_C=list_C,
                                           list_id_document=list_id_document,
                                           r=r, scale=scale)
            # 採用したリストをappend
            list_C.append([doc_id, doc])
            # 元の集合からremove
            list_id_document.remove([doc_id, doc])

        self._list_C = list_C


    
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