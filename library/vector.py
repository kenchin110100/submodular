# coding: utf-8
"""
劣モジュラ最適化によって文書要約をするためのクラス
"""
from filer2.filer2 import Filer
from igraph import *
import collections
import numpy as np
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cosine, sqeuclidean, euclidean
import copy


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
        self._dict_word_vec = Filer.readdump(dict_path)
        # 距離行列の作成
        self._matrix = self._cal_matrix()
        # 最終的に出力するリスト
        self._list_C = []
        # 総単語数の表示
        print 'num word: ', len(self._list_all_word)
        # 語彙数の表示
        print 'num vocabulary: ', len(self._list_unique_word)
        # 辞書に登録されていない単語数の表示
        list_nonword = [word for word in self._list_unique_word if word not in self._dict_word_vec.keys()]
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


class Modified_Vector(object):

    def __init__(self, list_sep_all, list_sep, dict_path):
        """
        :param list_bag: bag of words が記録されたリスト
        """
        self._list_bag_all = list_sep_all
        # 辞書の読み込み
        self._dict_word_vec = Filer.readdump(dict_path)
        list_keys = self._dict_word_vec.keys()
        self._list_bag = [[word for word in row if word in list_keys]
                          for row in list_sep]
        # すべての単語の列
        self._list_all_word = [word for row in self._list_bag for word in row]
        # ユニークな単語の列
        self._list_unique_word = list(set(self._list_all_word))
        # 単語をid化
        self._dict_word_id = {word: i for i, word in enumerate(self._list_unique_word)}
        self._dict_id_word = {i: word for word, i in self._dict_word_id.items()}
        # 単語の分散表現のmatrix
        self._matrix_word_vec = np.array([self._dict_word_vec[word] for i, word in enumerate(self._list_unique_word)])
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
            s_vector = self._matrix_word_vec[indice] * weight[:,np.newaxis]
            s_vector = np.average(s_vector, axis=0)
            s_matrix.append(s_vector)

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
