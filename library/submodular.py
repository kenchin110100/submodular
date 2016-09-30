# coding: utf-8
"""
劣モジュラ最適化によって文書要約をするためのクラス
"""
from igraph import *
import collections
import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.cluster import KMeans
import copy
from filer2.filer2 import Filer


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
            self._list_C = [[max_id,
                            self._list_bag_u[max_id],
                            self._list_bag_u_all[max_id]]]

        print '計算が終了しました'
        

class OpinionSubModular(object):
    def __init__(self, list_sep, list_sep_lemmas, dict_path):
        """
        Monotone Submodularity in Opinion Summaries (Jayanth, et al, 2015)の実装
        :param list_bag: bag of wordsが記録されたリスト
        """
        # インプットデータの読み込み
        self._list_bag_u = list_sep
        self._list_bag_u_lemmas = list_sep_lemmas
        self._dict_path = dict_path

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
        # 極性辞書の読み込み
        self._dict_word_pol = Filer.readdump(self._dict_path)
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
        # 各文書の極性を計算
        self._arr_s = []
        for row in self._list_bag_u_lemmas:
            s_tmp = 0
            for word in row:
                if word in self._dict_word_pol:
                    s_tmp += self._dict_word_pol[word][0] - self._dict_word_pol[word][1]
            self._arr_s.append(s_tmp)
        self._arr_s = np.array(self._arr_s)
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
        for cluster, r in zip(self._arr_cluster[list_id], self._arr_s[list_id]):
            dict_cluster_r[cluster].append(r)

        # R(S)の計算
        R = 0
        for _, value in dict_cluster_r.items():
            R += np.max(value)

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
            F_old = lamda*L_old + (1-lamda)*R_old
        for rest_id in list_rest_id:
            L_tmp = self._cal_L(self._list_C_id+[rest_id], alpha)
            R_tmp = self._cal_R(self._list_C_id+[rest_id])
            F_tmp = lamda*L_tmp + (1-lamda)*R_tmp
            delta = (F_tmp - F_old) / np.power(len(self._list_bag_u[rest_id]), r)
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
            if C_word + len(self._list_bag_u[next_id]) <= num_w:
                self._list_C_id.append(next_id)
                self._list_C.append([next_id,
                                     self._list_bag_u[next_id]])
                C_word += len(self._list_bag_u[next_id])
            list_rest_id.remove(next_id)
        
        # 全体から一つだけ抽出
        max_delta = 0
        max_id = 0
        for rest_id in list_rest_id_copy:
            L_tmp = self._cal_L([rest_id], alpha)
            R_tmp = self._cal_R([rest_id])
            F_tmp = lamda*L_tmp + (1-lamda)*R_tmp
            if F_tmp > max_delta:
                max_delta = F_tmp
                max_id = rest_id
                
        # Fを比較
        L_c = self._cal_L(self._list_C_id, alpha)
        R_c = self._cal_R(self._list_C_id)
        F_c = L_c + lamda*R_c

        if max_delta > F_c:
            self._list_C_id = [max_id]
            self._list_C = [[max_id, self._list_bag_u[max_id]]]

        print '計算が終了しました'