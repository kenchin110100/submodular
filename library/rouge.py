# coding: utf-8
"""
Rougeの計算をするためのクラス
"""
import collections
import numpy as np


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
    def rouge(cls, list_sep_test, list_list_sep_ans, N=1):
        dict_test_count = cls._get_ngram(list_sep_test, N=N)
        list_dict_ans_count = [cls._get_ngram(row, N=N) for row in list_list_sep_ans]
        # 分子と分母の初期化
        numer = 0
        denom = 0
        # 分母の計算
        for dict_ans_count in list_dict_ans_count:
            for key, value in dict_ans_count.items():
                denom += dict_ans_count[key]

        # 分子の計算
        for dict_ans_count in list_dict_ans_count:
            for key, value in dict_test_count.items():
                if key in dict_ans_count:
                    numer += np.amin([dict_ans_count[key], value])

        return float(numer)/denom
