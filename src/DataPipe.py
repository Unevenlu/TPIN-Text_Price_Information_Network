import itertools
import os
from config import dates, global_config, config_control, stock_symbols, vocab, vocab_size, path_parser, ss_size, logger, use_data
import io
import numpy as np
from datetime import datetime,timedelta
import random
import json
import copy


class DataPipe:
    def __init__(self):
        # load path
        self.movement_path = path_parser.movement
        self.msg_path = path_parser.preprocessed
        self.vocab_path = path_parser.vocab
        self.word_vec_path = path_parser.word_vec

        # load dates
        self.train_start_date = dates['train_start_date']  # '2014-01-01'
        self.train_end_date = dates['train_end_date']  # '2015-08-01'
        self.dev_start_date = dates['dev_start_date']  # '2015-08-01'
        self.dev_end_date = dates['dev_end_date']  # '2015-10-01'
        self.test_start_date = dates['test_start_date']  # '2015-10-01'
        self.test_end_date = dates['test_end_date']  # '2016-01-01'
        self.unit_test_start_date = dates['unit_test_start_date']
        self.unit_test_end_date = dates['unit_test_start_date']

        # load model config
        self.shuffle = global_config['shuffle']  # 1
        self.max_n_days = global_config['max_n_days']  # 5
        self.max_n_words = global_config['max_n_words']  # 40
        self.max_n_msgs = global_config['max_n_msgs']  # 30

        self.word_embed_type = global_config['word_embed_type']  # 'glove'
        # self.cn_word_vec_type = global_config['cn_word_vec_type']
        self.word_embed_size = global_config['word_embed_size']  # 50
        self.stock_embed_size = global_config['stock_embed_size']  # 150
        self.init_stock_with_word = global_config['init_stock_with_word']  # 0
        self.price_embed_size = global_config['word_embed_size']  # 50
        self.y_size = global_config['y_size']  # 2
        self.vocab_size = vocab_size

        # contorl
        self.use_data = config_control['use_data']

        # stock data
        self.stock_data = self.get_stock_data()

        assert self.word_embed_type in ('rand', 'glove')
        # assert self.cn_word_vec_type in ('rand', 'pretrain')

    def word_dict(self, key='TOKEN'):
        if key == 'TOKEN':
            id2word_dict = dict()   # 存储字符数字索引的dict

            word_list = list(vocab)
            word_list.insert(0, 'UNK')   # UNK

            for id in range(len(word_list)):
                id2word_dict[word_list[id]] = id    #word2Id，如：{'UNK':0,'fawn':1,...}

            return id2word_dict

        elif key == 'STOCK':
            ss_id_dict = dict()

            for id in range(len(stock_symbols)):
                ss_id_dict[stock_symbols[id]] = id

            return ss_id_dict

    # path 为词向量文件地址
    def word_table(self):
        path = self.word_vec_path
        word_embed_tabel = np.random.random((self.vocab_size, self.word_embed_size)) * 2 - 1  # [-1.0, 1.0]
        n_replacement = 0
        vocab_id_dict = self.word_dict('TOKEN')

        with io.open(path, 'r', encoding='utf-8') as f:
            for line in f:  # line--[wrod embedding-50d]
                tuples = line.split()
                word, embed = tuples[0], [float(embed_col) for embed_col in tuples[1:]]
                if word in ['<unk>', 'unk']:  # unify UNK
                    word = 'UNK'
                if word in vocab_id_dict:
                    n_replacement += 1
                    word_id = vocab_id_dict[word]
                    word_embed_tabel[word_id] = embed
        return word_embed_tabel

    def get_stock_data(self):
        stock_data_dict = {}
        for s_s in stock_symbols:
            s_s_prices_dict = {}
            movement_path = os.path.join(str(self.movement_path), '{}.txt'.format(s_s))  # 股票价格数据
            # 该处有问题，改成先读取各个股票价格信息，再从中取好一些，不然每一次都要打开87个文件，再遍历其中每一行，效率太低
            with io.open(movement_path, 'r', encoding='utf8') as movement_f:
                for line in movement_f:  # descend
                    data = line.split('\t')
                    t = datetime.strptime(data[0], '%Y-%m-%d').date()
                    # logger.info(t)
                    s_s_prices_dict[t] = data
            if use_data == 'my_am' or use_data == 'my_cn':
                s_s_prices_list = list(s_s_prices_dict.items())
                s_s_prices_list.reverse()
                s_s_prices_dict = dict(s_s_prices_list)
            stock_data_dict[s_s] = s_s_prices_dict

        return stock_data_dict

    # 各个阶段的开始与结束日期
    def _get_start_end_date(self, phase):
        """
            phase: train, dev, test, unit_test
            => start_date & end_date
        """
        assert phase in {'train', 'dev', 'test', 'whole', 'unit_test'}
        if phase == 'train':
            return self.train_start_date, self.train_end_date
        elif phase == 'dev':
            return self.dev_start_date, self.dev_end_date
        elif phase == 'test':
            return self.test_start_date, self.test_end_date
        elif phase == 'whole':
            return self.train_start_date, self.test_end_date
        else:
            return self.unit_test_start_date, self.unit_test_end_date

    # 将token转化为id
    @staticmethod
    def _convert_token_to_id(token, token_id_dict):
        if token not in token_id_dict:
            token = 'UNK'
        return token_id_dict[token]

    # 将words种的word转为id，以list返回
    def _convert_words_to_ids(self, words, vocab_id_dict):
        """
            Replace each word in the data set with its index in the dictionary

        :param words: words in tweet
        :param vocab_id_dict: dict, vocab-id
        :return:
        """
        return [self._convert_token_to_id(w, vocab_id_dict) for w in words]

    # 根据股票标识ss，获取对应date的语料标识
    def _get_unaligned_corpora(self, ss, main_target_date, vocab_id_dict):
        def get_ss_index(word_seq, ss):
            ss = ss.lower()
            ss_index = len(word_seq) - 1  # init
            if ss in word_seq:  # 若找到
                ss_index = word_seq.index(ss)   #取出其index
            else:
                if '$' in word_seq: #找到‘$’
                    dollar_index = word_seq.index('$')  #其index+1
                    if dollar_index is not len(word_seq) - 1 and ss in word_seq[dollar_index + 1]: #？
                        ss_index = dollar_index + 1
                    else:
                        for index in range(dollar_index + 1, len(word_seq)):
                            if ss in word_seq[index]:
                                ss_index = index
                                break
            return ss_index

        unaligned_corpora = list()  # list of sets: (d, msgs, ss_indices)
        stock_msg_path = os.path.join(str(self.msg_path), ss)   # 找到对应股票的文本资料所在文件夹

        # 非main target的日期
        d_d_max = main_target_date - timedelta(days=1)  # day_T-1
        d_d_min = main_target_date - timedelta(days=self.max_n_days)    # day_T-5

        d = d_d_max  # descend
        while d >= d_d_min:
            msg_fp = os.path.join(stock_msg_path, d.isoformat())  # 那一天的文本路径,如：../tweet/preprocessed/AAPL/2014-01-01
            if os.path.exists(msg_fp):  # 若存在
                word_mat = np.zeros([self.max_n_msgs, self.max_n_words], dtype=np.int32)    #(30,40)
                n_word_vec = np.zeros([self.max_n_msgs, ], dtype=np.int32)  #(30,),每个message的单词数目
                ss_index_vec = np.zeros([self.max_n_msgs, ], dtype=np.int32)    #(30,)，股票标识ss的索引
                msg_id = 0
                with open(msg_fp, 'r', encoding='utf-8') as msg_f:
                    for line in msg_f:
                        msg_dict = json.loads(line)  # 按照字典方式读入
                        text = msg_dict['text']

                        if not text:
                            continue    # 当天无文本则跳过

                        words = text[:self.max_n_words]
                        word_ids = self._convert_words_to_ids(words, vocab_id_dict)  # word2Id
                        n_words = len(word_ids)  # max_n_words

                        n_word_vec[msg_id] = n_words    # 这一天的30个文本中分别有多少个词，方便生成mask
                        word_mat[msg_id, :n_words] = word_ids   # 序号为msg_id的文本的单词id标识
                        ss_index_vec[msg_id] = get_ss_index(words, ss)  # 得到序号为msg_id的文本中股票标识ss的位置索引

                        msg_id += 1  # 文章索引
                        if msg_id == self.max_n_msgs:   # 若已经够30篇
                            break
                corpus = [d, word_mat[:msg_id], ss_index_vec[:msg_id], n_word_vec[:msg_id], msg_id]
                # [2014-05-12,(2, 40),(2,),(2,)]

                unaligned_corpora.append(corpus)
            d -= timedelta(days=1)

        unaligned_corpora.reverse()  # ascend
        return unaligned_corpora

    def _get_prices_and_ts(self, ss, main_target_date):

        def _get_mv_class(data, use_one_hot=False):
            mv = float(data[1])
            if self.y_size == 2:
                if mv <= 1e-7:
                    return [1.0, 0.0] if use_one_hot else 0     #<1e-7认为是下降
                else:
                    return [0.0, 1.0] if use_one_hot else 1     #否则认为是上升

            if self.y_size == 3:
                threshold_1, threshold_2 = -0.004, 0.005
                if mv < threshold_1:
                    return [1.0, 0.0, 0.0] if use_one_hot else 0
                elif mv < threshold_2:
                    return [0.0, 1.0, 0.0] if use_one_hot else 1
                else:
                    return [0.0, 0.0, 1.0] if use_one_hot else 2

        def _get_y(data):
            return _get_mv_class(data, use_one_hot=True)

        def _get_prices(data):
            return [float(p) for p in data[3:6]]

        def _get_mv_percents(data):
            return _get_mv_class(data)

        ts, ys, prices, mv_percents, other_prices, main_mv_percent = list(), list(), list(), list(), list(), 0.0
        d_t_min = main_target_date - timedelta(days=self.max_n_days-1)  # 辅助的day target
        # 如6-24，6-24是main target day，
        # 我们的语料用的是6-19 -- 6-23 的，
        # 我们的辅助target day 就是6-20--6-23

        ss_datas = self.stock_data[ss]
        for t in ss_datas:
            data = ss_datas[t]
            # logger.info(t)
            if t == main_target_date:
                # logger.info(t)
                ts.append(t)  # day
                ys.append(_get_y(data))  # [1.0,0] or [0.0,1.0]
                main_mv_percent = data[1]  # 移动趋势
                if -0.005 <= float(main_mv_percent) < 0.0055:  # 丢弃低变动率样本，discard sample with low movement percent
                    return None
            if d_t_min <= t < main_target_date:  # aux day
                ts.append(t)
                ys.append(_get_y(data))
                prices.append(_get_prices(data))  # high, low, close
                mv_percents.append(_get_mv_percents(data))  # 0 or 1
            if t < d_t_min:  # one additional line for x_1_prices. not a referred trading day
                prices.append(_get_prices(data))
                mv_percents.append(_get_mv_percents(data))
                break

        T = len(ts)
        if len(ys) != T or len(prices) != T or len(mv_percents) != T:  # ensure data legibility
            return None

        for s_s in stock_symbols:
            if s_s == ss:
                continue
            data_dict = self.stock_data[s_s]
            s_s_prices = list()
            for t in data_dict:
                data = data_dict[t]
                # logger.info(t)
                if d_t_min <= t < main_target_date:  # aux day
                    s_s_prices.append(_get_prices(data))  # high, low, close
                if t < d_t_min:  # one additional line for x_1_prices. not a referred trading day
                    s_s_prices.append(_get_prices(data))
                    break

            while len(s_s_prices) < T:
                s_s_prices.append([0.0, 0.0, 0.0])
            s_s_prices.reverse()
            other_prices.append(s_s_prices)

        # ascend
        for item in (ts, ys, mv_percents, prices):
            item.reverse()

        # 最终返回dict
        prices_and_ts = {
            'T': T,     # 这批样本中的交易日个数
            'ts': ts,   # [minday -- main target day]
            'ys': ys,   # [minday -- main target day]的y
            'main_mv_percent': main_mv_percent,  # main day的移动趋势
            'mv_percents': mv_percents,     # [minday-1 -- main target day-1]上升或下降的类别
            'prices': prices,   # [minday-1 -- main target day-1]的股价
            'other_prices': other_prices    # 其他股票[minday-1 -- main target day-1]的股价
        }

        return prices_and_ts

    def _trading_day_alignment(self, ts, T, unaligned_corpora):
        aligned_word_tensor = np.zeros([T, self.max_n_msgs, self.max_n_words], dtype=np.int32)  #(T,30,40)
        aligned_ss_index_mat = np.zeros([T, self.max_n_msgs], dtype=np.int32)   #(T,30),T天中每个消息中的ss索引位置
        aligned_n_words_mat = np.zeros([T, self.max_n_msgs], dtype=np.int32)    #(T,30),T天中每个文章含有的单词数
        aligned_n_msgs_vec = np.zeros([T, ], dtype=np.int32)

        # list for gathering
        aligned_msgs = [[] for _ in range(T)]
        aligned_ss_indices = [[] for _ in range(T)]
        aligned_n_words = [[] for _ in range(T)]
        aligned_n_msgs = [[] for _ in range(T)]

        corpus_t_indices = []
        max_threshold = 0

        for corpus in unaligned_corpora:
            d = corpus[0]
            for t in range(T):
                if d < ts[t]:
                    corpus_t_indices.append(t)  # aux day的索引？
                    break

        assert len(corpus_t_indices) == len(unaligned_corpora)

        for i in range(len(unaligned_corpora)):
            corpus, t = unaligned_corpora[i], corpus_t_indices[i]
            word_mat, ss_index_vec, n_word_vec, n_msgs = corpus[1:]
            aligned_msgs[t].extend(word_mat)    # message的40个词id标识，不足40后续补0
            aligned_ss_indices[t].extend(ss_index_vec)  # 股票标识索引
            aligned_n_words[t].append(n_word_vec)   # message单词数目
            aligned_n_msgs[t].append(n_msgs)    # 文章数目

        def is_eligible():
            n_fails = len([0 for n_msgs in aligned_n_msgs if sum(n_msgs) == 0])
            return n_fails <= max_threshold

        if not is_eligible():
            return None

        # gather into nd_array and clip exceeded part
        for t in range(T):
            n_msgs = sum(aligned_n_msgs[t])

            if aligned_msgs[t] and aligned_ss_indices[t] and aligned_n_words[t]:
                msgs, ss_indices, n_word = np.vstack(aligned_msgs[t]), np.hstack(aligned_ss_indices[t]), np.hstack(aligned_n_words[t])
                assert len(msgs) == len(ss_indices) == len(n_word)
                n_msgs = min(n_msgs, self.max_n_msgs)  # clip length
                aligned_n_msgs_vec[t] = n_msgs
                aligned_word_tensor[t, :n_msgs] = msgs[:n_msgs]
                aligned_ss_index_mat[t, :n_msgs] = ss_indices[:n_msgs]
                aligned_n_words_mat[t, :n_msgs] = n_word[:n_msgs]

        aligned_info_dict = {
            'msgs': aligned_word_tensor,
            'ss_indices': aligned_ss_index_mat,
            'n_words': aligned_n_words_mat,
            'n_msgs': aligned_n_msgs_vec,
        }

        return aligned_info_dict

    def sample_gen_from_one_stock(self, vocab_id_dict, stock_id_dict, s, phase):
        """
            generate samples for the given stock.

            => tuple, (x:dict, y_:int, price_seq: list of floats, prediction_date_str:str)
        """
        start_date, end_date = self._get_start_end_date(phase)  #各个阶段的开始与截至日期
        stock_movement_path = os.path.join(str(self.movement_path), '{}.txt'.format(s)) #根据股票标识，找到对应股票文本

        main_target_dates = []      # 主要预测目标的日期list

        with open(stock_movement_path, 'r') as movement_f:
            for line in movement_f:
                data = line.split('\t')
                main_target_date = datetime.strptime(data[0], '%Y-%m-%d').date()    # data类型生成日期
                main_target_date_str = main_target_date.isoformat()  # ‘xxxx-xx-xx’

                if start_date <= main_target_date_str < end_date:   # 对时间范围内的日期
                    main_target_dates.append(main_target_date)  # 将对应日期的data类型放入list

        if self.shuffle:  # shuffle data
            random.shuffle(main_target_dates)   # 将日期打乱顺序

        for main_target_date in main_target_dates:
            # logger.info('start _get_unaligned_corpora')
            unaligned_corpora = self._get_unaligned_corpora(s, main_target_date, vocab_id_dict)  # 生成一个sample，即根据shuffle后的日期生成之前的候选辅助贸易日的语料
            # logger.info('start _get_prices_and_ts')
            prices_and_ts = self._get_prices_and_ts(s, main_target_date)    # 得到目标day往下的各种信息，如[6-19,6-23]的价格，与移动趋势， [6-20,6-24]的y，[1,0] or [0,1]
            if not prices_and_ts:   # 若返回None，直接进入下一个main taget day 对应的sample
                continue

            # logger.info('start _trading_day_alignment')
            aligned_info_dict = self._trading_day_alignment(prices_and_ts['ts'], prices_and_ts['T'], unaligned_corpora)
            if not aligned_info_dict:
                continue

            sample_dict = {
                # meta info
                'stock': self._convert_token_to_id(s, stock_id_dict),
                'main_target_date': main_target_date.isoformat(),
                'T': prices_and_ts['T'],
                # target
                'ys': prices_and_ts['ys'],
                'main_mv_percent': prices_and_ts['main_mv_percent'],
                'mv_percents': prices_and_ts['mv_percents'],
                # source
                'prices': prices_and_ts['prices'],
                'other_prices': prices_and_ts['other_prices'],
                'msgs': aligned_info_dict['msgs'],
                'ss_indices': aligned_info_dict['ss_indices'],
                'n_words': aligned_info_dict['n_words'],
                'n_msgs': aligned_info_dict['n_msgs'],
            }

            yield sample_dict

    def generator2list(self, generators, phase):
        max_n_days = self.max_n_days
        max_n_msgs = self.max_n_msgs
        max_n_words = self.max_n_words
        y_size = self.y_size
        sample_list = list()
        numl = 0
        while(True):
            try:
                sample_dict = next(generators[0])
                stock = sample_dict['stock']
                T = sample_dict['T']
                y = np.zeros([max_n_days, y_size], dtype=np.float32)  # (32,5,2)
                main_mv_percent = sample_dict['main_mv_percent']
                mv_percent = np.zeros([max_n_days, ], dtype=np.float32)  # (30,5)
                price = np.zeros([max_n_days, 3], dtype=np.float32)  # (32,5,3)
                word = np.zeros([max_n_days, max_n_msgs, max_n_words],
                                dtype=np.int32)  # (32,5,30,40)
                ss_index = np.zeros([max_n_days, max_n_msgs], dtype=np.int32)  # (32,5,30)
                n_msgs = np.zeros([max_n_days], dtype=np.int32)  # (32,5)
                n_words = np.zeros([max_n_days, max_n_msgs], dtype=np.int32)  # (32,5,30)
                other_price = np.zeros([ss_size-1, max_n_days, 3], dtype=np.float32)

                y[:T] = sample_dict['ys']
                mv_percent[:T] = sample_dict['mv_percents']
                # source
                price[:T] = sample_dict['prices']
                word[:T] = sample_dict['msgs']
                ss_index[:T] = sample_dict['ss_indices']
                n_msgs[:T] = sample_dict['n_msgs']
                n_words[:T] = sample_dict['n_words']
                other_price[:, :T] = sample_dict['other_prices']

                return_sample_dict = dict()
                return_sample_dict['T'] = T
                return_sample_dict['y'] = y
                return_sample_dict['main_mv_percent'] = main_mv_percent
                return_sample_dict['stock'] = stock
                return_sample_dict['mv_percent'] = mv_percent
                return_sample_dict['price'] = price
                return_sample_dict['word'] = word
                return_sample_dict['ss_index'] = ss_index
                return_sample_dict['n_msgs'] = n_msgs
                return_sample_dict['n_words'] = n_words
                return_sample_dict['other_price'] = other_price
                sample_list.append(return_sample_dict)

                # numl += 1
                # stop_num = 100
                # if stop_num <= 1000:
                #     if numl == stop_num:
                #         break
                # else:
                #     if numl == stop_num / 5 and phase == 'dev':
                #         break
                #     if numl == stop_num and phase == 'train':
                #         break

            except StopIteration:
                del generators[0]
                if generators:
                    continue
                else:
                    break
        return sample_list

