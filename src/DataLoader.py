from torch.utils.data.dataloader import Dataset
from torch.utils.data.dataloader import DataLoader
from DataPipe import DataPipe
from config import global_config, stock_symbols, use_data, logger
import torch


class MyDataset(Dataset):
    def __init__(self, phase):
        self.phase = phase
        self.sample_list = self.get_all()
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    def __getitem__(self, index):
        return_sample_dict = self.sample_list[index]
        T = return_sample_dict['T']
        y = return_sample_dict['y']
        main_y = y[T-1]
        main_mv_percent = return_sample_dict['main_mv_percent']
        stock = return_sample_dict['stock']
        mv_percent = return_sample_dict['mv_percent']
        price = return_sample_dict['price']
        word = return_sample_dict['word']
        ss_index = return_sample_dict['ss_index']
        other_price = return_sample_dict['other_price']
        n_msgs = return_sample_dict['n_msgs']
        n_words = return_sample_dict['n_words']
        # mask = creat_mask(n_msgs.reshape(-1, self.config_model['max_n_days'])).reshape(self.config_model['max_n_days'],-1) #(5,30)
        # word用于输入，main_y是main target，ss_index股票索引位置用于BiGRU
        # price为价格，过linear以后计算跟msg的注意力，concat
        # y是跟数据对应的T天的target，T是max_n_days天中交易日的数目
        # n_msgs是每一个trading day的文章数目
        # n_words是每一篇文章中的单词数目

        # word = torch.from_numpy(word).to(self.device)
        # main_y = torch.from_numpy(main_y).to(self.device)
        # price = torch.from_numpy(price).to(self.device)
        # other_price = torch.from_numpy(other_price).to(self.device)

        return word, main_y, price, other_price
        # , n_words, mv_percent, main_mv_percent, stock, y, T, n_msgs, ss_index 暂时不用返回

    def __len__(self):
        return len(self.sample_list)

    def get_all(self):
        phase = self.phase
        data_pipe = DataPipe()
        word_id_dict = data_pipe.word_dict('TOKEN')
        stock_id_dict = data_pipe.word_dict('STOCK')
        generators = [data_pipe.sample_gen_from_one_stock(word_id_dict, stock_id_dict, s, phase) for s in stock_symbols]
        sample_list = data_pipe.generator2list(generators)
        return sample_list


def make_data(phase1, phase2):
    data_name = use_data
    logger.info('which data use: {}'.format(use_data))
    train_dataset = MyDataset(phase1)
    dev_dataset = MyDataset(phase2)
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=global_config['batch_size'],
                                  shuffle=global_config['shuffle'],
                                  num_workers=global_config['data_n_work'])

    dev_dataloader = DataLoader(dataset=dev_dataset,
                                batch_size=global_config['batch_size'],
                                shuffle=global_config['shuffle'],
                                num_workers=global_config['data_n_work'])

    return train_dataloader, dev_dataloader