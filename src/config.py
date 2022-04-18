import yaml
import itertools
import os
import io
import json
import sys
import logging.config


class PathParser:
    def __init__(self, config_path):
        self.root = '../'
        self.log = os.path.join(self.root, config_path['log'])
        self.save = os.path.join(self.root, config_path['save'])
        self.data = os.path.join(self.root, config_path['data'])
        self.res = os.path.join(self.root, config_path['res'])
        self.checkpoints = os.path.join(self.root, config_path['checkpoints'])
        self.word_vec = os.path.join(self.res, config_path['word_vec'])
        self.raw = os.path.join(self.data, config_path['msg_raw'])
        self.preprocessed = os.path.join(self.data, config_path['msg_preprocessed'])
        self.movement = os.path.join(self.data, config_path['price'])
        self.vocab = os.path.join(self.res, config_path['vocab'])
        self.graphs = os.path.join(self.root, config_path['graphs'])


config_fp = os.path.join(os.path.dirname(__file__), 'config.yml')
config = yaml.load(open(config_fp, 'r',encoding='utf-8'),Loader=yaml.FullLoader)
global_config = config['global variable'] #模型参数dict
layer_config = config['layer variable']

# control fig
config_control = config['control']
use_data = config_control['use_data']

# date
if use_data == 'stocknet':
    dates = config['dates']
else:
    dates = config['my_dates']

# path
if use_data == 'stocknet':
    path_config = config['paths']['stocknet_path']
elif use_data == 'my_am':
    path_config = config['paths']['my_am_path']
else:
    path_config = config['paths']['my_cn_path']

# stock
if use_data == 'stocknet':
    config_stocks = config['stocks']  # a list of lists
elif use_data == 'my_am':
    config_stocks = config['my_stocks']['am_stocks']  # a list of lists
else:
    config_stocks = config['my_stocks']['cn_stocks']
list_of_lists = [config_stocks[key] for key in config_stocks]
stock_symbols = list(itertools.chain.from_iterable(list_of_lists)) # 各股票标签
ss_size = len(stock_symbols)    # 股票数目

path_parser = PathParser(config_path=path_config)

# logger
logger = logging.getLogger('my_logger')
logger.setLevel(logging.DEBUG)
log_fp = os.path.join(path_parser.log, '{0}.log'.format('model'))   # ../log/model.log
file_handler = logging.FileHandler(log_fp)
console_handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# stocknet vocab
with io.open(str(path_parser.vocab), 'r', encoding='utf-8') as vocab_f:
    vocab = json.load(vocab_f)  # vocab list
    vocab_size = len(vocab) + 1  # 29867,for unk
