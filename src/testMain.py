import os
import copy

import numpy
import torch
import numpy as np
from NewModel import model_info, CaseStudyModel
from config import stock_symbols
from DataLoader import make_data
import time
from config import logger
import torch.nn as nn
from metrics import cal_mcc_num, cal_acc, cal_mcc, CustomSchedule
from DataPipe import DataPipe
import matplotlib.pyplot as plt


def find_stock(main_stock, index):
    func_use_list = copy.deepcopy(stock_symbols)
    func_use_list.remove(main_stock)

    return func_use_list[index]


def key2value(word_dict):
    return_dict = {}
    for key in word_dict:
        value = word_dict[key]
        return_dict[value] = key
    return return_dict


def id2word(word, id2word_dict):
    '''
    :param word: (batch_size, n_head)
    :return:
    '''
    word_list = list(word)
    str_list = []
    for tensor in word_list:
        char_list = []
        for i in range(len(tensor)):
            if tensor[i].item() != 0:
                char = id2word_dict[tensor[i].item()]
            else:
                char = ' '
            char_list.append(char)
        string = ' '.join(char_list)
        str_list.append(string)
    return str_list


if __name__ == '__main__':
    word_dict = DataPipe().word_dict()
    id2word_dict = key2value(word_dict)
    use_gpu = torch.cuda.is_available()
    print("GPU environment: ", use_gpu)
    device_name = 'cuda' if use_gpu else 'cpu'
    device = torch.device(device_name)

    SEED=0
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    np.random.seed(SEED)

    checkpoint = torch.load("D:/study/科研/torchStockNet/checkpoints/stocknet/src2_devacc0.57_all_days-5.msgs-20-words-30_word_embed-glove_batch-4.opt-adam.lr-0.01-drop-0.3-cell-gru_bestModel_ckpt.tar")
    model = CaseStudyModel()
    model.load_state_dict(checkpoint["model_sd"])

    loss_func = torch.nn.CrossEntropyLoss(reduction='sum')
    if use_gpu:
        model.to(device)
        loss_func.to(device)

    train_dataloader, dev_dataloader = make_data('test', 'test')

    epoch_loss = 0.0
    tp = fp = tn = fn = 0.0
    i = 1
    for step, (word, main_y, price, other_price, T, stock) in enumerate(dev_dataloader, start=1):
        torch.cuda.empty_cache()
        word = word.to(device)
        main_y = main_y.to(device)
        price = price.to(device)
        other_price = other_price.to(device)
        T = T.to(device)
        model.eval()
        with torch.no_grad():
            pred, max_msg_i, max_price_i = model(word, price, other_price, T)   # (batch_size, n_head)

            n_head = max_msg_i.shape[-1]
            if step % 1 == 0:
                price_attention_score = model.attention_price.attn[0].squeeze(dim=0)
                message_attention_score = model.attention_price2msg.attn
                max_word = word[0].transpose(0, 1)[max_msg_i[0]].reshape([-1, word.shape[-1]])
                max_price = other_price[0][max_price_i[0]].reshape([-1, price.shape[-1]]).transpose(0, 1).cpu().numpy()
                my_price = price[0].transpose(0, 1).cpu().numpy()
                str_list = id2word(max_word, id2word_dict)
                my_stock_name = stock_symbols[stock[0].item()]
                relevant_stock = find_stock(my_stock_name, max_price_i[0].item())
                plt.figure(figsize=(20, 20))
                plt.title('No.'+str(i)+' target stock:'+str(my_stock_name)+' relevant stock:'+str(relevant_stock), fontsize=20)
                plt.xlabel('days', fontsize=14)
                plt.ylabel('price', fontsize=14)
                day = np.arange(int(T[0].item()))+1

                plt.plot(day, my_price[0, 0:int(T[0].item())], linestyle=':', label='target price', marker='o')
                # plt.plot(day, my_price[1, 0:int(T[0].item())], linestyle='--', label='target price2', marker='+')
                # plt.plot(day, my_price[2, 0:int(T[0].item())], linestyle='-', label='target price3', marker='*')
                plt.plot(day, max_price[0, 0:int(T[0].item())], linestyle='--', label='most relevant price', marker='+')
                # plt.plot(day, max_price[1, 0:int(T[0].item())], linestyle='--', label='most relevant price2', marker='+')
                # plt.plot(day, max_price[2, 0:int(T[0].item())], linestyle='-', label='most relevant price3', marker='*')
                plt.show()

                print(i)
                print("真实:", main_y[0])
                print("预测:", pred[0])
                print("本股票名称:", my_stock_name)
                print("最相关的股票名称:", relevant_stock)
                print("最相关的文本:", str_list)
                print("本股票价格:", my_price)
                print("最相关的股票价格:", max_price)
                print('-------------------------------------------------------------------------------------------------------------------------------------------------')
                i += 1

            loss = loss_func(pred, main_y.argmax(dim=-1))

        b_tp, b_fp, b_tn, b_fn = cal_mcc_num(pred, main_y)
        tp += b_tp
        fp += b_fp
        tn += b_tn
        fn += b_fn
        del (word)
        del (main_y)
        del (price)
        del (other_price)
        epoch_loss += loss.item()

    epoch = 1
    acc = cal_acc(tp, fp, tn, fn)
    mcc = cal_mcc(tp, fp, tn, fn)
    epoch_loss /= step
    logger.info('DEV:\tepoch {}\tdev_acc:{:.2f}\tdev_mcc:{}\tdev_loss:{:.2f}'.format(epoch, acc, mcc, epoch_loss))
