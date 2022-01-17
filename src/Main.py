import os.path
import copy
import torch
from Model import Model2, model_info
from config import global_config, path_parser
from DataLoader import make_data
import time
from config import logger
import torch.nn as nn
from metrics import cal_mcc_num, cal_acc, cal_mcc

if __name__ == '__main__':
    use_gpu = torch.cuda.is_available()
    print(use_gpu)
    device_name = 'cuda' if use_gpu else 'cpu'
    device = torch.device(device_name)
    model = Model2()
    for p in model.parameters():
        if p.dim() > 1 and p.requires_grad == True:
            nn.init.xavier_uniform_(p)
    logger.info('INIT: Build Model, Model Name: ' + model_info.model_name)

    optimizer = torch.optim.Adam(model.parameters(), lr=global_config['lr'])
    loss_func = torch.nn.CrossEntropyLoss(reduction='sum')
    if use_gpu:
        torch.cuda.empty_cache()
        model.to(device)
        loss_func.to(device)

    train_dataloader, dev_dataloader = make_data('dev', 'dev')

    epochs = global_config['n_epochs']
    start_time = time.time()
    best_acc = 0.
    for epoch in range(1, epochs+1):
        logger.info('Epoch: {0}/{1} start'.format(epoch, epochs))
        epoch_loss = 0
        tp = fp = tn = fn = 0.
        torch.cuda.empty_cache()
        for step, (word, main_y, price, other_price) in enumerate(train_dataloader, start=1):
            word = word.to(device)
            main_y = main_y.to(device)
            price = price.to(device)
            other_price = other_price.to(device)
            model.train()
            torch.autograd.set_detect_anomaly(True)
            pred = model(word, price, other_price)
            b_tp, b_fp, b_tn, b_fn = cal_mcc_num(pred, main_y)
            tp += b_tp
            fp += b_fp
            tn += b_tn
            fn += b_fn
            model.zero_grad()
            loss = loss_func(pred, main_y.argmax(dim=-1))
            del(word)
            del(main_y)
            del(price)
            del(other_price)

            with torch.autograd.detect_anomaly():
                loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        acc = cal_acc(tp, fp, tn, fn)
        mcc = cal_mcc(tp, fp, tn, fn)
        epoch_loss /= step
        logger.info(
            'TRAIN:\tepoch {}\ttrain_acc:{:.2f}\ttrain_mcc:{}\ttrain_loss:{:.2f}'.format(epoch, acc, mcc, epoch_loss))
        logger.info(
            'TRAIN:\ttp:{:.0f}\tfp:{:.0f}\ttn:{:.0f}\tfn:{:.0f}'.format(tp, fp, tn, fn))

        epoch_loss = 0.0
        tp = fp = tn = fn = 0.0
        for step, (word, main_y, price, other_price) in enumerate(dev_dataloader, start=1):
            torch.cuda.empty_cache()
            word = word.to(device)
            main_y = main_y.to(device)
            price = price.to(device)
            other_price = other_price.to(device)
            model.eval()
            with torch.no_grad():
                pred = model(word, price, other_price)
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

        acc = cal_acc(tp, fp, tn, fn)
        mcc = cal_mcc(tp, fp, tn, fn)
        epoch_loss /= step
        logger.info('DEV:\tepoch {}\tdev_acc:{:.2f}\tdev_mcc:{}\tdev_loss:{:.2f}'.format(epoch, acc, mcc, epoch_loss))

        if acc > best_acc:
            best_acc = acc
            checkpoint = os.path.join(path_parser.checkpoints, '{}_bestModel_ckpt.tar'.format(model_info.model_name))
            model_sd = copy.deepcopy(model.state_dict())
            torch.save({
                'loss': loss,
                'epoch': epoch,
                'model_sd': model_sd,
                'acc': acc,
                'mcc': mcc,
                'opt_sd': optimizer.state_dict(),
                'optimizer': optimizer.state_dict()
            }, checkpoint)
            logger.info('best model saved!epoch={}\tacc={:.2f}\tmcc={}'.format(epoch, acc, mcc))

        checkpoint = os.path.join(path_parser.checkpoints, '{}_newModel_ckpt.tar'.format(model_info.model_name))
        model_sd = copy.deepcopy(model.state_dict())
        torch.save({
            'loss': loss,
            'epoch': epoch,
            'model_sd': model_sd,
            'acc': acc,
            'mcc': mcc,
            'opt_sd': optimizer.state_dict(),
            'optimizer': optimizer.state_dict()
        }, checkpoint)

        logger.info('\n')

    logger.info('finishing training...')
    end_time = time.time()
    time_elapsed = end_time - start_time
    logger.info('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    
