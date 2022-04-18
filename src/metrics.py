import torch
import numpy as np
import math
from config import global_config


class Loss:
    def __init__(self):
        self.compute_loss = torch.nn.CrossEntropyLoss()

    def loss_func(self, pred, real):
        '''
        对每一个batch计算loss
        :param pred: (batch_size,y_size)
        :param real: (batch_size,y_size)
        :return:scalar
        '''
        # pred = torch.argmax(pred,dim=1) #(batch_size,1)
        real = torch.argmax(real,dim=1).long()   #(batch_size)
        _loss = self.compute_loss(pred, real)  # [batch_size]

        return _loss


class CustomSchedule(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, d_model=global_config['msg_embed_size'], warm_steps=global_config['warm_step']):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warm_steps

        super(CustomSchedule, self).__init__(optimizer)

    def get_lr(self):
        """
        # rsqrt 函数用于计算 x 元素的平方根的倒数.  即= 1 / sqrt{x}
        arg1 = torch.rsqrt(torch.tensor(self._step_count, dtype=torch.float32))
        arg2 = torch.tensor(self._step_count * (self.warmup_steps ** -1.5), dtype=torch.float32)
        dynamic_lr = torch.rsqrt(self.d_model) * torch.minimum(arg1, arg2)
        """
        # print('*'*27, self._step_count)
        arg1 = self._step_count ** (-0.5)
        arg2 = self._step_count * (self.warmup_steps ** -1.5)
        dynamic_lr = (self.d_model ** (-0.5)) * min(arg1, arg2)
        # print('dynamic_lr:', dynamic_lr)
        return [dynamic_lr for group in self.optimizer.param_groups]


def cal_mcc_num(pred, target, is_distribution=True):
    '''
    :param pred: (batch_size,2),tensor
    :param target: (batch_size,2),tensor
    :return: scalar
    '''
    y = pred
    y_ = target  # (batch_size,y_size),tensor
    n_samples = float(y_.shape[0])
    if is_distribution:
        label_ref = torch.argmax(y_, dim=-1)  # 1-d array of 0 and 1,(batch_size,)
        label_hyp = torch.argmax(y, dim=-1)
    else:
        label_ref, label_hyp = y, y_

    # p & n in prediction
    p_in_hyp = torch.sum(label_hyp)    # 预测中的正类个数
    n_in_hyp = n_samples - p_in_hyp     # 预测中的负类个数

    # Positive class: up
    tp = torch.sum(label_ref * label_hyp, dim=-1)
    # np.multiply,将对应位置相乘，若对应位置都为1，乘出来结果为1，则认为是真正类
    # 再用np.sum,求和求出tp真正例的个数

    fp = p_in_hyp - tp  # predicted positive, but false

    # Negative class: down
    tn = n_samples - torch.count_nonzero(label_ref + label_hyp)  # both 0 can remain 0
    fn = n_in_hyp - tn  # predicted negative, but false

    return float(tp.item()), float(fp.item()), float(tn.item()), float(fn.item())


def cal_mcc(tp, fp, tn, fn):
    core_de = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    if core_de:
        mcc = (tp * tn - fp * fn) / math.sqrt(core_de)
    else:
        mcc = None
    return mcc


def cal_acc(tp, fp, tn, fn):
    acc = (tp + tn)/(tp +fp +tn + fn)
    return acc


class Regularization(torch.nn.Module):
    def __init__(self, model, weight_decay, p=2):
        '''
        :param model 模型
        :param weight_decay:正则化参数
        :param p: 范数计算中的幂指数值，默认求2范数,
                  当p=2为L2正则化,p=1为L1正则化
        '''
        super(Regularization, self).__init__()
        if weight_decay <= 0:
            print("param weight_decay can not <=0")
            exit(0)
        self.model = model
        self.weight_decay = weight_decay
        self.p = p
        self.weight_list = self.get_weight(model)

    def to(self, device):
        '''
        指定运行模式
        :param device: cude or cpu
        :return:
        '''
        self.device = device
        super().to(device)
        return self

    def forward(self, model):
        self.weight_list = self.get_weight(model)  # 获得最新的权重
        reg_loss = self.regularization_loss(self.weight_list, self.weight_decay, p=self.p)
        return reg_loss

    def get_weight(self, model):
        '''
        获得模型的权重列表
        :param model:
        :return:
        '''
        weight_list = []
        for name, param in model.named_parameters():
            if 'weight' in name:
                weight = (name, param)
                weight_list.append(weight)
        return weight_list

    def regularization_loss(self, weight_list, weight_decay, p=2):
        '''
        计算张量范数
        :param weight_list:
        :param p: 范数计算中的幂指数值，默认求2范数
        :param weight_decay:
        :return:
        '''
        # weight_decay=Variable(torch.FloatTensor([weight_decay]).to(self.device),requires_grad=True)
        # reg_loss=Variable(torch.FloatTensor([0.]).to(self.device),requires_grad=True)
        # weight_decay=torch.FloatTensor([weight_decay]).to(self.device)
        # reg_loss=torch.FloatTensor([0.]).to(self.device)
        reg_loss = 0
        for name, w in weight_list:
            l2_reg = torch.norm(w, p=p)
            reg_loss = reg_loss + l2_reg

        reg_loss = weight_decay * reg_loss
        return reg_loss

    def weight_info(self, weight_list):
        '''
        打印权重列表信息
        :param weight_list:
        :return:
        '''
        print("---------------regularization weight---------------")
        for name, w in weight_list:
            print(name)
        print("---------------------------------------------------")