import torch.nn as nn
import torch.nn.functional as F
import torch
from config import global_config, layer_config, ss_size, path_parser
from config import vocab_size
import math
import copy
import numpy as np
import os
from DataPipe import DataPipe


class WordEmbed(nn.Module):
    def __init__(self):
        super(WordEmbed,self).__init__()
        word_embed_table = DataPipe().word_table()
        word_embed_size = global_config['word_embed_size']
        self.embed = nn.Embedding(vocab_size,word_embed_size)  # vocab_size*word_dim
        if global_config['use_in_bn']:
            self.b_n = nn.BatchNorm3d(5, affine=False)
        self.dropout = nn.Dropout(p=global_config['word_dropout'])
        # 使用预训练好的向量
        # self.embed.weight.data.copy_(torch.from_numpy(embeding_vector))
        # self.embed.weight.requires_grad = False

        if global_config['word_embed_type'] != 'rand':
            self.embed.weight.data.copy_(torch.from_numpy(word_embed_table))
            self.embed.weight.requires_grad = False

    def forward(self,x):
        '''
        :param x: (batch_size,max_n_days,max_n_msgs,max_n_words)
        :return: (batch_size,max_n_days,max_n_msgs,max_n_words,word_dim)
        '''
        torch.cuda.empty_cache()
        x = self.embed(x)
        if global_config['use_in_bn']:
            x = self.b_n(x)
        x = self.dropout(x) #shape:(b_s,5,30,40,50)
        return x


class MsgEmbed(nn.Module):
    def __init__(self):
        # ss_index:(b_n,5,30)
        super(MsgEmbed,self).__init__()
        self.input_dim = global_config['word_embed_size']
        self.output_dim = global_config['msg_embed_size']
        self.gru = nn.GRU(input_size=self.input_dim,
                          hidden_size=self.output_dim,
                          num_layers=layer_config['msg_rnn_layers'],
                          batch_first=True,
                          bidirectional=True)
        self.ac_fun = nn.ReLU()
        if global_config['ac_func'] == 'tanh':
            self.ac_fun = nn.Tanh()
        elif global_config['ac_func'] == 'relu':
            self.ac_fun == nn.ReLU()
        elif global_config['ac_func'] == 'sigmoid':
            self.ac_fun == nn.Sigmoid()

        # self.linear = nn.Linear(input_dim=100,output_dim=100)

    def forward(self,x):
        '''
        :param x: 5d,(?,max_n_days,max_n_msgs,max_n_words,word_embed_size)
        :return:  4d,(?,max_n_days,max_n_msgs,output_dim)
        '''
        torch.cuda.empty_cache()
        batch_size = x.size(0)
        max_n_words = global_config['max_n_words']
        max_n_days = global_config['max_n_days']

        x = x.view(-1, max_n_words, self.input_dim)  # (?*max_n_days*max_n_msgs,max_n_words,word_embed_size)
        torch.cuda.empty_cache()
        out,h_st = self.gru(x)  #(10*2,?,100)

        '''
        out: (?,40,output_dim*num_direction),其中[?,i,0:output_dim]代表前向GRU在序列第i个输入上的结果,其中[?,i,output_dim:]代表后向GRU在序列第i个输入上的结果
        h_st: (num_layers*num_direction,?,output_dim)
        h_st transpose(?,num_layers*num_direction,output_dim),其中第二维优先按层排列，即先输出第1个layer的前后向GRU最终结果，再输入第2……，最后输出最后一层的前后向GRU最终结果（
                                                                                                                            也即是第二维的最后两个
        '''
        h_st = torch.transpose(h_st, 0, 1)
        h_fb = h_st[:, -2:, :]        #(?,2,100)
        msg_embed = torch.mean(h_fb, dim=1)  #(?,100)
        msg_embed = msg_embed.view(batch_size, max_n_days, -1, self.output_dim)  # (?,5,30,100)
        # msg_embed = self.linear(msg_embed)
        # msg_embed = F.tanh(msg_embed)
        msg_embed = self.ac_fun(msg_embed)
        return msg_embed


class MsgSsindexEmbed(nn.Module):
    def __init__(self):
        # ss_index:(b_n,5,30)
        super(MsgSsindexEmbed,self).__init__()
        self.input_dim = global_config['word_embed_size']
        self.output_dim = global_config['msg_embed_size']
        self.gru = nn.GRU(input_size=self.input_dim,
                          hidden_size=self.output_dim,
                          num_layers=global_config['rnn_layers'],
                          batch_first=True,
                          bidirectional=True)
        self.ac_fun = nn.ReLU()
        if global_config['ac_func'] == 'tanh':
            self.ac_fun = nn.Tanh()
        elif global_config['ac_func'] == 'relu':
            self.ac_fun == nn.ReLU()
        elif global_config['ac_func'] == 'sigmoid':
            self.ac_fun == nn.Sigmoid()
        # self.linear = nn.Linear(input_dim=100,output_dim=100)

    def forward(self,x,ss_index=None):
        '''
        :param x: 5d,(?,max_n_days,max_n_msgs,max_n_words,word_embed_size)
        :return:  4d,(?,max_n_days,max_n_msgs,output_dim)
        '''

        max_n_words = global_config['max_n_words']
        max_n_days = global_config['max_n_days']

        x = x.view(-1, max_n_words, self.input_dim)   #(?*max_n_days*max_n_msgs,max_n_words,word_embed_size)
        out,h_st = self.gru(x)  #(10*2,?,100)

        '''
        out: (?*5*30,40,output_dim*num_direction),其中[?,i,0:output_dim]代表前向GRU在序列第i个输入上的结果,其中[?,i,output_dim:]代表后向GRU在序列第i个输入上的结果
        h_st: (num_layers*num_direction,?*5*30,output_dim)
        h_st transpose(?*5*30,num_layers*num_direction,output_dim),其中第二维优先按层排列，即先输出第1个layer的前后向GRU最终结果，再输入第2……，最后输出最后一层的前后向GRU最终结果（
                                                                                                                             也即是第二维的最后两个
        '''


        if ss_index != None:
            out_f = out[:, :, 0:self.output_dim] #(?*5*30,40,100)
            out_b = out[:, :, self.output_dim:] #(?*5*30,40,100)
            ss_index = torch.from_numpy(ss_index) #(?,5,30)
            batch_size = ss_index.size(0)
            ss_index = ss_index.view(-1, 1, 1)  #(?*5*30,1,1)
            ss_index = ss_index.expand(-1, 1, self.output_dim)    #(?*5*30,1,100),将索引在最后一个维度升维
            ss_index = ss_index.long()
            me_f = torch.gather(input=out_f, index=ss_index, dim=1)   #(?*5*30,1,100)
            me_b = torch.gather(input=out_b, index=ss_index, dim=1)   #(?*5*30,1,100)
            # gather需要有相同的维度，其中一般把索引tensor处理成我们目标的shape，但是需要注意能否对应的上
            # me_f[i][j][k] = out_f[i][ss_index[i][j][k]][k]
            # me_b[i][j][k] = out_b[i][ss_index[i][j][k]][k]
            # out[i][j][k] = input[index[i][j][k]][j][k]   dim=0
            # out[i][j][k] = input[i][index[i][j][k]][k]   dim=1
            # out[i][j][k] = input[i][j][index[i][j][k]]   dim=2
            msg_embed = (me_f + me_b) / 2   #(?*5*30,1,100)
            # msg_embed = self.linear(msg_embed)
        else:
            h_st = torch.transpose(h_st, 0, 1)
            h_fb = h_st[:, -2:, :]  # (?,2,100)
            msg_embed = torch.mean(h_fb, dim=1)  # (?,100)

        msg_embed = msg_embed.view(batch_size,max_n_days,-1,self.output_dim) #(?,5,30,100)
        # msg_embed = F.tanh(msg_embed)
        msg_embed = self.ac_fun(msg_embed)
        return msg_embed


def creat_mask(n_msgs):
    '''
    :param n_msgs: (batch_size,max_n_days),numpy ndarray
    :return: (batch_size,max_n_days,max_n_msgs)
    '''
    n_msgs = n_msgs.numpy().astype(int)
    batch_size = n_msgs.shape[0]
    max_n_days = n_msgs.shape[1]
    max_n_msgs = global_config['max_n_msgs']
    mask = np.zeros([batch_size*max_n_days,max_n_msgs])
    n_msgs = n_msgs.reshape([batch_size*max_n_days,])
    for i in range(batch_size*max_n_days):
        mask[i,n_msgs[i]:] = 1
    mask = mask.reshape([batch_size,max_n_days,max_n_msgs])
    mask = torch.from_numpy(mask)
    return mask


def attention(q,k,v,mask=None,dropout=None):
    '''
    :param msg_embed(k,v): (?*max_n_days,num_of_head,max_n_msgs,msg_embed_size/num_of_head), use as K and V
    :param price_embed(q): (?*max_n_days,num_of_head,1,msg_embed_size/num_of_head)
    :param mask: (?*max_n_days,1,1,max_n_masgs),True or False,False mean no mask
    :return: (?*max_n_days,num_of_head,1,msg_embed_size/num_of_head)
    '''

    '''Compute 'Scaled Dot Product Attention' '''
    d_k = q.size(-1)  #25=d_k
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    # 然后除以sqrt(d_k)=8，防止过大的亲密度。
    if mask is not None:
        # scores = scores.masked_fill(mask == 0, -1e9)
        scores += (mask * -1e9)
        # 使用mask，对已经计算好的scores，按照mask矩阵，填-1e9，
        # 然后在下一步计算softmax的时候，被设置成-1e9的数对应的值~0,被忽视
    p_attn = F.softmax(scores, dim=-1)
    # 对scores的最后一个维度执行softmax，得到的还是一个tensor,里面存放着att的权重，或者说相似度

    if dropout is not None:
        p_attn = dropout(p_attn)  # 执行一次dropout

    return torch.matmul(p_attn, v), p_attn
    # 注意，这里返回p_attn主要是用来可视化显示多头注意力机制。


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class PriceMsgMultiHaedAtt(nn.Module):
    def __init__(self):
        super(PriceMsgMultiHaedAtt, self).__init__()
        self.msg_embed_size = global_config['msg_embed_size']
        self.h = layer_config['num_of_head']
        assert self.msg_embed_size % self.h == 0  # 如果维度数模head数不为0，返回异常
        self.d_k = self.msg_embed_size // self.h
        self.price_size = global_config['price_size']
        self.linears = clones(nn.Linear(self.msg_embed_size, self.msg_embed_size), 4)
        # 定义四个Linear networks, 每个的大小是(msg_embed_size,msg_embed_size)的，
        self.dropout = nn.Dropout(p=global_config['att_dropout'])
        # self.price_linear = nn.Linear(self.price_size,self.msg_embed_size)

    def forward(self, msg_embed, price, mask=None):
        '''
        :param msg_embed: (batch_size,max_n_days,max_n_msgs,msg_embed_size)
        :param price: (batch_size,max_n_days,msg_embed_size)
        :param mask: (batch_size,max_n_days,max_n_msgs),True or False,False means no mask
        :return: x,(batch_size,max_n_msgs,msg_embed_size)
        '''

        max_n_msgs = global_config['max_n_msgs']
        batch_size = msg_embed.size(0)
        if mask is not None:  # Same mask applied to all h heads.
            mask = mask.view(-1,1,1,max_n_msgs)    #(batch_size*max_n_days,1,1,max_n_msgs)
        # 1) Do all the linear projections in batch from
        # d_model => h x d_k
        msg_embed = msg_embed.view(-1,max_n_msgs,self.msg_embed_size)
        n_batch_day = msg_embed.size(0)
        q, k, v = [l(x).view(n_batch_day, -1, self.h, self.d_k)
                             .transpose(1, 2) for l, x in
                             zip(self.linears, (price, msg_embed, msg_embed))]
        # zip将linears的前2个与（msg_embed,msg_embed）一一配对
        # 这里是前三个Linear Networks的具体应用，
        # 例如msg_embed=(batch_size*max_n_days,max_n_msgs, word_embed_size) -> Linear network -> (batch_size*max_n_days,max_n_msgs, word_embed_size)
        # -> view -> (batch_size*max_n_days,max_n_msgs,num_of_head,msg_embed_size/num_of_head)
        # -> transpose(1,2) -> (?*max_n_days,num_of_head,max_n_msgs,msg_embed_size/num_of_head)
        # 2) Apply attention on all the projected vectors in batch.
        # view函数的作用为重构张量的维度,
        # torch.view(参数a，-1)，则表示在参数b未知，参数a已知的情况下自动补齐列向量长度

        x, self.attn = attention(q,k,v, mask=mask,dropout = self.dropout)
        # 调用上面定义好的attention函数，输出的x形状为(batch_size*max_n_days,num_of_head,max_n_msgs,msg_embed_size/num_of_head)
        # attn的形状为(batch_size*max_n_days,num_of_head,1,max_n_msgs)
        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous(). \
            view(batch_size, -1, self.h * self.d_k)
        # contiguous用于断开这两个变量之间的依赖（x本身是contiguous的），就要使用contiguous()针对x进行变化，感觉上就是我们认为的深拷贝
        x = self.linears[-1](x)
        return x


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        # features=d_model=512, eps=epsilon 用于分母的非0化平滑
        super(LayerNorm, self).__init__()
        self.w = nn.Parameter(torch.ones(features))
        # a_2 是一个可训练参数向量，(512)
        self.b = nn.Parameter(torch.zeros(features))
        # b_2 也是一个可训练参数向量, (512)
        self.eps = eps

    def forward(self, x):
        # x 的形状为(batch.size, sequence.len, embed_size)
        mean = x.mean(-1, keepdim=True)
        # 对x的最后一个维度，取平均值，得到tensor (batch.size, seq.len)
        std = x.std(-1, keepdim=True)
        # 对x的最后一个维度，取标准方差，得(batch.size, seq.len)
        return self.w * (x - mean) / (std + self.eps) + self.b
        # 本质上类似于（x-mean)/std，不过这里加入了两个可训练向量
        # a_2 and b_2，以及分母上增加一个极小值epsilon，用来防止std为0
        # 的时候的除法溢出


class AddNorm(nn.Module):
    def __init__(self):
        super(AddNorm, self).__init__()
        self.embed_size = global_config['msg_embed_size']
        self.norm = LayerNorm(self.embed_size)
        self.dropout = nn.Dropout(p=global_config['add_norm_dropout'])

    def forward(self,x):
        y = self.norm(x)
        y = self.dropout(y)
        return x + y


class FNN(nn.Module):
    def __init__(self):
        super(FNN, self).__init__()
        self.embed_size = global_config['msg_embed_size']
        self.size = global_config['fnn_size']
        self.w_1 = nn.Linear(self.embed_size,self.size)
        self.w_2 = nn.Linear(self.size, self.embed_size)
        self.dropout = nn.Dropout(p=global_config['fnn_dropout'])
        self.ac_fun = nn.ReLU()
        if global_config['ac_func'] == 'tanh':
            self.ac_fun = nn.Tanh()
        elif global_config['ac_func'] == 'relu':
            self.ac_fun == nn.ReLU()
        elif global_config['ac_func'] == 'sigmoid':
            self.ac_fun == nn.Sigmoid()

    def forward(self, x):
        '''
        :param x: 最后一个维度为msg_embed_size即可
        :return:
        '''
        x = self.w_1(x)
        x = self.ac_fun(x)
        x = self.dropout(x)
        x = self.w_2(x)
        return x


# 求与price有关系的meassage信息
class PriceMsg_Layer(nn.Module):

    def __init__(self):
        super(PriceMsg_Layer, self).__init__()
        self.att = PriceMsgMultiHaedAtt()
        self.add_norm_list = clones(AddNorm(),2)
        self.fnn = FNN()

    def forward(self,msg_embed,price,mask=None):
        '''
        :param msg_embed: (batch_size,max_n_days,max_n_msgs,msg_embed_size)
        :param price: (batch_size,max_n_days,msg_embed_size)
        :param mask: (batch_size,max_n_days,max_n_msgs),True or False,False means no mask
        :return: x,(batch_size,max_n_msgs,msg_embed_size)
        '''
        torch.cuda.empty_cache()
        x = self.att(msg_embed,price,mask)
        # x = self.add_norm_list[0](x)
        # x = self.fnn(x)
        # x = self.add_norm_list[1](x)
        return x


class MsgMultiHaedAtt(nn.Module):
    def __init__(self):
        super(MsgMultiHaedAtt, self).__init__()
        self.msg_embed_size = global_config['msg_embed_size']
        self.h = layer_config['num_of_head']
        assert self.msg_embed_size % self.h == 0  # 如果维度数模head数不为0，返回异常
        self.d_k = self.msg_embed_size // self.h
        self.linears = clones(nn.Linear(self.msg_embed_size, self.msg_embed_size), 4)
        # 定义四个Linear networks, 每个的大小是(msg_embed_size,msg_embed_size)的，
        self.dropout = nn.Dropout(p=global_config['att_dropout'])
        # self.price_linear = nn.Linear(self.price_size,self.msg_embed_size)

    def forward(self, msg_embed, mask=None):
        '''
        :param msg_embed: (batch_size,max_n_days,max_n_msgs,msg_embed_size)
        :param mask: (batch_size,max_n_days,max_n_msgs),True or False,False means no mask
        :return: x,(batch_size,max_n_days,max_n_msgs,msg_embed_size)
        '''

        max_n_msgs = global_config['max_n_msgs']
        batch_size = msg_embed.size(0)
        if mask is not None:  # Same mask applied to all h heads.
            mask = mask.view(-1, 1, 1, max_n_msgs)  # (batch_size*max_n_days,1,1,max_n_msgs)
        # 1) Do all the linear projections in batch from
        # d_model => h x d_k
        msg_embed = msg_embed.view(-1, max_n_msgs, self.msg_embed_size)
        n_batch_day = msg_embed.size(0)
        q, k, v = [l(x).view(n_batch_day, -1, self.h, self.d_k)
                    .transpose(1, 2) for l, x in
                zip(self.linears, (msg_embed, msg_embed,msg_embed))]
        # zip将linears的前2个与（msg_embed,msg_embed）一一配对
        # 这里是前三个Linear Networks的具体应用，
        # 例如msg_embed=(batch_size*max_n_days,max_n_msgs, word_embed_size) -> Linear network -> (batch_size*max_n_days,max_n_msgs, word_embed_size)
        # -> view -> (batch_size*max_n_days,max_n_msgs,num_of_head,msg_embed_size/num_of_head)
        # -> transpose(1,2) -> (?*max_n_days,num_of_head,max_n_msgs,msg_embed_size/num_of_head)
        # 2) Apply attention on all the projected vectors in batch.
        # view函数的作用为重构张量的维度,
        # torch.view(参数a，-1)，则表示在参数b未知，参数a已知的情况下自动补齐列向量长度
        x, self.attn = attention(q, k, v, mask=mask, dropout=self.dropout)
        # 调用上面定义好的attention函数，输出的x形状为(batch_size*max_n_days,num_of_head,max_n_msgs,msg_embed_size/num_of_head)
        # attn的形状为(batch_size*max_n_days,num_of_head,max_n_msgs,max_n_msgs)
        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous(). \
            view(batch_size, -1,max_n_msgs,self.h * self.d_k)
        # contiguous用于断开这两个变量之间的依赖（x本身是contiguous的），就要使用contiguous()针对x进行变化，感觉上就是我们认为的深拷贝
        x = self.linears[-1](x)
        return x


# 求文章内部 乘以注意力系数的信息
class TransformerEncoderLayer(nn.Module):
    def __init__(self):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MsgMultiHaedAtt()
        self.fnn = FNN()
        self.add_norm_list = clones(AddNorm(),2)
        # 使用深度克隆方法，完整地复制出来两个add_norm

    def forward(self, msg_embed, mask=None):
        '''
        :param msg_embed: (batch_size,max_n_days,max_n_msgs,msg_embed_size)
        :param mask: (batch_size,max_n_days,max_n_msgs),True or False,False means no mask
        :return: x,(batch_size,max_n_days,max_n_msgs,msg_embed_size)
        '''
        torch.cuda.empty_cache()
        x = self.self_attn(msg_embed,mask)   # (batch_size,max_n_days,max_n_msgs,msg_embed_size)
        x = self.add_norm_list[0](x)    # (batch_size,max_n_days,max_n_msgs,msg_embed_size)
        x = self.fnn(x)  # (batch_size,max_n_days,max_n_msgs,msg_embed_size)
        x = self.add_norm_list[1](x)    # (batch_size,max_n_days,max_n_msgs,msg_embed_size)

        return x


# 求price相关的message信息，以及message内部自相关的信息
class PriceMsgEncoderLayer(nn.Module):
    def __init__(self):
        super(PriceMsgEncoderLayer, self).__init__()
        self.price_msg_encode_layer = PriceMsg_Layer()
        self.trans_encoder = TransformerEncoderLayer()

    def forward(self,msg_embed,price,mask=None):
        '''
        :param msg_embed: (batch_size,max_n_days,max_n_msgs,msg_embed_size)
        :param price: (batch_size,max_n_days,msg_embed_size)
        :param mask: (batch_size,max_n_days,max_n_msgs),True or False,False means no mask
        :return:self_att_msg(batch_size,max_n_days,max_n_msgs,msg_embed_size)
                price_msg(batch_size,max_n_days,msg_embed_size)
        '''
        torch.cuda.empty_cache()
        price_msg = self.price_msg_encode_layer(msg_embed,price,mask)
        self_att_msg = self.trans_encoder(msg_embed,mask)

        return self_att_msg, price_msg


'''最开始的msgEncoder时这个部分，后面感觉会不会是太冗余不work，改成了下面的'''
# class PriceMsgEncoder(nn.Module):
#     def __init__(self):
#         super(PriceMsgEncoder, self).__init__()
#         self.n_layers = layer_config['n_encoder_layers']
#         self.encoder = clones(PriceMsgEncoderLayer(),self.n_layers)
#
#     def forward(self,msg_embed,price,mask=None):
#         '''
#         :param msg_embed: (batch_size,max_n_days,max_n_msgs,msg_embed_size)
#         :param price: (batch_size,max_n_days,msg_embed_size)
#         :param mask: (batch_size,max_n_days,max_n_msgs),True or False,False means no mask
#         :return:price_msg,(batch_size,max_n_days,msg_embed_size)
#         '''
#
#         for layer in self.encoder:
#             msg_embed, price = layer(msg_embed,price,mask)
#
#         price_msg = price
#         return price_msg

class PriceMsgEncoder(nn.Module):
    def __init__(self):
        super(PriceMsgEncoder, self).__init__()
        self.n_layers = layer_config['n_encoder_layers']
        self.msg_encoder = clones(TransformerEncoderLayer(), self.n_layers)
        self.price_msg_encoder = PriceMsg_Layer()

    def forward(self, msg_embed, price, mask=None):
        '''
        :param msg_embed: (batch_size,max_n_days,max_n_msgs,msg_embed_size)
        :param price: (batch_size,max_n_days,msg_embed_size)
        :param mask: (batch_size,max_n_days,max_n_msgs),True or False,False means no mask
        :return:price_msg,(batch_size,max_n_days,msg_embed_size)
        '''
        torch.cuda.empty_cache()
        for layer in self.msg_encoder:
            msg_embed = layer(msg_embed, mask)

        price_msg = self.price_msg_encoder(msg_embed, price, mask)
        return price_msg


# 将price_msg过rnn求出最后生成target day预测的向量
class PriceMsgRnn(nn.Module):
    def __init__(self):
        super(PriceMsgRnn, self).__init__()
        self.dim = global_config['msg_embed_size']
        self.rnn = nn.LSTM(input_size=self.dim,
                           hidden_size=self.dim,
                           num_layers=layer_config['msg_rnn_layers'],
                           batch_first=True)

    def forward(self,price_msg, T = None):
        '''
        :param price_msg: (batch_size,max_n_days,msg_embed_size)
        :param price_msg: (batch_size)
        :return : (batch_size, msg_embed_size)
        '''
        out, _ = self.rnn(price_msg)
        output_dim = self.dim
        if T != None:
            T = T.view(-1,1)    #(batch_size,1)
            T = torch.unsqueeze(T,dim=1)    #(batch_size,1,1)
            T = T.expand(-1,1,output_dim)
            T = T.long()
            rnn_price_msg = torch.gather(out, dim=1, index=T-1)   #(batch_size,1,msg_embed_size)
            # x[i][j][k] = out[i][T[i][j][k]][k]
        else:
            rnn_price_msg = out[:, -1]
        rnn_price_msg = rnn_price_msg.view(-1, output_dim)
        return rnn_price_msg


#
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.msg_price_concat_size = 3 * global_config['msg_embed_size']
        self.y_size = global_config['y_size']
        self.proj = nn.Linear(self.msg_price_concat_size, self.y_size)

    def forward(self, x):
        '''
        :param x: (batch_size,msg_price_concat_size)
        :param T: (batch_size)
        :return: x,(batch_size,y_size)
        '''

        x = self.proj(x)
        x = F.softmax(x, dim=-1)
        return x


# 将price升维
class PriceLinear(nn.Module):
    def __init__(self):
        super(PriceLinear, self).__init__()
        self.price_size = global_config['price_size']
        self.msg_embed_size = global_config['msg_embed_size']
        self.linear = nn.Linear(self.price_size,self.msg_embed_size)
        self.ac_fun = nn.ReLU()
        self.dropout = nn.Dropout(p=global_config['price_dropout'])
        if global_config['price_ac_func'] == 'tanh':
            self.ac_fun = nn.Tanh()
        elif global_config['price_ac_func'] == 'relu':
            self.ac_fun == nn.ReLU()
        elif global_config['price_ac_func'] == 'sigmoid':
            self.ac_fun == nn.Sigmoid()

    def forward(self, price):
        '''
        :param price: (batch_size,max_n_days,price_size)
        :return: (batch_size,max_n_days,msg_embed_size)
        '''
        torch.cuda.empty_cache()
        price = self.linear(price)
        price = self.ac_fun(price)
        price = self.dropout(price)

        return price


# 根据T生成price注意力的mask
def creat_stock_price_mask(T):
    '''
    :param T:(32,), num of trading day of each sample in batch, tensor
    :return: price_mask, (32,5)
    '''
    T = T.numpy().astype(int)
    batch_size = T.shape[0]
    max_n_days = global_config['max_n_days']
    price_mask = np.zeros([batch_size, max_n_days])
    for i in range(batch_size):
        price_mask[i, T[i]:] = 1
    price_mask = torch.from_numpy(price_mask)
    return price_mask


# 将经过rnn的price做多头注意力
class RnnPriceMultiHeadAtt(nn.Module):
    def __init__(self):
        super(RnnPriceMultiHeadAtt, self).__init__()
        self.msg_embed_size = global_config['msg_embed_size']
        self.h = layer_config['num_of_head']
        assert self.msg_embed_size % self.h == 0  # 如果维度数模head数不为0，返回异常
        self.d_k = self.msg_embed_size // self.h
        self.linears = clones(nn.Linear(self.msg_embed_size, self.msg_embed_size), 4)
        # 定义四个Linear networks, 每个的大小是(msg_embed_size,msg_embed_size)的，
        self.dropout = nn.Dropout(p=global_config['att_dropout'])

    def forward(self, h_st, hT, mask=None):
        '''
        :param h_st: composed of hi through rnn, (batch_size, max_n_days, msg_embed_size), key and value
        :param hT: last hidden state of rnn, (batch_size, msg_embed_size), query
        :param mask: (batch_size, max_n_days)
        :return: att_price:(batch_size, msg_embed_size)
        '''

        max_n_days = global_config['max_n_days']
        batch_size = h_st.size(0)
        if mask is not None:  # Same mask applied to all h heads.
            mask = mask.view(-1, 1, 1, max_n_days)  # (batch_size,1,1,max_n_days)
        # 1) Do all the linear projections in batch from
        # d_model => h x d_k
        q, k, v = [l(x).view(batch_size, -1, self.h, self.d_k)
                       .transpose(1, 2) for l, x in
                   zip(self.linears, (hT, h_st, h_st))]

        att_price, self.attn = attention(q, k, v, mask=mask, dropout=self.dropout)
        att_price = att_price.transpose(1, 2).contiguous(). \
            view(batch_size, self.h * self.d_k)  # (batch_size, msg_embed_size)
        # contiguous用于断开这两个变量之间的依赖（x本身是contiguous的），就要使用contiguous()针对x进行变化，感觉上就是我们认为的深拷贝
        att_price = self.linears[-1](att_price)
        return att_price


# 将每一个股票的price经过att-lstm
class RnnAtt(nn.Module):
    def __init__(self):
        super(RnnAtt, self).__init__()
        self.dim = global_config['msg_embed_size']
        # self.linear = PriceLinear() 应该为外层共享
        self.rnn_type = layer_config['price_rnn_cell']
        self.rnn = nn.LSTM(input_size=self.dim,
                           hidden_size=self.dim,
                           num_layers=layer_config['price_rnn_layers'],
                           batch_first=True)
        if self.rnn_type == 'gru':
            self.rnn = nn.GRU(input_size=self.dim,
                              hidden_size=self.dim,
                              num_layers=layer_config['price_rnn_layers'],
                              batch_first=True)
        self.attn = RnnPriceMultiHeadAtt()
        self.layer_norm = LayerNorm(self.dim)

    def forward(self, sotck_price, mask=None):
        '''
        :param sotck_price: price of 1 stocks,  (batch_size, max_n_days, msg_embed_size)
        :param mask: (batch_size, max_n_days)
        :return: att_rnn_price:(batch_size, msg_embed_size)
        '''
        torch.cuda.empty_cache()
        rnn_price, _ = self.rnn(sotck_price)  # (batch_size, max_n_days, msg_embed_size)
        att_rnn_price = self.attn(rnn_price, rnn_price[:, -1], mask)   # (batch_size, msg_embed_size)
        att_rnn_price = self.layer_norm(att_rnn_price)  # (batch_size, msg_embed_size)
        return att_rnn_price, rnn_price


class PriceOtherPriceMultiHeadAtt(nn.Module):
    def __init__(self):
        super(PriceOtherPriceMultiHeadAtt, self).__init__()
        self.msg_embed_size = global_config['msg_embed_size']
        self.h = layer_config['num_of_head']
        assert self.msg_embed_size % self.h == 0  # 如果维度数模head数不为0，返回异常
        self.d_k = self.msg_embed_size // self.h
        self.linears = clones(nn.Linear(self.msg_embed_size, self.msg_embed_size), 4)
        # 定义四个Linear networks, 每个的大小是(msg_embed_size,msg_embed_size)的，
        self.dropout = nn.Dropout(p=global_config['att_dropout'])

    def forward(self, price, other_price):
        '''
        :param price: (batch_size, msg_embed_size),q
        :param other_price: (batch_size, ss_size-1, msg_embed_size), k,v
        :return: (batch_size, msg_embed_size)
        '''
        torch.cuda.empty_cache()
        max_n_days = global_config['max_n_days']
        batch_size = price.size(0)
        q, k, v = [l(x).view(batch_size, -1, self.h, self.d_k)
                       .transpose(1, 2) for l, x in
                   zip(self.linears, (price, other_price, other_price))]

        price_att_other_embed, self.attn = attention(q, k, v, dropout=self.dropout)  # (batch_size,num_of_head,1,msg_embed_size/num_of_head)
        price_att_other_embed = price_att_other_embed.transpose(1, 2).contiguous(). \
            view(batch_size, self.h * self.d_k)  # (batch_size, msg_embed_size)
        # contiguous用于断开这两个变量之间的依赖（x本身是contiguous的），就要使用contiguous()针对x进行变化，感觉上就是我们认为的深拷贝
        price_att_other_embed = self.linears[-1](price_att_other_embed) # (batch_size,msg_embed_size)
        return price_att_other_embed


# 目标股票与其他股票的att-lstm-price算注意力
class PriceOtherPriceEncoderLayer(nn.Module):
    def __init__(self):
        super(PriceOtherPriceEncoderLayer, self).__init__()
        self.att = PriceOtherPriceMultiHeadAtt()
        self.add_norm_list = clones(AddNorm(), 2)
        self.fnn = FNN()
        self.otherPriceAtt = OtherPriceAtt()

    def forward(self, price, other_price):
        '''
        :param price: (batch_size, msg_embed_size),q
        :param other_price: (batch_size, ss_size-1, msg_embed_size), k,v
        :return: (batch_size, msg_embed_size)
        '''
        other_price = self.otherPriceAtt(other_price)
        x = self.att(price, other_price)
        x = self.add_norm_list[0](x)
        x = self.fnn(x)
        x = self.add_norm_list[1](x)
        return x, other_price


class OtherPriceAtt(nn.Module):
    def __init__(self):
        super(OtherPriceAtt, self).__init__()
        self.msg_embed_size = global_config['msg_embed_size']
        self.h = layer_config['num_of_head']
        assert self.msg_embed_size % self.h == 0  # 如果维度数模head数不为0，返回异常
        self.d_k = self.msg_embed_size // self.h
        self.linears = clones(nn.Linear(self.msg_embed_size, self.msg_embed_size), 4)
        # 定义四个Linear networks, 每个的大小是(msg_embed_size,msg_embed_size)的，
        self.dropout = nn.Dropout(p=global_config['att_dropout'])

    def forward(self, other_price):
        '''
        :param other_price: (batch_size, ss_size-1, msg_embed_size), k,v
        :return: (batch_size, msg_embed_size)
        '''
        torch.cuda.empty_cache()
        max_n_days = global_config['max_n_days']
        batch_size = other_price.size(0)
        q, k, v = [l(x).view(batch_size, -1, self.h, self.d_k)
                       .transpose(1, 2) for l, x in
                   zip(self.linears, (other_price, other_price, other_price))]

        att_other_price, self.attn = attention(q, k, v,
                                                     dropout=self.dropout)  # (batch_size,num_of_head,1,msg_embed_size/num_of_head)
        att_other_price = att_other_price.transpose(1, 2).contiguous(). \
            view(batch_size, -1, self.h * self.d_k)  # (batch_size, ss_size-1, msg_embed_size)
        # contiguous用于断开这两个变量之间的依赖（x本身是contiguous的），就要使用contiguous()针对x进行变化，感觉上就是我们认为的深拷贝
        att_other_price = self.linears[-1](att_other_price)  # (batch_size,msg_embed_size)
        return att_other_price


class PriceOtherPriceEncoder(nn.Module):
    def __init__(self):
        super(PriceOtherPriceEncoder, self).__init__()
        self.n_layers = layer_config['n_encoder_layers']
        self.encoder = clones(PriceOtherPriceEncoderLayer(), self.n_layers)

    def forward(self, price, other_price):
        '''
        :param price: (batch_size, msg_embed_size),q
        :param other_price: (batch_size, ss_size-1, msg_embed_size), k,v
        :return: (batch_size, msg_embed_size)
        '''
        torch.cuda.empty_cache()
        # 将这部分也修改为多次自注意加一次相互注意
        x = price
        for layer in self.encoder:
            x, other_price = layer(x, other_price)   # 此处考虑给other，price也计算一个内部self-att

        return x


# 最终的模型
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.price_linear = PriceLinear()
        self.word_embed = WordEmbed()
        self.msg_embed = MsgEmbed()
        if global_config['use_ssindex']:
            self.msg_embed = MsgSsindexEmbed()
        else:
            self.msg_embed = MsgEmbed()
        self.price_msg_encoder = PriceMsgEncoder()
        self.price_msg_rnn = PriceMsgRnn()
        self.generator = Generator()
        self.price_rnn_att = RnnAtt()
        self.other_price_rnn_atts = clones(RnnAtt(), ss_size-1)
        self.price_other_price_encoder = PriceOtherPriceEncoder()

    def forward(self, word, price, other_price, msg_mask=None, day_mask=None, ss_index=None, T=None):
        '''
        :param word:(batch_size,max_n_days,max_n_msgs,max_n_words)
        :param price:(batch_size,max_n_days,price_size)
        :param other_price:(batch_size,ss_size-1,max_n_days,price_size)
        :param msg_mask:(batch_size,max_n_days,max_n_msgs),True or False,False means no mask
        :param day_mask:(batch_size,max_n_days),True or False,False means no mask
        :param ss_index:(batch_size,max_n_days,max_n_days),the index of stock symbol in the message
        :param T:(batch_size,), the number of trading days in the sample
        :return:result,(batch_size,max_n_days,y_size)
        '''
        price = self.price_linear(price)
        other_price = self.price_linear(other_price)
        att_rnn_price, rnn_price = self.price_rnn_att(price, day_mask)
        price = rnn_price
        word_embed = self.word_embed(word)
        if global_config['use_ssindex']  and ss_index != None:
            msg_embed = self.msg_embed(word_embed, ss_index)
        else:
            msg_embed = self.msg_embed(word_embed)
        price_msg_embed = self.price_msg_encoder(msg_embed, price, msg_mask)  # (batch_size,max_n_days,msg_embed_size)
        rnn_price_msg_embed = self.price_msg_rnn(price_msg_embed, T)   # (batch_size,msg_embed_size),

        other_att_rnn_price = []
        for i in range(ss_size-1):
            stock_att_rnn_price, _ = self.other_price_rnn_atts[i](other_price[:, i], day_mask)
            other_att_rnn_price.append(stock_att_rnn_price)

        other_att_rnn_price = torch.stack(other_att_rnn_price).transpose(0, 1)  # (batch_size,ss_size-1,msg_embed_size)

        price_other_price_embed = self.price_other_price_encoder(att_rnn_price, other_att_rnn_price)    # (batch_size, msg_embed_size)

        concat_embed = torch.cat((rnn_price_msg_embed, att_rnn_price, price_other_price_embed), dim=-1)
        result = self.generator(concat_embed)

        return result


class ModelInfo:
    def __init__(self):
        # model config
        self.mode = global_config['mode']  # all
        self.opt = global_config['opt']  # adam
        self.lr = global_config['lr']  # 0.001
        self.decay_step = global_config['decay_step']  # 100
        self.decay_rate = global_config['decay_rate']  # 0.96
        self.momentum = global_config['momentum']  # 0.9
        self.max_n_days = global_config['max_n_days']
        self.max_n_msgs = global_config['max_n_msgs']
        self.max_n_words = global_config['max_n_words']
        self.word_embed_type = global_config['word_embed_type']
        self.batch_size_for_name = global_config['batch_size']
        self.opt = global_config['opt']
        self.lr = global_config['lr']
        self.dropout_train_in = global_config['word_dropout']
        self.rnn_cell_type = layer_config['msg_rnn_cell']
        # model name
        name_pattern_max_n = 'days-{0}.msgs-{1}-words-{2}'  # days-{0}.msgs-{1}-words-{2}
        name_max_n = name_pattern_max_n.format(self.max_n_days, self.max_n_msgs,
                                               self.max_n_words)  # days-5.msgs-30-words-40

        name_pattern_input_type = 'word_embed-{0}'  # ‘word_embed-{0}’
        name_input_type = name_pattern_input_type.format(self.word_embed_type,)  # 'word_embed-glove'

        name_pattern_train = 'batch-{0}.opt-{1}.lr-{2}-drop-{3}-cell-{4}'
        name_train = name_pattern_train.format(self.batch_size_for_name, self.opt, self.lr, self.dropout_train_in,
                                               self.rnn_cell_type)
        # 'batch-32.opt-adam.lr-0.001-drop-0.3-cell-gru'

        name_tuple = (self.mode, name_max_n, name_input_type, name_train)
        self.model_name = '_'.join(
            name_tuple)  # 'all_days-5.msgs-30-words-40_word_embed-glove_batch-32.opt-adam.lr-0.001-drop-0.3-cell-gru'


class Model2(nn.Module):
    def __init__(self):
        super(Model2, self).__init__()
        self.price_linear = PriceLinear()
        self.word_embed = WordEmbed()
        self.msg_embed = MsgEmbed()
        if global_config['use_ssindex']:
            self.msg_embed = MsgSsindexEmbed()
        self.price_msg_encoder = PriceMsgEncoder()
        self.msg_rnn_att = RnnAtt()
        self.generator = Generator()
        self.price_rnn_att = RnnAtt()
        self.other_price_rnn_atts = clones(RnnAtt(), ss_size - 1)
        self.price_other_price_encoder = PriceOtherPriceEncoder()
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    def forward(self, word, price, other_price):
        torch.cuda.empty_cache()
        word_embed = self.word_embed(word)
        msg_embed = self.msg_embed(word_embed)
        price_embed = self.price_linear(price)
        other_price_embed = self.price_linear(other_price)  # (32,100)

        price_att_msg_embed = self.price_msg_encoder(msg_embed, price_embed)    # (32,5,100)
        att_rnn_price_embed, _ = self.price_rnn_att(price_embed)    # (32,100)

        att_rnn_other_price_embed = torch.zeros([other_price.shape[0], ss_size-1, msg_embed.shape[-1]],
                                                dtype=torch.float32).to(self.device)    # (32,87,100)

        for i in range(ss_size-1):
            att_rnn_other_price_embed[:, i], _ = self.other_price_rnn_atts[i](other_price_embed[:, i])

        price_att_other_price_embed = self.price_other_price_encoder(att_rnn_price_embed, att_rnn_other_price_embed)    # (32,100)
        rnn_att_price_msg_embde, _ = self.msg_rnn_att(price_att_msg_embed)

        stock_embed = torch.cat([att_rnn_price_embed, price_att_other_price_embed, rnn_att_price_msg_embde], dim=-1)
        prediction = self.generator(stock_embed)
        return prediction


model_info = ModelInfo()