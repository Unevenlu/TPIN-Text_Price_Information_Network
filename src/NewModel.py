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

def creat_mask(n_msgs):
    '''
    :param n_msgs: (max_n_days, ),numpy ndarray
    :return: (max_n_days,max_n_msgs)
    '''
    n_msgs = n_msgs.astype(int)

    max_n_days = n_msgs.shape[0]
    max_n_msgs = global_config['max_n_msgs']
    mask = np.zeros([max_n_days, max_n_msgs])
    n_msgs = n_msgs.reshape([max_n_days, ])
    for i in range(max_n_days):
        mask[i, n_msgs[i]:] = 1
    mask = mask.reshape([max_n_days, max_n_msgs])
    return mask

def creat_T_mask(T):
    '''
    :param T: (batch_size, )
    :return:
    '''
    batch_size = T.shape[0]
    max_n_days = global_config['max_n_days']
    mask = torch.zeros([batch_size,max_n_days])
    for i in range(len(mask)):
        mask[i,T[i]:] = 1
    return mask

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def attention(q, k, v, mask=None, dropout=None):
    '''Compute 'Scaled Dot Product Attention' '''
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores += (mask * -1e9)
    p_attn = F.softmax(scores, dim=-1)

    if dropout is not None:
        p_attn = dropout(p_attn)

    return torch.matmul(p_attn, v), p_attn

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, hidden_dim=None, n_head=layer_config['num_of_head'], dropout=global_config['att_dropout']):
        ''' Multi Head Attention Mechanism
        :param embed_dim:
        :param hidden_dim:
        :param n_head: num of head (Multi-Head Attention)
        :return (?, q_len, out_dim,)
        '''
        super(MultiHeadAttention, self).__init__()
        if hidden_dim is None:
            hidden_dim = embed_dim // n_head
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.n_head = n_head
        self.dropout = nn.Dropout(dropout)
        self.linears = clones(nn.Linear(self.embed_dim, self.embed_dim), 4)

    def forward(self, q, k, mask=None):
        '''
        :param q: (batch_size, embed_dim)
        :param k(v): (batch_size, sssize-1/max_n_msgs, embed_dim)
        :param mask: (batch_size, max_n_days)
        :return: (batch_size, embed_dim)
        '''

        if mask is not None:  # Same mask applied to all h heads.
            mask = mask.reshape(-1, 1, 1, mask.shape[-1])  # (batch_size,1,1,max_n_days)

        batch_size = q.size(0)
        q, k, v = [l(x).view(batch_size, -1, self.n_head, self.hidden_dim)
                       .transpose(1, 2) for l, x in
                   zip(self.linears, (q, k, k))]

        output, self.attn = attention(q, k, v, mask=mask, dropout=self.dropout)
        output = output.transpose(1, 2).contiguous(). \
            view(batch_size, self.n_head * self.hidden_dim)  # (batch_size, msg_embed_size)
        return self.linears[-1](output) # (batch_size,msg_embed_size)


class WordEmbed(nn.Module):
    def __init__(self):
        super(WordEmbed, self).__init__()
        word_embed_size = global_config['word_embed_size']
        self.embed = nn.Embedding(vocab_size,word_embed_size)  # vocab_size*word_dim
        if global_config['use_in_bn']:
            self.b_n = nn.BatchNorm3d(5, affine=False)
        self.dropout = nn.Dropout(p=global_config['word_dropout'])

        if global_config['word_embed_type'] != 'rand':
            # 使用预训练好的向量
            word_embed_table = DataPipe().word_table()
            self.embed.weight.data.copy_(torch.from_numpy(word_embed_table))
            self.embed.weight.requires_grad = False

    def forward(self, x):
        '''
        :param x: (batch_size,max_n_days,max_n_msgs,max_n_words)
        :return: (batch_size,max_n_days,max_n_msgs,max_n_words,word_dim)
        '''
        x = self.embed(x)
        if global_config['use_in_bn']:
            x = self.b_n(x)
        x = self.dropout(x) #shape:(b_s,5,30,40,50)
        return x

class MsgEmbed(nn.Module):
    def __init__(self):
        # ss_index:(b_n,5,30)
        super(MsgEmbed, self).__init__()
        self.input_dim = global_config['word_embed_size']
        self.output_dim = global_config['msg_embed_size']
        self.gru = nn.GRU(input_size=self.input_dim,
                          hidden_size=self.output_dim,
                          num_layers=layer_config['msg_rnn_layers'],
                          batch_first=True,
                          bidirectional=True,
                          dropout=global_config['gru_dropout'])

        self.linear = nn.Linear(self.output_dim, self.output_dim)

    def forward(self, x, ss_index=None):
        '''
        :param x: 5d,(?,max_n_days,max_n_msgs,max_n_words,word_embed_size)
        :return:  4d,(?,max_n_days,max_n_msgs,output_dim)
        '''
        batch_size = x.size(0)
        max_n_words = global_config['max_n_words']
        max_n_days = global_config['max_n_days']

        x = x.view(-1, max_n_words, self.input_dim)  # (?*max_n_days*max_n_msgs,max_n_words,word_embed_size)
        out, h_st = self.gru(x)  #(10*2,?,100)

        '''
        out: (?,40,output_dim*num_direction),其中[?,i,0:output_dim]代表前向GRU在序列第i个输入上的结果,其中[?,i,output_dim:]代表后向GRU在序列第i个输入上的结果
        h_st: (num_layers*num_direction,?,output_dim)
        h_st transpose(?,num_layers*num_direction,output_dim),其中第二维优先按层排列，即先输出第1个layer的前后向GRU最终结果，再输入第2……，最后输出最后一层的前后向GRU最终结果（
                                                                                                                            也即是第二维的最后两个
        '''
        if ss_index != None:
            out_f = out[:, :, 0:self.output_dim]  # (?*5*30,40,100)
            out_b = out[:, :, self.output_dim:]  # (?*5*30,40,100)
            ss_index = torch.from_numpy(ss_index)  # (?,5,30)
            batch_size = ss_index.size(0)
            ss_index = ss_index.view(-1, 1, 1)  # (?*5*30,1,1)
            ss_index = ss_index.expand(-1, 1, self.output_dim)    # (?*5*30,1,100),将索引在最后一个维度升维
            ss_index = ss_index.long()
            me_f = torch.gather(input=out_f, index=ss_index, dim=1)   # (?*5*30,1,100)
            me_b = torch.gather(input=out_b, index=ss_index, dim=1)   # (?*5*30,1,100)
            # gather需要有相同的维度，其中一般把索引tensor处理成我们目标的shape，但是需要注意能否对应的上
            # me_f[i][j][k] = out_f[i][ss_index[i][j][k]][k]
            # me_b[i][j][k] = out_b[i][ss_index[i][j][k]][k]
            # out[i][j][k] = input[index[i][j][k]][j][k]   dim=0
            # out[i][j][k] = input[i][index[i][j][k]][k]   dim=1
            # out[i][j][k] = input[i][j][index[i][j][k]]   dim=2
            msg_embed = (me_f + me_b) / 2   # (?*5*30,1,100)
            # msg_embed = self.linear(msg_embed)
        else:
            h_st = torch.transpose(h_st, 0, 1)
            h_fb = h_st[:, -2:, :]  # (?,2,100)
            msg_embed = torch.mean(h_fb, dim=1)  # (?,100)

        msg_embed = msg_embed.view(batch_size, max_n_days, -1, self.output_dim)  # (?,5,30,100)
        msg_embed = self.linear(msg_embed)
        msg_embed = F.relu(msg_embed)

        return msg_embed


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


# 注意此处
class AddNorm(nn.Module):
    def __init__(self, dim=global_config['msg_embed_size']):
        super(AddNorm, self).__init__()
        self.embed_size = dim
        self.norm = LayerNorm(self.embed_size)
        self.dropout = nn.Dropout(p=global_config['add_norm_dropout'])

    def forward(self, x, func):
        y = func(x)
        y1 = self.norm(y)
        y2 = self.dropout(y1)
        return (x + y2)


class FNN(nn.Module):
    def __init__(self):
        super(FNN, self).__init__()
        self.embed_size = global_config['msg_embed_size']
        self.size = global_config['fnn_size']
        self.w_1 = nn.Linear(self.embed_size, self.size)
        self.w_2 = nn.Linear(self.size, self.embed_size)
        self.dropout = nn.Dropout(p=global_config['fnn_dropout'])

    def forward(self, x):
        '''
        :param x: 最后一个维度为msg_embed_size即可
        :return:
        '''
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class MsgMultiHeadAtt(nn.Module):
    def __init__(self):
        super(MsgMultiHeadAtt, self).__init__()
        self.msg_embed_size = global_config['msg_embed_size']
        self.h = layer_config['num_of_head']
        assert self.msg_embed_size % self.h == 0  # 如果维度数模head数不为0，返回异常
        self.d_k = self.msg_embed_size // self.h
        self.linears = clones(nn.Linear(self.msg_embed_size, self.msg_embed_size), 4)
        self.dropout = nn.Dropout(p=global_config['att_dropout'])
        self.attn = None

    def forward(self, msg_embed, mask=None):
        '''
        :param msg_embed: (batch_size,max_n_days,max_n_msgs,msg_embed_size)
        :param mask: (batch_size,max_n_days,max_n_msgs),True or False,False means no mask
        :return: x,(batch_size,max_n_days,max_n_msgs,msg_embed_size)
        '''
        max_n_msgs = global_config['max_n_msgs']
        batch_size = msg_embed.size(0)
        if mask is not None:  # Same mask applied to all h heads.
            mask = mask.reshape(-1, 1, 1, max_n_msgs)  # (batch_size*max_n_days,1,1,max_n_msgs)
        # 1) Do all the linear projections in batch from
        # d_model => h x d_k
        msg_embed = msg_embed.view(-1, max_n_msgs, self.msg_embed_size)
        n_batch_day = msg_embed.size(0)
        q, k, v = [l(x).view(n_batch_day, -1, self.h, self.d_k)
                  .transpose(1, 2) for l, x in
                  zip(self.linears, (msg_embed, msg_embed, msg_embed))]
        # zip将linears的前2个与（msg_embed,msg_embed）一一配对
        # 这里是前三个Linear Networks的具体应用，
        # 例如msg_embed=(batch_size*max_n_days, max_n_msgs, word_embed_size)
        # -> Linear network -> (batch_size*max_n_days,max_n_msgs, word_embed_size)
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
            view(batch_size, -1, max_n_msgs, self.h * self.d_k)
        # contiguous用于断开这两个变量之间的依赖（x本身是contiguous的），就要使用contiguous()针对x进行变化，感觉上就是我们认为的深拷贝
        x = self.linears[-1](x)
        return x


# 求文章内部 乘以注意力系数的信息
class MsgEncoderLayer(nn.Module):
    def __init__(self):
        super(MsgEncoderLayer, self).__init__()
        self.self_attn = MsgMultiHeadAtt()
        self.fnn = FNN()
        self.add_norm_list = clones(AddNorm(), 2)
        # 使用深度克隆方法，完整地复制出来两个add_norm

    def forward(self, msg_embed, mask=None):
        '''
        :param msg_embed: (batch_size,max_n_days,max_n_msgs,msg_embed_size)
        :param mask: (batch_size,max_n_days,max_n_msgs),True or False,False means no mask
        :return: x,(batch_size,max_n_days,max_n_msgs,msg_embed_size)
        '''
        x = msg_embed
        x = self.add_norm_list[0](x,
            lambda x: self.self_attn(x, mask))   # (batch_size,max_n_days,max_n_msgs,msg_embed_size)

        return self.add_norm_list[1](x, self.fnn)


# 文章的表示
class MsgEncoder(nn.Module):
    def __init__(self):
        super(MsgEncoder, self).__init__()
        self.dim = global_config['msg_embed_size'] * global_config['max_n_msgs']
        self.n_layers = layer_config['n_encoder_layers']
        self.msg_encoder_layers = clones(MsgEncoderLayer(), self.n_layers)

        self.linear = nn.Linear(global_config['msg_embed_size'], global_config['stock_embed_size'])

    def forward(self, msg_embed, T, mask=None):
        batch_size = msg_embed.shape[0]
        for layer in self.msg_encoder_layers:
            msg_embed = layer(msg_embed, mask)  # (batch_size, max_n_days, max_n_msgs, msg_embed_size)

        msg_embed = msg_embed.view(batch_size, -1, self.dim)
        att_rnn_msg_embed, _ = attention(q=msg_embed[range(len(msg_embed)), T-1].unsqueeze(dim=1), k=msg_embed, v=msg_embed)
        msg_embed = att_rnn_msg_embed.view(batch_size, -1, global_config['msg_embed_size'])
        return F.tanh(self.linear(msg_embed))    # (batch_size, max_n_msgs, msg_embed_size)


# 将price升维
class PriceLinear(nn.Module):
    def __init__(self):
        super(PriceLinear, self).__init__()
        self.price_size = global_config['price_size']
        self.stock_embed_size = global_config['stock_embed_size']
        self.linear = nn.Linear(self.price_size, self.stock_embed_size)
        self.dropout = nn.Dropout(p=global_config['price_dropout'])

    def forward(self, price):
        '''
        :param price: (batch_size,max_n_days,price_size)
        :return: (batch_size,max_n_days,msg_embed_size)
        '''
        price = self.linear(price)
        price = F.relu(price)
        price = self.dropout(price)
        return price


def res_add(x, func):
    y = func(x)

    return x+y


class PriceEmbed(nn.Module):
    def __init__(self):
        super(PriceEmbed, self).__init__()
        self.linear = PriceLinear()
        self.attn = MultiHeadAttention(global_config['stock_embed_size'])

    def forward(self, price, other_price, T):

        price = res_add(price, self.linear)
        other_price = res_add(other_price, self.linear)
        # other_price = self.linear(other_price)  # (batch_size, stock_size-1, max_n_days, embed_size)
        dim_size = other_price.shape[-1]
        n_days = other_price.shape[-2]
        batch_size = other_price.shape[0]
        other_price = other_price.reshape(-1, n_days, dim_size)

        T_mask = creat_T_mask(T)
        T_mask = T_mask.to(torch.device('cuda'))
        price_embed = self.attn(price[range(len(price)), T-1], price, T_mask)   # (batch_size, embed_size)
        # price_embed += price[range(len(price)), T-1]-price[range(len(price)), T-2]

        other_T = T.reshape(batch_size,1).expand(batch_size,ss_size-1).reshape([-1])
        other_T_mask = T_mask.unsqueeze(dim=1).expand(batch_size,ss_size-1,T_mask.shape[-1])
        other_price_embed = self.attn(other_price[range(len(other_price)), other_T-1], other_price, other_T_mask)
        other_price_embed += other_price[range(len(other_price)), other_T-1] - other_price[range(len(other_price)), other_T - 2]
        other_price_embed = other_price_embed.reshape(batch_size, -1, dim_size)

        return price_embed, other_price_embed


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.msg_price_concat_size = 3 * global_config['stock_embed_size']

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

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.word_embed = WordEmbed()
        self.msg_embed = MsgEmbed()
        self.msg_encoder = MsgEncoder()
        self.price_embed = PriceEmbed()
        #self.stock_price_att = MultiHeadAttention(global_config['stock_embed_size'])
        #self.stock_msg_price_att = MultiHeadAttention(global_config['stock_embed_size'])

        self.hops = 3
        self.attention_msg = MultiHeadAttention(global_config['stock_embed_size'])
        self.attention_price = MultiHeadAttention(global_config['stock_embed_size'])
        self.attention_price2msg = MultiHeadAttention(global_config['stock_embed_size'])
        self.attention_msg2price = MultiHeadAttention(global_config['stock_embed_size'])

        self.gru_cell_text = nn.GRUCell(global_config['stock_embed_size'], global_config['stock_embed_size'])
        self.gru_cell_img = nn.GRUCell(global_config['stock_embed_size'], global_config['stock_embed_size'])

        self.generator = Generator()

    def forward(self, word, price, other_price, T, mask=None, ss_index=None):
        word_embed = self.word_embed(word.long())
        msg_embed = self.msg_embed(word_embed, ss_index)
        msg_embed = self.msg_encoder(msg_embed, T, mask)   # (batch_size, max_n_msgs, embed_size)
        price_embed, other_price_embed = self.price_embed(price, other_price, T)

        #relevant_price_embed = self.stock_price_att(price_embed, other_price_embed)  # (batch_size, embed_size)
        #relevant_msg_embed = self.stock_msg_price_att(price_embed, msg_embed)  # (batch_size, embed_size)

        et_price = price_embed
        et_message = price_embed



        for _ in range(self.hops):
            it_al_msg2msg = self.attention_msg(et_message, msg_embed).squeeze(dim=1)
            it_al_price2msg = self.attention_price2msg(et_price, msg_embed).squeeze(dim=1)
            it_al_msg = (it_al_msg2msg + it_al_price2msg)/2

            it_al_price2price = self.attention_price(et_price, other_price_embed).squeeze(dim=1)
            it_al_msg2price = self.attention_msg2price(et_message, other_price_embed).squeeze(dim=1)
            it_al_price = (it_al_price2price + it_al_msg2price)/2

            et_message = self.gru_cell_text(it_al_msg, et_message)
            et_price = self.gru_cell_img(it_al_price, et_price)

        price_embed = price_embed + (price[range(len(price)), T - 1] - price[range(len(price)), T - 2])
        stock_embed = torch.cat((et_message, price_embed, et_price), dim=-1)
        prediction = self.generator(stock_embed)

        return prediction


class CaseStudyModel(nn.Module):
    def __init__(self):
        super(CaseStudyModel, self).__init__()
        self.word_embed = WordEmbed()
        self.msg_embed = MsgEmbed()
        self.msg_encoder = MsgEncoder()
        self.price_embed = PriceEmbed()
        #self.stock_price_att = MultiHeadAttention(global_config['stock_embed_size'])
        #self.stock_msg_price_att = MultiHeadAttention(global_config['stock_embed_size'])

        self.hops = 3
        self.attention_msg = MultiHeadAttention(global_config['stock_embed_size'])
        self.attention_price = MultiHeadAttention(global_config['stock_embed_size'])
        self.attention_price2msg = MultiHeadAttention(global_config['stock_embed_size'])
        self.attention_msg2price = MultiHeadAttention(global_config['stock_embed_size'])

        self.gru_cell_text = nn.GRUCell(global_config['stock_embed_size'], global_config['stock_embed_size'])
        self.gru_cell_img = nn.GRUCell(global_config['stock_embed_size'], global_config['stock_embed_size'])

        self.generator = Generator()

    def forward(self, word, price, other_price, T, mask=None, ss_index=None):
        word_embed = self.word_embed(word.long())
        msg_embed = self.msg_embed(word_embed, ss_index)
        msg_embed = self.msg_encoder(msg_embed, T, mask)   # (batch_size, max_n_msgs, embed_size)
        price_embed, other_price_embed = self.price_embed(price, other_price, T)

        #relevant_price_embed = self.stock_price_att(price_embed, other_price_embed)  # (batch_size, embed_size)
        #relevant_msg_embed = self.stock_msg_price_att(price_embed, msg_embed)  # (batch_size, embed_size)

        et_price = price_embed
        et_message = price_embed



        for _ in range(self.hops):
            it_al_msg2msg = self.attention_msg(et_message, msg_embed, mask).squeeze(dim=1)
            it_al_price2msg = self.attention_price2msg(et_price, msg_embed, mask).squeeze(dim=1)
            it_al_msg = (it_al_msg2msg + it_al_price2msg)/2

            it_al_price2price = self.attention_price(et_price, other_price_embed).squeeze(dim=1)
            it_al_msg2price = self.attention_msg2price(et_message, other_price_embed).squeeze(dim=1)
            it_al_price = (it_al_price2price + it_al_msg2price)/2

            et_message = self.gru_cell_text(it_al_msg, et_message)
            et_price = self.gru_cell_img(it_al_price, et_price)
            if _ == 0:
                message_scores = self.attention_price2msg.attn  # (batch_size,n_head,1,max_n_msgs)
                price_scores = self.attention_price.attn

        n_head = message_scores.shape[1]

        max_msg_i = torch.argmax(message_scores, dim=-1).reshape([-1, n_head])
        # 在每一个head上的每一个样本中得分最高的样本索引,(batch_size, n_head)
        max_price_i = torch.argmax(price_scores, dim=-1).reshape([-1, n_head])
        price_embed = price_embed + (price[range(len(price)), T - 1] - price[range(len(price)), T - 2])
        stock_embed = torch.cat((et_message, price_embed, et_price), dim=-1)
        prediction = self.generator(stock_embed)

        return prediction, max_msg_i, max_price_i

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


model_info = ModelInfo()
