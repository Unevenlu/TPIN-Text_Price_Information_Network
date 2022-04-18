import math
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import torch.nn.functional as F
from config import global_config, layer_config

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim = global_config['msg_embed_size'], hidden_dim=None, out_dim=None, n_head=layer_config['num_of_head'], dropout=global_config['att_dropout']):
        ''' Multi Head Attention Mechanism
        :param embed_dim:
        :param hidden_dim:
        :param out_dim:
        :param n_head: num of head (Multi-Head Attention)
        :param score_function: scaled_dot_product / mlp (concat)
        :return (?, q_len, out_dim,)
        '''
        super(MultiHeadAttention, self).__init__()
        if hidden_dim is None:
            hidden_dim = embed_dim // n_head
        if out_dim is None:
            out_dim = embed_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.n_head = n_head
        self.score_function = score_function
        self.dropout = nn.Dropout(dropout)
        self.linears = clones(nn.Linear(self.embed_dim, self.embed_dim), 4)

    def forward(self, q, k):
        '''
        :param q: (batch_size, embed_dim)
        :param k(v): (batch_size, sssize-1/max_n_msgs, embed_dim)
        :return: (batch_size, out_dim)
        '''
        batch_size = q.size(0)
        q, k, v = [l(x).view(batch_size, -1, self.n_head, self.hidden_dim)
                       .transpose(1, 2) for l, x in
                   zip(self.linears, (q, k, k))] # (batch_size,num_of_head,1,msg_embed_size/num_of_head)

        scores = torch.bmm(q, k.transpose(-2, -1)) / math.sqrt(self.hidden_dim)
        self.attn = F.softmax(scores, dim=-1)
        if dropout is not None:
            self.attn = dropout(self.attn)
        output = torch.bmm(self.attn, v)

        output = output.transpose(1, 2).contiguous().view(batch_size, self.n_head * self.hidden_dim)  # (batch_size, msg_embed_size)
        return self.linears[-1](output) # (batch_size,msg_embed_size)

# 这个函数是用来修剪卷积之后的数据的尺寸，让其与输入数据尺寸相同。
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class CausalConvBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout):
        super(CausalConvBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        return self.relu(out + x)

class CausalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(CausalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [CausalConvBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        #...... Previous statement

        self.hops = global_config['hops']
        self.attention_msg = MultiHeadAttention()
        self.attention_price = MultiHeadAttention()
        self.attention_price2msg = MultiHeadAttention()
        self.attention_msg2price = MultiHeadAttention()

        self.gru_cell_text = nn.GRUCell(global_config['msg_embed_size'], global_config['msg_embed_size'])
        self.gru_cell_img = nn.GRUCell(global_config['msg_embed_size'], global_config['msg_embed_size'])

    def forward():

        #...... Previous processing

        # att_rnn_price : target price (batch_size, msg_embed_size)
        # other_att_rnn_price : other price memory (batch_size, sssize-1, msg_embed_size)
        # msg_embed : message memory (batch_size, max_n_msgs, msg_embed_size)

        et_price = att_rnn_price
        et_message = att_rnn_price

        for _ in range(self.hops):
            it_al_msg2msg = self.attention_msg(rnn_msg_embed, et_message).squeeze(dim=1)
            it_al_price2msg = self.attention_price2msg(rnn_msg_embed, et_price).squeeze(dim=1)
            it_al_msg = (it_al_msg2msg + it_al_price2msg)/2

            it_al_price2price = self.attention_price(other_att_rnn_price, et_price).squeeze(dim=1)
            it_al_msg2price = self.attention_msg2price(other_att_rnn_price, et_message).squeeze(dim=1)
            it_al_price = (it_al_price2price + it_al_msg2price)/2

            et_message = self.gru_cell_text(it_al_msg, et_message)
            et_price = self.gru_cell_img(it_al_price, et_price)

        concat_embed = torch.cat((et_message, att_rnn_price, et_price), dim=-1)
        result = self.generator(concat_embed)

        return result