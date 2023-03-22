# Code reused from https://github.com/arghosh/AKT and https://github.com/tianlinyang/DKVMN
import torch
from torch import nn
from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
from torch.nn.init import xavier_normal_
from memory import DKVMN
import utils as utils
import math
import torch.nn.functional as F
from enum import IntEnum
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Dim(IntEnum):
    batch = 0
    seq = 1
    feature = 2

class MAN(nn.Module):
    def __init__(self, n_skill, n_exercise, d_model, n_blocks, kq_same, dropout, model_type, memory_size,
                batch_size, seqlen, final_fc_dim=512, n_heads=8, d_ff=2048, l2=1e-5, epsilon=1e-5, separate_qa=False):
        super().__init__()
        self.n_skill = n_skill
        self.dropout = dropout
        self.kq_same = kq_same
        self.n_exercise = n_exercise
        self.l2 = l2
        self.model_type = model_type
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.seqlen = seqlen
        self.final_fc_dim = final_fc_dim
        self.separate_qa = separate_qa
        self.epsilon = epsilon
        self.log_vars = nn.Parameter(torch.zeros((2)))
        embed_l = d_model
        concat_embed_l = embed_l

        if self.n_exercise > 0:
            self.s_embed_diff = nn.Embedding(self.n_exercise+ 1, embed_l)
            self.sa_embed_diff = nn.Embedding(2 * self.n_exercise + 1, embed_l)
            concat_embed_l += embed_l

        self.s_embed = nn.Embedding(self.n_skill+1, embed_l)
        if self.separate_qa:
            self.sa_embed = nn.Embedding(2*self.n_skill+1, embed_l)
        else:
            self.sa_embed = nn.Embedding(2, embed_l)
        
        d_model = concat_embed_l 
        self.s_w1_param = nn.Embedding(self.n_skill+3, d_model)
        self.s_w2_param = nn.Embedding(self.n_skill+3, d_model)

        self.e_spare_param = nn.Embedding(self.n_exercise+2, 1)
        self.s_personalized_param = nn.Embedding(self.n_skill+2, 1)
        self.x_softmax = nn.Sequential(nn.Linear(concat_embed_l, concat_embed_l), nn.Softmax(1), nn.Dropout(self.dropout))
        self.y_softmax = nn.Sequential(nn.Linear(concat_embed_l, concat_embed_l), nn.Softmax(1), nn.Dropout(self.dropout))
        
        # Architecture Object. It contains stack of attention block
        self.model = Architecture(n_skill=n_skill, n_blocks=n_blocks, n_heads=n_heads, dropout=dropout, d_model=d_model, 
                                d_feature=d_model / n_heads, d_ff=d_ff, kq_same=self.kq_same, model_type=self.model_type)

        self.init_memory_key = nn.Parameter(torch.randn(self.memory_size, d_model))
        nn.init.kaiming_normal_(self.init_memory_key)
        self.init_memory_value = nn.Parameter(torch.randn(self.memory_size, d_model))
        nn.init.kaiming_normal_(self.init_memory_value)

        self.mem = DKVMN(memory_size=self.memory_size,
                   memory_key_state_dim=d_model,
                   memory_value_state_dim=d_model, init_memory_key=self.init_memory_key)

        memory_value = nn.Parameter(torch.cat([self.init_memory_value.unsqueeze(0) for _ in range(self.batch_size)], 0).data)
        self.mem.init_value_memory(memory_value)

        self.weight = nn.Sequential(
            nn.Linear(d_model + d_model,d_model + d_model), nn.Softmax(1), nn.Dropout(self.dropout)
        )

        self.out = nn.Sequential(
            nn.Linear(d_model + d_model + d_model,
                      final_fc_dim), nn.ReLU(), nn.Dropout(self.dropout),
            nn.Linear(final_fc_dim, 256), nn.ReLU(), nn.Dropout(self.dropout),
            nn.Linear(256, 1)
        )

        self.weight_1 = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.weight_2 = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.weight_1.data.fill_(0.8)
        self.weight_2.data.fill_(1.0)

        self.rho = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.rho.data.fill_(0.)

        self.reset()

    def reset(self):
        for p in self.parameters():
            if p.size(0) == self.n_exercise+2 and self.n_exercise > 0:
                torch.nn.init.constant_(p, 0.8)
            if p.size(0) == self.n_skill+2 and self.n_skill > 0:
                torch.nn.init.constant_(p, 0.7)
            if p.size(0) == self.n_skill+3 and self.n_skill > 0:
                torch.nn.init.constant_(p, 0.0)

    def forward(self, s_data, sa_data, target, e_data=None):
        # Batch First
        batch_size = s_data.shape[0]
        s_embed_data = self.s_embed(s_data)

        a_data = (sa_data-s_data)//self.n_skill
        sa_embed_data = self.sa_embed(a_data)+s_embed_data

        x_embed = s_embed_data
        y_embed = sa_embed_data

        if self.n_exercise > 0:
            e_spare_weight = self.e_spare_param(e_data)
            s_embed_diff_data = self.s_embed_diff(e_data)
            x_embed = torch.cat([s_embed_data, e_spare_weight*s_embed_diff_data], dim=-1)
            sa_embed_diff_data = self.sa_embed_diff(a_data)
            y_embed = torch.cat([sa_embed_data, e_spare_weight*(sa_embed_diff_data+s_embed_diff_data)], dim=-1)

        x_embed = self.x_softmax(x_embed)
        y_embed = self.y_softmax(y_embed)

        d_output, y_hat_embed, x_context_embed = self.model(x_embed, y_embed)

        s_personalized_weight = self.s_personalized_param(s_data)
        s_weight1 = self.s_w1_param(s_data)
        s_weight2 = self.s_w2_param(s_data)
        
        a = x_embed.transpose(1, 2)
        b = x_context_embed.transpose(1, 2)
        scores = torch.matmul(a, b.transpose(-2, -1))
        a1 = s_weight1.transpose(1, 2)
        a2 = s_weight2.transpose(1, 2)
        bias = torch.matmul(a1, a2.transpose(-2, -1))
        
        slice_s_embed_data = torch.chunk(x_embed, self.seqlen, 1)
        slice_sa_embed_data = torch.chunk(y_hat_embed, self.seqlen, 1)
        slice_w_embed_data = torch.chunk(s_personalized_weight, self.seqlen, 1)
        value_read_content_l = []
        input_embed_l = []
        for i in range(self.seqlen):
            ## Attention
            s = slice_s_embed_data[i].squeeze(1)
            correlation_weight = self.mem.attention(s)
            if_memory_write = slice_s_embed_data[i].squeeze(1).ge(1)
            if_memory_write = utils.varible(torch.FloatTensor(if_memory_write.data.tolist()), 1)

            ## Read Process
            read_content = self.mem.read(correlation_weight)
            value_read_content_l.append(read_content)
            input_embed_l.append(s)
            ## Write Process
            sa = slice_sa_embed_data[i].squeeze(1)
            write_weight = slice_w_embed_data[i].squeeze(1)
            new_memory_value = self.mem.write(correlation_weight, sa, if_memory_write, write_weight)

        all_read_value_content = torch.cat([value_read_content_l[i].unsqueeze(1) for i in range(self.seqlen)], 1)
        h1 = torch.matmul(scores+self.rho, d_output.transpose(1, 2))
        h2 = torch.matmul(1-scores-self.rho, all_read_value_content.transpose(1, 2))
        h = self.weight(torch.cat([h1.transpose(2, 1), h2.transpose(2, 1)], dim=-1))
        concat_c = torch.cat([h, x_embed*s_personalized_weight], dim=-1)
        output = self.out(concat_c)

        labels = target.reshape(-1)

        m = nn.Sigmoid()
        preds = (output.reshape(-1))
        mask = labels > -0.9
        masked_labels = labels[mask].float()
        masked_preds = preds[mask]
        loss = nn.BCEWithLogitsLoss(reduction='none')
        output = loss(masked_preds, masked_labels)
        return output.sum(), m(preds), mask.sum()


class Architecture(nn.Module):
    def __init__(self, n_skill,  n_blocks, d_model, d_feature,
                 d_ff, n_heads, dropout, kq_same, model_type):
        super().__init__()
        self.d_model = d_model
        self.model_type = model_type

        if model_type in {'man'}:
            self.blocks_1 = nn.ModuleList([
                TransformerLayer(d_model=d_model, d_feature=d_model // n_heads,
                                 d_ff=d_ff, dropout=dropout, n_heads=n_heads, kq_same=kq_same)
                for _ in range(n_blocks)
            ])
            self.blocks_2 = nn.ModuleList([
                TransformerLayer(d_model=d_model, d_feature=d_model // n_heads,
                                 d_ff=d_ff, dropout=dropout, n_heads=n_heads, kq_same=kq_same)
                for _ in range(n_blocks)
            ])
            self.blocks_3 = nn.ModuleList([
                TransformerLayer(d_model=d_model, d_feature=d_model // n_heads,
                                 d_ff=d_ff, dropout=dropout, n_heads=n_heads, kq_same=kq_same)
                for _ in range(n_blocks*3)
            ])

    def forward(self, s_embed_data, sa_embed_data):
        seqlen, batch_size = s_embed_data.size(1), s_embed_data.size(0)

        sa_pos_embed = sa_embed_data
        s_pos_embed = s_embed_data

        y = sa_pos_embed
        seqlen, batch_size = y.size(1), y.size(0)
        x = s_pos_embed
        x1 = s_pos_embed

        # encoder
        for block in self.blocks_1:  # encode qas
            y = block(mask=1, query=y, key=y, values=y)
        for block in self.blocks_2:  # encode qas
            x1 = block(mask=0, query=x1, key=x1, values=x1)
        flag_first = True
        for block in self.blocks_3:
            if flag_first:
                x = block(mask=1, query=x, key=x,
                          values=x, apply_pos=False)
                flag_first = False
            else:
                x = block(mask=0, query=x, key=x, values=y, apply_pos=True)
                flag_first = True
            y = block(mask=1, query=x, key=x, values=y, apply_pos=True)
        return x,y,x1


class TransformerLayer(nn.Module):
    def __init__(self, d_model, d_feature,
                 d_ff, n_heads, dropout,  kq_same):
        super().__init__()
        """
            This is a Basic Block of Transformer paper. It containts one Multi-head attention object. Followed by layer norm and postion wise feedforward net and dropout layer.
        """
        kq_same = kq_same == 1
        # Multi-Head Attention Block
        self.masked_attn_head = MultiHeadAttention(
            d_model, d_feature, n_heads, dropout, kq_same=kq_same)

        # Two layer norm layer and two droput layer
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, mask, query, key, values, apply_pos=True):
        seqlen, batch_size = query.size(1), query.size(0)
        nopeek_mask = np.triu(
            np.ones((1, 1, seqlen, seqlen)), k=mask).astype('uint8')
        src_mask = (torch.from_numpy(nopeek_mask) == 0).to(device)
        if mask == 0:  # If 0, zero-padding is needed.
            # Calls block.masked_attn_head.forward() method
            query2 = self.masked_attn_head(
                query, key, values, mask=src_mask, zero_pad=True)
        else:
            # Calls block.masked_attn_head.forward() method
            query2 = self.masked_attn_head(
                query, key, values, mask=src_mask, zero_pad=False)

        query = query + self.dropout1((query2))
        query = self.layer_norm1(query)
        if apply_pos:
            query2 = self.linear2(self.dropout(
                self.activation(self.linear1(query))))
            query = query + self.dropout2((query2))
            query = self.layer_norm2(query)
        return query


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_feature, n_heads, dropout, kq_same, bias=True):
        super().__init__()
        """
        It has projection layer for getting keys, queries and values. Followed by attention and a connected layer.
        """
        self.d_model = d_model
        self.d_k = d_feature
        self.h = n_heads
        self.kq_same = kq_same

        self.v_linear = nn.Linear(d_model, d_model, bias=bias)
        self.k_linear = nn.Linear(d_model, d_model, bias=bias)
        if kq_same is False:
            self.q_linear = nn.Linear(d_model, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.proj_bias = bias
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        self.gammas = nn.Parameter(torch.zeros(n_heads, 1, 1))
        torch.nn.init.xavier_uniform_(self.gammas)

        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.k_linear.weight)
        xavier_uniform_(self.v_linear.weight)
        if self.kq_same is False:
            xavier_uniform_(self.q_linear.weight)

        if self.proj_bias:
            constant_(self.k_linear.bias, 0.)
            constant_(self.v_linear.bias, 0.)
            if self.kq_same is False:
                constant_(self.q_linear.bias, 0.)
            constant_(self.out_proj.bias, 0.)

    def forward(self, q, k, v, mask, zero_pad):

        bs = q.size(0)

        # perform linear operation and split into h heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        if self.kq_same is False:
            q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        else:
            q = self.k_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * h * sl * d_model
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        # calculate attention using function we will define next
        gammas = self.gammas
        scores = attention(q, k, v, self.d_k,
                           mask, self.dropout, zero_pad, gammas)

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous()\
            .view(bs, -1, self.d_model)

        output = self.out_proj(concat)

        return output


def attention(q, k, v, d_k, mask, dropout, zero_pad, gamma=None):
    """
    This is called by Multi-head atention object to find the values.
    """
    scores = torch.matmul(q, k.transpose(-2, -1)) / \
             math.sqrt(d_k)
    bs, head, seqlen = scores.size(0), scores.size(1), scores.size(2)

    x1 = torch.arange(seqlen).expand(seqlen, -1).to(device)
    x2 = x1.transpose(0, 1).contiguous()

    with torch.no_grad():
        scores_ = scores.masked_fill(mask == 0, -1e32)
        scores_ = F.softmax(scores, dim=-1)
        scores_ = scores_ * mask.float().to(device)
        distcum_scores = torch.cumsum(scores_, dim=-1)
        disttotal_scores = torch.sum(
            scores_, dim=-1, keepdim=True) 
        position_effect = torch.abs(
            x1-x2)[None, None, :, :].type(torch.FloatTensor).to(device)
        dist_scores = torch.clamp(
            (disttotal_scores-distcum_scores)*position_effect, min=0.)
        dist_scores = dist_scores.sqrt().detach()
    m = nn.Softplus()
    gamma = -1. * m(gamma).unsqueeze(0)
    # Now after do exp(gamma*distance) and then clamp to 1e-5 to 1e5
    total_effect = torch.clamp(torch.clamp(
        (dist_scores*gamma).exp(), min=1e-5), max=1e5)
    scores = scores * total_effect

    scores.masked_fill_(mask == 0, -1e32)
    scores = F.softmax(scores, dim=-1)
    if zero_pad:
        pad_zero = torch.zeros(bs, head, 1, seqlen).to(device)
        scores = torch.cat([pad_zero, scores[:, :, 1:, :]], dim=2)
    scores = dropout(scores)
    output = torch.matmul(scores, v)
    return output


class LearnablePositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        # Compute the positional encodings once in log space.
        pe = 0.1 * torch.randn(max_len, d_model)
        pe = pe.unsqueeze(0)
        self.weight = nn.Parameter(pe, requires_grad=True)

    def forward(self, x):
        return self.weight[:, :x.size(Dim.seq), :]


class CosinePositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        # Compute the positional encodings once in log space.
        pe = 0.1 * torch.randn(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.weight = nn.Parameter(pe, requires_grad=False)

    def forward(self, x):
        return self.weight[:, :x.size(Dim.seq), :]
