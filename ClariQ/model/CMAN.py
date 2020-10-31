import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math, copy
import numpy as np
from torch.autograd import Variable


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask=None):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        # return self.norm(x).view(x.size(0),-1)
        return self.norm(x)


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask=None):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        # assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(gelu(self.w_1(x))))


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab, pretrained=None):
        super(Embeddings, self).__init__()
        if pretrained is None:
            self.lut = nn.Embedding(vocab, d_model)
        else:
            self.lut = nn.Embedding(vocab, d_model).from_pretrained(pretrained)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0.0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0.0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)


def make_model(N, d_model, h, d_ff, seq_len, vocab_size, pretrained=None, dropout=0.1):
    '''
    N: number of stack
    d_model: d_model
    h: head
    d_ff: inner hidden layer
    input_size: this is for final DNN
    output_size: this is for final DNN
    '''
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    FFN = PositionwiseFeedForward(d_model, d_ff)
    enc = EncoderLayer(d_model, c(attn), c(FFN), dropout)
    final_encoder = Encoder(enc, N)
    word_embedding = Embeddings(d_model, vocab_size,pretrained)
    pos_emb = PositionalEncoding(d_model, dropout)

    final_model = nn.Sequential(
        final_encoder
    )

    for p in final_model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return final_model, word_embedding, pos_emb


class MwAN_trans(nn.Module):
    def __init__(self, d_model, number_block, head_number, d_ff, seq_len, vocab_size, pretrained=None, drop_out=0.3,
                 emb_dropout=0.1, BINs=99):
        super(MwAN_trans, self).__init__()
        self.dropout = nn.Dropout(drop_out)
        self.word_dropout = nn.Dropout(emb_dropout)
        # self.p_encoder, p_word_embedding, p_pos_emb = make_model(N=number_block, d_model=d_model,h=head_number,d_ff=d_ff, seq_len=seq_len, vocab_size=vocab_size)
        # self.c_encoder, c_word_embedding, c_pos_emb = make_model(N=number_block, d_model=d_model,h=head_number,d_ff=d_ff, seq_len=seq_len, vocab_size=vocab_size)

        self.encoder, self.word_embedding, self.pos_emb = make_model(N=number_block, d_model=d_model, h=head_number,
                                                                     d_ff=d_ff, seq_len=seq_len, pretrained=pretrained,vocab_size=vocab_size)
        # Concat Attention
        # self.encoder_2, _, _ = make_model(N=1, d_model=3 * d_model, h=head_number, d_ff=3 * d_ff, seq_len=seq_len,
        #                                   vocab_size=vocab_size)

        self.encoder_2, _, _ = make_model(N=1, d_model=4 * d_model, h=head_number, d_ff=4 * d_ff, seq_len=seq_len,
                                          vocab_size=vocab_size)

        ## for hp
        self.Wc1 = nn.Linear(d_model, d_model, bias=False)
        self.Wc2 = nn.Linear(d_model, d_model, bias=False)
        self.vc = nn.Linear(d_model, 1, bias=False)

        ## for hc
        self.Wc1_ = nn.Linear(d_model, d_model, bias=False)
        self.Wc2_ = nn.Linear(d_model, d_model, bias=False)
        self.vc_ = nn.Linear(d_model, 1, bias=False)

        # Bilinear Attention
        ## for hp
        self.Wb = nn.Linear(d_model, d_model, bias=False)

        ## for hc
        self.Wb_ = nn.Linear(d_model, d_model, bias=False)

        # Dot Attention
        ## for hp
        self.Wd = nn.Linear(d_model, d_model, bias=False)
        self.vd = nn.Linear(d_model, 1, bias=False)

        ## for hc
        self.Wd_ = nn.Linear(d_model, d_model, bias=False)
        self.vd_ = nn.Linear(d_model, 1, bias=False)


        # Minus Attention
        ## for hp
        self.Wm = nn.Linear(d_model, d_model, bias=False)
        self.Vm = nn.Linear(d_model, 1, bias=False)

        ## for hc
        self.Wm_ = nn.Linear(d_model, d_model, bias=False)
        self.Vm_ = nn.Linear(d_model, 1, bias=False)

        self.Ws = nn.Linear(d_model, d_model, bias=False)
        self.vs = nn.Linear(d_model, 1, bias=False)

        self.Ws_ = nn.Linear(d_model, d_model, bias=False)
        self.vs_ = nn.Linear(d_model, 1, bias=False)

        # self.trans_agg = nn.GRU(12 * encoder_size, encoder_size, batch_first=True, bidirectional=True)
        '''
        prediction layer
        '''
        self.W_agg_p = nn.Linear(4 * d_model, d_model)
        self.W_agg_p_ = nn.Linear(4 * d_model, d_model)
        self.W_agg_c = nn.Linear(2 * d_model, d_model)
        self.W_agg_p_s = nn.Linear(2 * d_model, d_model)
        self.Wp = nn.Linear(d_model, d_model, bias=False)
        self.vp = nn.Linear(d_model, 1, bias=False)
        # self.Wc1_p = nn.Linear(4 * d_model, d_model, bias=False)
        # self.Wc1_p_ = nn.Linear(4 * d_model, d_model, bias=False)

        # self.Wc1_p_self = nn.Linear(2 * d_model, d_model, bias=False)
        # self.Wc1_p_self_ = nn.Linear(2 * d_model, d_model, bias=False)

        self.Wc2_p = nn.Linear(d_model, d_model, bias=False)
        self.vc_p = nn.Linear(d_model, 1, bias=False)
        self.vc_p_ = nn.Linear(d_model, 1, bias=False)
        self.vc_p_self = nn.Linear(d_model, 1, bias=False)
        self.vc_p_self_ = nn.Linear(d_model, 1, bias=False)
        # self.prediction = nn.Linear(4 * d_model, class_num, bias=False)
        # self.prediction = nn.Linear(6 * d_model, class_num, bias=False)
        
        self.l0 = nn.Linear(4 * 4 * d_model + 2 * 2 * d_model, 2)
        torch.nn.init.normal_(self.l0.weight, std=0.02)

        self.l1 = nn.Linear(4 * 4 * d_model + 2 * 2 * d_model, BINs)
        torch.nn.init.normal_(self.l1.weight, std=0.02)
        
        self.l2 = nn.Linear(4 * 4 * d_model + 2 * 2 * d_model, BINs)
        torch.nn.init.normal_(self.l2.weight, std=0.02)
        
#         self.prediction = nn.Linear(4 * 4 * d_model + 2 * 2 * d_model, class_num, bias=False)
        self.pred_dropout = nn.Dropout(drop_out)

        # self.self_atten_r = self_atten_r
        # self.multi_atten_r = multi_atten_r

    def forward(self, post, comm):
        """
        post: analysis bs * seq_len
        comm: reference answer bs* seq_len

        return:
        attention matrix and sigmoid score
        """
        batch_size = post.shape[0]
        p_embedding = self.word_embedding(post)
        p_embedding = self.word_dropout(p_embedding)

        p_embedding = p_embedding + self.pos_emb(p_embedding)
        c_embedding = self.word_embedding(comm)
        c_embedding = self.word_dropout(c_embedding)

        c_embedding = c_embedding + self.pos_emb(c_embedding)

        hp = self.encoder(p_embedding)
        hp = self.dropout(hp)

        hc = self.encoder(c_embedding)
        hc = self.dropout(hc)

        # hp
        # Add # done no add
        _s1 = self.Wc1(hp).unsqueeze(1)
        _s2 = self.Wc2(hc).unsqueeze(2)
        sjt = self.vc(torch.tanh(_s1 + _s2)).squeeze()
        ait = F.softmax(sjt, 2)
        ait_add = ait
        ptc = ait.bmm(hp)

        # mul
        _s1 = self.Wb(hp).transpose(2, 1)
        sjt = hc.bmm(_s1)
        ait = F.softmax(sjt, 2)
        ptb = ait.bmm(hp)
        ait_mul = ait

        # Dot
        _s1 = hp.unsqueeze(1)
        _s2 = hc.unsqueeze(2)
        sjt = self.vd(torch.tanh(self.Wd(_s1 * _s2))).squeeze()
        ait = F.softmax(sjt, 2)
        ptd = ait.bmm(hp)
        ait_dot = ait

        # sub
        _s1 = hp.unsqueeze(1)
        _s2 = hc.unsqueeze(2)
        sjt = self.Vm(torch.tanh(self.Wm(_s1 - _s2))).squeeze()
        ait = F.softmax(sjt, 2)
        ptm = ait.bmm(hp)
        ait_sub = ait


        # hc
        # Add # done no add
        _s1 = self.Wc1_(hc).unsqueeze(1)
        _s2 = self.Wc2_(hp).unsqueeze(2)
        sjt = self.vc_(torch.tanh(_s1 + _s2)).squeeze()
        ait = F.softmax(sjt, 2)
        ptc_ = ait.bmm(hc)

        # mul
        _s1 = self.Wb_(hc).transpose(2, 1)
        sjt = hp.bmm(_s1)
        ait = F.softmax(sjt, 2)
        ptb_ = ait.bmm(hc)

        # Dot
        _s1 = hc.unsqueeze(1)
        sjt = self.vd_(torch.tanh(self.Wd_(_s1 * _s2))).squeeze()
        # print((_s1*_s2).shape)
        ait = F.softmax(sjt, 2)
        ptd_ = ait.bmm(hc)
        _s2 = hp.unsqueeze(2)

        # sub
        sjt = self.Vm_(torch.tanh(self.Wm_(_s1 - _s2))).squeeze()
        ait = F.softmax(sjt, 2)
        ptm_ = ait.bmm(hc)

        _s1 = hc.unsqueeze(1)
        _s2 = hc.unsqueeze(2)
        sjt = self.vs(torch.tanh(self.Ws(_s1 * _s2))).squeeze()
        ait = F.softmax(sjt, 2)
        qts = ait.bmm(hc)

        _s1 = hp.unsqueeze(1)
        _s2 = hp.unsqueeze(2)
        sjt = self.vs_(torch.tanh(self.Ws_(_s1 * _s2))).squeeze()
        ait = F.softmax(sjt, 2)
        pts = ait.bmm(hp)

        # aggregation_p = self.W_agg_p(torch.cat([ptc,ptb,ptd,ptm],2))
        # without add
        aggregation_p = torch.cat([ptc, ptb, ptd, ptm], 2)
        aggregation_p_ = torch.cat([ptc_, ptb_, ptd_, ptm_], 2)

        aggregation_c = torch.cat([hc, qts], 2)
        # aggregation_c = self.W_agg_c(torch.cat([hc],2))
        aggregation_p_s = torch.cat([hp, pts], 2)
        # aggregation_p_s = self.W_agg_p_s(torch.cat([hp],2))

        # self_attention to make left matrix in vector
        sj = F.softmax(self.vc_p(self.W_agg_p(aggregation_p)).transpose(2, 1), 2)
        rc = sj.bmm(aggregation_p)
        rc = rc.squeeze()

        # self_attention to make right matrix in vector
        sj_ = F.softmax(self.vc_p_(self.W_agg_p_(aggregation_p_)).transpose(2, 1), 2)
        rc_ = sj_.bmm(aggregation_p_)
        rc_ = rc_.squeeze()

        # self_attention to make left matrix in vector
        sj = F.softmax(self.vc_p_self(self.W_agg_c(aggregation_c)).transpose(2, 1), 2)
        rc_self = sj.bmm(aggregation_c)
        rc_self = rc_self.squeeze()

        # self_attention to make right matrix in vector
        sj_ = F.softmax(self.vc_p_self_(self.W_agg_p_s(aggregation_p_s)).transpose(2, 1), 2)
        rc_self_ = sj_.bmm(aggregation_p_s)
        rc_self_ = rc_self_.squeeze()

        predict_feas = torch.cat([rc_self, rc_self_, rc, rc_, rc * rc_, torch.abs(rc - rc_)], 1)
        
        if self.training:

            logits0 = torch.mean(
                torch.stack(
                    [self.l0(self.pred_dropout(predict_feas)) for _ in range(10)],
                    dim=0,
                ),
                dim=0,
            )

            logits1 = torch.mean(
                torch.stack(
                    [self.l1(self.pred_dropout(predict_feas)) for _ in range(10)],
                    dim=0,
                ),
                dim=0,
            )
            
            logits2 = torch.mean(
                torch.stack(
                    [self.l2(self.pred_dropout(predict_feas)) for _ in range(10)],
                    dim=0,
                ),
                dim=0,
            )            
        else:
            logits0 = self.l0(predict_feas)   #[batch, 2]
            logits1 = self.l1(predict_feas)   #[batch, BINs]
            logits2 = self.l2(predict_feas)   #[batch, BINs]
        return logits0, logits1, logits2