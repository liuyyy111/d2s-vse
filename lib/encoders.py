import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import BertModel, BertTokenizer, SwinModel, ViTModel
import logging


def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X

def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""
    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)

logger = logging.getLogger(__name__)


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.output_dim = output_dim
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.bns = nn.ModuleList(nn.BatchNorm1d(k) for k in h + [output_dim])

    def forward(self, x):
        B, N, D = x.size()
        x = x.reshape(B * N, D)
        for i, (bn, layer) in enumerate(zip(self.bns, self.layers)):
            x = F.relu(bn(layer(x))) if i < self.num_layers - 1 else layer(x)
        x = x.view(B, N, self.output_dim)
        return x


def positional_encoding_1d(d_model, length):
    """
    :param d_model: dimension of the model
    :param length: length of positions
    :return: length*d_model position matrix
    """
    if d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(d_model))
    pe = torch.zeros(length, d_model)
    position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                          -(math.log(10000.0) / d_model)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe


class GPO(nn.Module):
    def __init__(self, d_pe, d_hidden):
        super(GPO, self).__init__()
        self.d_pe = d_pe
        self.d_hidden = d_hidden

        self.pe_database = {}
        self.gru = nn.GRU(self.d_pe, d_hidden, 1, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(self.d_hidden, 1, bias=False)

    def compute_pool_weights(self, lengths, features):
        max_len = int(lengths.max())
        pe_max_len = self.get_pe(max_len)
        pes = pe_max_len.unsqueeze(0).repeat(lengths.size(0), 1, 1).to(lengths.device)
        mask = torch.arange(max_len).expand(lengths.size(0), max_len).to(lengths.device)
        mask = (mask < lengths.long().unsqueeze(1)).unsqueeze(-1)
        pes = pes.masked_fill(mask == 0, 0)

        self.gru.flatten_parameters()
        packed = pack_padded_sequence(pes, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.gru(packed)
        padded = pad_packed_sequence(out, batch_first=True)
        out_emb, out_len = padded
        out_emb = (out_emb[:, :, :out_emb.size(2) // 2] + out_emb[:, :, out_emb.size(2) // 2:]) / 2
        scores = self.linear(out_emb)
        scores[torch.where(mask == 0)] = -10000

        weights = torch.softmax(scores / 0.1, 1)
        return weights, mask

    def forward(self, features, lengths):
        """
        :param features: features with shape B x K x D
        :param lengths: B x 1, specify the length of each data sample.
        :return: pooled feature with shape B x D
        """
        pool_weights, mask = self.compute_pool_weights(lengths, features)

        features = features[:, :int(lengths.max()), :]
        sorted_features = features.masked_fill(mask == 0, -10000)
        sorted_features, sort = sorted_features.sort(dim=1, descending=True)

        sorted_features = sorted_features.masked_fill(mask == 0, 0)

        pooled_features = (sorted_features * pool_weights).sum(1)
        return pooled_features, pool_weights

    def get_pe(self, length):
        """

        :param length: the length of the sequence
        :return: the positional encoding of the given length
        """
        length = int(length)
        if length in self.pe_database:
            return self.pe_database[length]
        else:
            pe = positional_encoding_1d(self.d_pe, length)
            self.pe_database[length] = pe
            return pe


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.)


def get_text_encoder(opt):
    txt_enc = EncoderText_BERT(opt)   
    return txt_enc


def get_image_encoder(opt):
    img_enc = VisionTransEncoder(opt)
    return img_enc

def get_sim_encoder(opt):
    return EncoderSimilarity(opt)


class EncoderImageAggr(nn.Module):
    def __init__(self, img_dim, embed_size):
        super(EncoderImageAggr, self).__init__()
        self.embed_size = embed_size
        self.fc = nn.Linear(img_dim, embed_size)
        self.gpool = GPO(32, 32)
        self.init_weights()

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features +
                                  self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

    def forward(self, images, image_lengths):
        """Extract image feature vectors."""
        features = self.fc(images)

        features, pool_weights = self.gpool(features, image_lengths)

        # if not self.no_imgnorm:
        features = l2norm(features, dim=-1)

        return features

# ViT encoder
class VisionTransEncoder(nn.Module):
    def __init__(self, opt):
        super().__init__()

        self.opt = opt

        # Swin model
        if 'swin' in opt.vit_type:                           
            # img_res 224 * 224, 7*7 patch
            # self.visual_encoder = SwinModel.from_pretrained("microsoft/swin-base-patch4-window7-224-in22k")
            # self.visual_encoder = SwinModel.from_pretrained("/home/sculiuyang/.cache/huggingface/hub/swin", local_files_only=True)
            # self.visual_encoder = SwinModel.from_pretrained("/home/sculiuyang/.cache/huggingface/hub/swin-base-patch4-window7-224-in22k", local_files_only=True)
            self.visual_encoder = SwinModel.from_pretrained(opt.vit_type)

            opt.num_patches = 49
            print('swin model')
        #  ViT model
        elif 'vit': 
            print("DDDDD")             
            # img_res 224 * 224, 14*14 patch
            # self.visual_encoder = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
            # self.visual_encoder = ViTModel.from_pretrained("/home/sculiuyang/.cache/huggingface/hub/vit-base-patch16-224-in21k")
            # from lib.modeling_vit import ViTModel
            self.visual_encoder = ViTModel.from_pretrained(opt.vit_type)

            # self.visual_encoder = ViTModel.from_pretrained("../weights_models/google--vit-base-patch16-224-in21k")
            opt.num_patches = 196
            print('vit model')
        

        self.dropout = nn.Dropout(0.2)  
        if 'swin' in opt.vit_type:
            self.image_encoder = EncoderImageAggr(1024, 512)
        else:
            self.image_encoder = EncoderImageAggr(768, 512)

    # @get_local('attention_map')
    def forward(self, images, return_atttention=False):
        # print(images.shape)
        # (B, L_v, C_hidden)
        base_features = self.visual_encoder(images, interpolate_pos_encoding=True).last_hidden_state

        if self.training:
            # Size Augmentation during training, randomly drop grids
            base_length = base_features.size(1)
            features = []
            feat_lengths = []
            rand_list_1 = np.random.rand(base_features.size(0), base_features.size(1))
            rand_list_2 = np.random.rand(base_features.size(0))
            for i in range(base_features.size(0)):
                if rand_list_2[i] > 0.2:
                    feat_i = base_features[i][np.where(rand_list_1[i] > 0.20 * rand_list_2[i])]
                    len_i = len(feat_i)
                    pads_i = torch.zeros(base_length - len_i, base_features.size(-1)).to(base_features.device)
                    feat_i = torch.cat([feat_i, pads_i], dim=0)
                else:
                    feat_i = base_features[i]
                    len_i = base_length
                feat_lengths.append(len_i)
                features.append(feat_i)
            base_features = torch.stack(features, dim=0)
            base_features = base_features[:, :max(feat_lengths), :]
            feat_lengths = torch.tensor(feat_lengths).to(base_features.device)
        else:
            feat_lengths = torch.zeros(base_features.size(0)).to(base_features.device)
            feat_lengths[:] = base_features.size(1)
        
        features = self.image_encoder(base_features, feat_lengths)
        
        return features
        
    def freeze_backbone(self):
        for param in self.visual_encoder.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.visual_encoder.parameters():  
            param.requires_grad = True     

# Language Model with BERT backbone
class EncoderText_BERT(nn.Module):
    def __init__(self, opt):
        super(EncoderText_BERT, self).__init__()

        self.opt = opt
        self.embed_size = opt.embed_size
        
        # self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        # from transformers import CLIPModel
        # clip = CLIPModel.from_pretrained("/home/sculiuyang/.cache/huggingface/hub/clip-vit-base-patch32", local_files_only=True)
        # # self.visual_encoder = 
        # self.bert = clip.text_model
        
        # self.tokenizer = BertTokenizer.from_pretrained(opt.bert_path)
        # self.bert = BertModel.from_pretrained(opt.bert_path)
        
        self.fc = nn.Linear(self.bert.config.hidden_size, opt.embed_size)
        self.dropout = nn.Dropout(0.2)
        self.ln = nn.LayerNorm(self.embed_size)

        self.mlp = MLP(768, self.embed_size // 2, self.embed_size, 2)
        self.gpool = GPO(32, 32)


    def forward(self, x, lengths):

        # Embed word ids to vectors
        # pad 0 for redundant tokens in previous process
        bert_attention_mask = (x != 0).float()

        # all hidden features, D=768 in bert-base model
        # attention_mask： Mask to avoid performing attention on padding token indices.
        # bert_output[0] is the last/final hidden states of all tokens
        # bert_output[1] is the hidden state of [CLS] + one fc layer + Tanh, can be used for classification tasks.

        # N = max_cap_lengths, D = 768
        bert_emb = self.bert(input_ids=x, attention_mask=bert_attention_mask)[0]  # B x N x D

        bert_emb = self.dropout(bert_emb)

        cap_len = lengths
        cap_emb = self.fc(bert_emb)

        features = self.mlp(bert_emb) + cap_emb
        pooled_features, pool_weights = self.gpool(features, cap_len.to(features.device))

        pooled_features = self.ln(pooled_features)
        
        return pooled_features, features

    def freeze_backbone(self):
        for param in self.bert.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.bert.parameters():  
            param.requires_grad = True  


def log_sinkhorn_iterations(Z: torch.Tensor, log_mu: torch.Tensor, log_nu: torch.Tensor, iters: int) -> torch.Tensor:
    """ Perform Sinkhorn Normalization in Log-space for stability"""
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)
    return Z + u.unsqueeze(2) + v.unsqueeze(1)


def log_optimal_transport(scores: torch.Tensor, alpha: torch.Tensor, iters: int) -> torch.Tensor:
    """ Perform Differentiable Optimal Transport in Log-space for stability"""
    b, m, n = scores.shape
    one = scores.new_tensor(1)
    ms, ns = (m * one).to(scores), (n * one).to(scores)

    bins0 = alpha.expand(b, m, 1)
    bins1 = alpha.expand(b, 1, n)
    alpha = alpha.expand(b, 1, 1)

    couplings = torch.cat([torch.cat([scores, bins0], -1),
                           torch.cat([bins1, alpha], -1)], 1)

    norm = - (ms + ns).log()
    log_mu = torch.cat([norm.expand(m), ns.log()[None] + norm])
    log_nu = torch.cat([norm.expand(n), ms.log()[None] + norm])
    log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)

    Z = log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)
    Z = Z - norm  # multiply probabilities by M+N
    return Z


class EncoderSimilarity(nn.Module):
    def __init__(self, opt):
        super(EncoderSimilarity, self).__init__()
        self.opt = opt
        self.block_dim = [128,256]
        bin_score = torch.nn.Parameter(torch.tensor(0.))
        self.register_parameter('bin_score', bin_score)

    def forward(self, img_emb, cap_emb):
        cap_emb = l2norm(cap_emb, -1)
        n_cap, cap_dim = cap_emb.size(0), cap_emb.size(1)
        n_img, img_dim = img_emb.size(0), img_emb.size(1)
        sims = []
        for i, block_dim in enumerate(self.block_dim):
            img_blk_num, cap_blk_num = img_emb.size(1) // block_dim, cap_emb.size(1) // block_dim
            img_emb_blocks = torch.chunk(img_emb, img_blk_num, -1)  # (bs, 2*n, block_dim)
            cap_emb_blocks = torch.chunk(cap_emb, cap_blk_num, -1)  # (bs, n, block_dim)

            img_emb_blocks = torch.stack(img_emb_blocks, dim=1)  # (bs, 2*n, block_dim)
            cap_emb_blocks = torch.stack(cap_emb_blocks, dim=1)  # (bs, n, block_dim)

            img_emb_blocks = l2norm(img_emb_blocks, -1)  # (bs, 2*n, block_dim)
            cap_emb_blocks = l2norm(cap_emb_blocks, -1)

            logits = torch.einsum("avc,btc->abvt", [img_emb_blocks, cap_emb_blocks])  # (bs, bs, 2*n, n)

            # logits = log_optimal_transport(logits.reshape(-1, img_blk_num, cap_blk_num), self.bin_score, 20)[:, :-1,
            #          :-1].reshape(n_img, n_cap, img_blk_num, cap_blk_num)
            t2i_logits = logits.max(dim=-2)[0]
            sims.append(t2i_logits.sum(dim=-1))

        sims = torch.stack(sims, -1).sum(-1)

        return sims

if __name__ == '__main__':

    pass
