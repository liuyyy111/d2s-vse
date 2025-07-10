import torch
import torch.nn as nn
import torch.nn.init
import lib.utils as utils
import logging

from timm.models.vision_transformer import Block

from lib.encoders import GPO, get_image_encoder, get_sim_encoder, get_text_encoder
from lib.loss import loss_select

from lib.cross_net import CrossSparseAggrNet_v2

logger = logging.getLogger(__name__)

def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X



class PromptLearner(nn.Module):
    def __init__(self, num_tokens=4, dim=512, use_decoder=False, decoder_depth=4, gpo=False, no_res_embed=False):
        super().__init__()
        self.prompt_tokens = nn.Parameter(torch.randn(num_tokens, dim))
        self.num_tokens = num_tokens
        self.use_decoder = use_decoder
        self.gpo = gpo
        self.no_res_embed = no_res_embed
        if self.use_decoder:
            print('use decoder')
            self.mask_token = nn.Parameter(torch.zeros(num_tokens, dim))

            self.decoder_blocks = nn.ModuleList([
                Block(dim, 4, mlp_ratio=4., qkv_bias=True, norm_layer=nn.LayerNorm)
                for i in range(decoder_depth)])

            self.decoder_norm = nn.LayerNorm(dim)

            if self.gpo:
                self.pooling = GPO(32, 32)
        else:
            print("use cross attention")
            self.cross_attn = nn.MultiheadAttention(dim, num_heads=8)

    def forward(self, text_feat, lengths=None):
        # text_feat: [seq_len, batch_size, dim]

        if self.use_decoder:
            text_feat = text_feat.permute(1, 0, 2)
            bs, seq_len, dim = text_feat.shape
            # mask = torch.arange(seq_length).expand(batch_size, seq_length).to(last_hidden_states.device) < lengths  # (batch_size, seq_length)

            prompt = self.prompt_tokens.unsqueeze(0).repeat(text_feat.size(0), 1, 1)
            x = torch.cat([text_feat, prompt], dim=1)

            for blk in self.decoder_blocks:
                x = blk(x)
            x = self.decoder_norm(x)
            # if self.no_res_embed:
            x = x[:, -self.num_tokens:, :]
            
            if self.gpo:
                x = self.pooling(x, torch.ones_like(x.mean(dim=-1)).sum(dim=-1))[0]
                return x
            else:
                return x.mean(dim=1)
        else:
            prompt = self.prompt_tokens.unsqueeze(1).repeat(1, text_feat.size(1), 1)

            attn_output, _ = self.cross_attn(
                query=prompt,
                key=text_feat,
                value=text_feat
            )
            attn_output += prompt

            return attn_output.mean(dim=0) 


class VSEModel(nn.Module):
    def __init__(self, opt):
        super().__init__()

        self.opt = opt

        self.img_enc = get_image_encoder(opt)
        self.txt_enc = get_text_encoder(opt)

        self.criterion = loss_select(opt, loss_type=opt.loss)

        # iteration
        self.Eiters = 0

        if self.opt.distill:
            self.long_txt_enc = get_text_encoder(opt)
            path = self.opt.weight_path
            ckpt = torch.load(path, map_location='cpu')
            base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['model'].items()}
            img_ckpt = {k.replace("img_enc.", ""): v for k, v in base_ckpt.items() if 'img_enc' in k}
            self.img_enc.load_state_dict(img_ckpt, strict=True)

            txt_ckpt = {k.replace("txt_enc.", ""): v for k, v in base_ckpt.items() if 'txt_enc' in k}
            self.txt_enc.load_state_dict(txt_ckpt, strict=True)

            self.long_txt_enc.load_state_dict(txt_ckpt, strict=True)

            for param in self.long_txt_enc.parameters():
                param.requires_grad = False

            self.prompt_learner = PromptLearner(num_tokens=opt.num_tokens, use_decoder=opt.use_decoder, decoder_depth=opt.decoder_depth, gpo=opt.gpo, no_res_embed=opt.no_res_embed)
            
    def freeze_backbone(self):
        self.img_enc.freeze_backbone()
        self.txt_enc.freeze_backbone()

    def unfreeze_backbone(self):
        self.img_enc.unfreeze_backbone()
        self.txt_enc.unfreeze_backbone()

    def set_max_violation(self, max_violation=True):
        if max_violation:
            self.criterion.max_violation_on()
        else:
            self.criterion.max_violation_off()

    # Compute the image and caption embeddings
    def forward_emb(self, images, captions, lengths):
        img_emb = self.img_enc(images)

        cap_emb, word_emb = self.txt_enc(captions, lengths)

        if self.opt.distill:
            prompt = self.prompt_learner(word_emb.permute(1,0,2), lengths)  # 增加序列维度
            prompt = l2norm(prompt, dim=-1)
            if self.opt.no_res_embed:
                cap_emb = cap_emb + prompt
                cap_emb = l2norm(cap_emb, dim=-1)
            else:
                cap_emb = prompt
                cap_emb = l2norm(prompt, dim=-1)
        
        return img_emb, cap_emb, lengths
    
    # compute the similarity on cross-attention interaction
    def forward_sim(self, img_embs, cap_embs, cap_lens):
        img_embs = l2norm(img_embs, -1)
        cap_embs = l2norm(cap_embs, -1)

        sims = img_embs.mm(cap_embs.t())

        return sims

    # One training step given images and captions
    def forward(self, images, captions, lengths, long_captions=None, long_lengths=None, img_ids=None, warmup_alpha=1.,):

        self.Eiters += 1
      
        img_emb = self.img_enc(images)
        cap_emb, word_emb = self.txt_enc(captions, lengths)

        if self.opt.distill:
            with torch.no_grad():
                long_cap_emb, _ = self.long_txt_enc(long_captions, long_lengths)
            # enhanced text embeding by prompt
            prompt = self.prompt_learner(word_emb.permute(1,0,2), lengths)  

            if self.opt.no_res_embed:
                cap_emb = cap_emb + prompt
            else:
                cap_emb = prompt
            
        # get all samples for compute loss function
        # if self.opt.multi_gpu and (not self.opt.cross_attention):
        if self.opt.multi_gpu:
            lengths = utils.concat_all_gather(lengths, keep_grad=False)
            img_ids = utils.concat_all_gather(img_ids, keep_grad=False)
                

            img_emb = utils.all_gather_with_grad(img_emb)
            cap_emb = utils.all_gather_with_grad(cap_emb) 

            if self.opt.distill:
                long_lengths = utils.concat_all_gather(long_lengths, keep_grad=False)
                prompt = utils.all_gather_with_grad(prompt)           
                long_cap_emb = utils.all_gather_with_grad(long_cap_emb)

        # compute similarity matrix
        improved_sims = self.forward_sim(img_emb, cap_emb, lengths)

        # basic alignment loss
        align_loss = self.criterion(img_emb, cap_emb, img_ids, improved_sims) * warmup_alpha

        if self.opt.distill:
            if self.opt.distill_loss == 'cos':
                distill_loss = distillation_loss
            elif self.opt.distill_loss == 'l2':
                distill_loss = l2_distillation_loss
                
            distill_loss = distill_loss(cap_emb, long_cap_emb.detach())  
            # basic alignment loss
            # long_sims = self.forward_sim(long_cap_emb.detach(), cap_emb, lengths)
            # distill_loss = self.criterion(long_cap_emb.detach(), cap_emb, img_ids, long_sims) * warmup_alpha
            # loss = align_loss + distill_loss + long_align_loss
            loss = align_loss + distill_loss

        else:
            loss = align_loss

        return loss
    

def l2_distillation_loss(student, teacher):
    l2_loss = nn.MSELoss()(student, teacher)  

    return l2_loss


def distillation_loss(student, teacher, temperature=0.1):
    student = student / student.norm(dim=1, keepdim=True)
    teacher = teacher / teacher.norm(dim=1, keepdim=True)
    return 1 - (student * teacher).sum(dim=1).mean()

def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


# optimizer init
def create_optimizer(opt, model):

    # Set up the lr for different parts of the VSE model
    decay_factor = 1e-4  
    cross_lr_rate = 1.0
        
    # bert params
    all_text_params = list(model.txt_enc.parameters())
    bert_params = list(model.txt_enc.bert.parameters())
    bert_params_ptr = [p.data_ptr() for p in bert_params]
    text_params_no_bert = list()

    for p in all_text_params:
        if p.data_ptr() not in bert_params_ptr:
            text_params_no_bert.append(p)

    # bert   
    params_list = [
        {'params': text_params_no_bert, 'lr': opt.learning_rate},
        {'params': bert_params, 'lr': opt.learning_rate * 0.1},
    ]

    # vit
    params_list += [
        {'params': model.img_enc.visual_encoder.parameters(), 'lr': opt.learning_rate * 0.1},
        {'params': model.img_enc.image_encoder.parameters(), 'lr': opt.learning_rate},
    ]

    if opt.distill:
        params_list += [
            {'params': model.prompt_learner.parameters(), 'lr': opt.learning_rate},
        ]
    # # cross-moadl alignment 
    # params_list += [
    #     {'params': model.cross_net.parameters(), 'lr': opt.learning_rate * cross_lr_rate},
    #     {'params': model.criterion.parameters(), 'lr': opt.learning_rate},
    # ]   
  
    optimizer = torch.optim.AdamW(params_list, lr=opt.learning_rate, weight_decay=decay_factor)
    
    return optimizer


if __name__ == '__main__':

    pass

