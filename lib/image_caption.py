import cv2
import numpy as np
import torch
import torch.utils.data as data
import os
import torchvision.transforms as T
import random
import json
import logging
import lib.utils as utils
from PIL import Image, ImageFile
from imageio import imread

ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger(__name__)


def build_transforms(img_size=224, is_train=True):

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if not is_train:
        transform = T.Compose([
            T.Resize((img_size, img_size) , interpolation=Image.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
    else:
        transform = T.Compose([
            T.RandomResizedCrop(img_size, scale=(0.6, 1.0), interpolation=Image.BICUBIC),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

    return transform


class RawImageDataset(data.Dataset):

    def __init__(self, opt, data_path, split, tokenizer, train):
        
        self.opt = opt

        self.train = train
        self.data_path = data_path
        self.split = split
        self.tokenizer = tokenizer
        self.train = train

        # f30k: 31014 imgs, 145000 train_captions
        # coco: 119287 imgs, 
        loc = os.path.join(opt.data_path, opt.dataset)

        self.image_base = opt.f30k_img_path if opt.dataset == 'f30k' else opt.coco_img_path

        with open(os.path.join(loc, 'id_mapping.json'), 'r') as f:
            self.id_to_path = json.load(f)

        # Read Captions
        self.captions = []
        # data_split: train or dev
        with open(os.path.join(loc, '%s_caps.txt' % self.split), 'r') as f:
            for line in f:
                self.captions.append(line.strip())

        if self.opt.pretrain and self.train:
            self.captions = []
            # 打开文件，逐行读取
            # with open(os.path.join(loc, f'{opt.dataset}_{self.split}_temp{opt.temp}.jsonl'), "r", encoding="utf-8") as f:
            with open(os.path.join(loc, f'{opt.dataset}_{self.split}_short200.jsonl'), "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        # 解析每一行为 JSON 对象
                        json_obj = json.loads(line.strip())
                        self.captions.append(json_obj['text'])  # 添加到列表
                    except json.JSONDecodeError as e:
                        print(f"Error parsing line: {e}")
        if self.opt.distill:
            self.long_caption = []
            # 打开文件，逐行读取
            with open(os.path.join(loc, f'{opt.dataset}_{self.split}.jsonl'), "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        # 解析每一行为 JSON 对象
                        json_obj = json.loads(line.strip())
                        self.long_caption.append(json_obj['text'])  # 添加到列表
                    except json.JSONDecodeError as e:
                        print(f"Error parsing line: {e}")

        # Get the image ids
        with open(os.path.join(loc, '{}_ids.txt'.format(self.split)), 'r') as f:
            image_ids = f.readlines()
            self.images = [int(x.strip()) for x in image_ids]

        self.preprocess = build_transforms(img_size=opt.img_res, is_train=train)
        
        self.length = len(self.captions)
        self.num_images = len(self.images)

        self.im_div = 5 if self.num_images != self.length else 1
        self.imagenet_mean = [0.485, 0.456, 0.406]
        self.imagenet_std = [0.229, 0.224, 0.225]
        print(opt.dataset, self.split)

    def __getitem__(self, index):
        
        img_index = index // self.im_div
        caption = self.captions[index]

        # Convert caption (string) to word ids (with Size Augmentation at training time).
        if 'clip' in self.opt.vit_type:
            caption_tokens = self.tokenizer.tokenize(caption)  
            target = process_caption_clip(self.tokenizer, caption_tokens, self.train, size_augment=self.opt.size_augment)
        else:
            caption_tokens = self.tokenizer.basic_tokenizer.tokenize(caption)  
            target = process_caption_bert(self.tokenizer, caption_tokens, self.train, size_augment=self.opt.size_augment)

        if self.opt.distill:
            if self.train:
                long_caption = self.long_caption[img_index]
            else:
                long_caption = self.long_caption[index // 5]

            if 'clip' in self.opt.vit_type:
                long_caption_tokens = self.tokenizer.basic_tokenizer.tokenize(long_caption)  
                long_target = process_caption_clip(self.tokenizer, long_caption_tokens, self.train, size_augment=self.opt.size_augment)
            else:
                long_caption_tokens = self.tokenizer.basic_tokenizer.tokenize(long_caption)  
                long_target = process_caption_bert(self.tokenizer, long_caption_tokens, self.train, size_augment=self.opt.size_augment)
        
        
        # image_id = self.images[img_index]
        # image_path = os.path.join(self.image_base, self.id_to_path[str(image_id)])
        # image = Image.open(image_path).convert("RGB")     
        # image = self.preprocess(image)  

        image_id = self.images[img_index]
        image_path = os.path.join(self.image_base, self.id_to_path[str(image_id)])
        im_in = np.array(imread(image_path))
        processed_image = self._process_image(im_in)
        image = torch.Tensor(processed_image)
        image = image.permute(2, 0, 1)            

        if self.opt.distill:
            return image, target, long_target, index, img_index
        return image, target, index, img_index

    def __len__(self):
        return self.length

    def _process_image(self, im_in):
        """
            Converts an image into a network input, with pre-processing including re-scaling, padding, etc, and data
        augmentation.
        """
        if len(im_in.shape) == 2:
            im_in = im_in[:, :, np.newaxis]
            im_in = np.concatenate((im_in, im_in, im_in), axis=2)

        im = im_in.astype(np.float32, copy=True)

        target_size = self.opt.img_res

        # 2. Random crop when in training mode, elsewise just skip
        if self.train:
            crop_ratio = np.random.random() * 0.4 + 0.6
            crop_size_h = int(im.shape[0] * crop_ratio)
            crop_size_w = int(im.shape[1] * crop_ratio)
            processed_im = self._crop(im, crop_size_h, crop_size_w, random=True)
        else:
            processed_im = im

        # 3. Resize to the target resolution
        im_shape = processed_im.shape
        im_scale_x = float(target_size) / im_shape[1]
        im_scale_y = float(target_size) / im_shape[0]
        processed_im = cv2.resize(processed_im, None, None, fx=im_scale_x, fy=im_scale_y,
                                  interpolation=cv2.INTER_LINEAR)

        if self.train:
            if np.random.random() > 0.5:
                processed_im = self._hori_flip(processed_im)

        processed_im = self._imagenet_norm(processed_im)

        return processed_im

    def _imagenet_norm(self, im_in):
        im_in = im_in.astype(np.float32)
        im_in = im_in / 255
        for i in range(im_in.shape[-1]):
            im_in[:, :, i] = (im_in[:, :, i] - self.imagenet_mean[i]) / self.imagenet_std[i]
        return im_in
    
    @staticmethod
    def _crop(im, crop_size_h, crop_size_w, random):
        h, w = im.shape[0], im.shape[1]
        if random:
            if w - crop_size_w == 0:
                x_start = 0
            else:
                x_start = np.random.randint(w - crop_size_w, size=1)[0]
            if h - crop_size_h == 0:
                y_start = 0
            else:
                y_start = np.random.randint(h - crop_size_h, size=1)[0]
        else:
            x_start = (w - crop_size_w) // 2
            y_start = (h - crop_size_h) // 2

        cropped_im = im[y_start:y_start + crop_size_h, x_start:x_start + crop_size_w, :]

        return cropped_im

    @staticmethod
    def _hori_flip(im):
        im = np.fliplr(im).copy()
        return im



def process_caption_clip(tokenizer, tokens, train=True, mask_rate=0.2, size_augment=True):
    output_tokens = []
    sot_token = tokenizer.convert_tokens_to_ids('<|startoftext|>')
    eot_token = tokenizer.convert_tokens_to_ids('<|endoftext|>')

    tokens = tokenizer.convert_tokens_to_ids(tokens)

    deleted_idx = []
    for i, token in enumerate(tokens):

        # the sentence is tokenized twice 
        # text -> basic token (basic_tokenizer.tokenize) -> sub_token (wordpiece_tokenizer.tokenize)
        # sub_tokens = tokenizer.wordpiece_tokenizer.tokenize(token)
        prob = random.random()
        # first, 20% probability use the augmenation operations
        if size_augment and prob < mask_rate and train:  # mask/remove the tokens only during training
            prob /= mask_rate
            # 50% change token to mask token
            if prob < 0.5:
                    output_tokens.append(tokenizer.convert_tokens_to_ids("[MASK]"))
            # 10% randomly change token to random token from the BERT-vocab
            elif prob < 0.6:
                    output_tokens.append(random.choice(list(range(tokenizer.vocab_size))))
            # -> 40% delete the token
            else:
                pass
        # 80% probability keep the token
        else:
            output_tokens.append(token)

   

    # and first and last notations for BERT model
    output_tokens = [sot_token] + output_tokens + [eot_token]

    # Convert each token to vocabulary indices
    # [PAD] -> 0
    # [UNK] -> 100
    # [CLS] -> 101
    # [SEP] -> 102
    # [MASK] -> 103
    # target = tokenizer.convert_tokens_to_ids(output_tokens)

    # convert to the torch.Tensor, torch.int64 (long)
    target = torch.tensor(output_tokens)

    return target

def process_caption_bert(tokenizer, tokens, train=True, mask_rate=0.2, size_augment=True):

    output_tokens = []
    deleted_idx = []

    for i, token in enumerate(tokens):

        # the sentence is tokenized twice 
        # text -> basic token (basic_tokenizer.tokenize) -> sub_token (wordpiece_tokenizer.tokenize)
        sub_tokens = tokenizer.wordpiece_tokenizer.tokenize(token)

        prob = random.random()

        # first, 20% probability use the augmenation operations
        if size_augment and prob < mask_rate and train:  # mask/remove the tokens only during training
            prob /= mask_rate

            # 50% change token to mask token
            if prob < 0.5:
                for sub_token in sub_tokens:
                    output_tokens.append("[MASK]")
            # 10% randomly change token to random token from the BERT-vocab
            elif prob < 0.6:
                for sub_token in sub_tokens:
                    output_tokens.append(random.choice(list(tokenizer.vocab.keys())))
                    
            # -> 40% delete the token
            else:
                for sub_token in sub_tokens:
                    output_tokens.append(sub_token)
                    # record the index of sub_token
                    deleted_idx.append(len(output_tokens) - 1)
        
        # 80% probability keep the token
        else:
            for sub_token in sub_tokens:
                # no masking token (will be ignored by loss function later)
                output_tokens.append(sub_token)

    if len(deleted_idx) != 0:
        output_tokens = [output_tokens[i] for i in range(len(output_tokens)) if i not in deleted_idx]

    # and first and last notations for BERT model
    output_tokens = ['[CLS]'] + output_tokens + ['[SEP]']

    # Convert each token to vocabulary indices
    # [PAD] -> 0
    # [UNK] -> 100
    # [CLS] -> 101
    # [SEP] -> 102
    # [MASK] -> 103
    target = tokenizer.convert_tokens_to_ids(output_tokens)

    # convert to the torch.Tensor, torch.int64 (long)
    target = torch.tensor(target)

    return target


def collate_fn_ours(data):

    # Sort a data list by caption length, for GRU/BERT
    data.sort(key=lambda x: len(x[1]), reverse=True)

    # distill = len(zip(*data)) != 4
    columns = list(zip(*data))  # 将 data 解压为列
    num_columns = len(columns)  # 获取列数

    if num_columns != 4:
        images, captions, long_captions, ids, img_ids = zip(*data)
    else:
        images, captions, ids, img_ids = zip(*data)

    # img label
    img_ids = torch.tensor(img_ids)
    # cap label (five captions with one image)
    ids = torch.tensor(ids)
    
    images = torch.stack(images, 0)

    # Merget captions (convert tuple of 1D tensor to 2D tensor)
    lengths = torch.tensor([len(cap) for cap in captions])

    targets = torch.zeros(len(captions), max(lengths)).long()

    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]

    if num_columns != 4:
        long_lengths = torch.tensor([len(cap) for cap in long_captions])
        long_targets = torch.zeros(len(long_captions), max(long_lengths)).long()
        for i, cap in enumerate(long_captions):
            end = long_lengths[i]
            long_targets[i, :end] = cap[:end]
        
        return images, targets, lengths, long_targets, long_lengths, ids, img_ids
    
    # print("===", max(lengths),min(lengths),lengths.float().mean(), "---")
    return images, targets, lengths, ids, img_ids


def get_loader(opt, data_path, split, tokenizer, 
               batch_size=128, shuffle=True, 
               num_workers=2, train=True,
               ):

    dataset = RawImageDataset(opt, data_path, split, tokenizer, train)
    collate_fn = collate_fn_ours

    # DDP with multi GPUS
    # only for train_loader
    if opt.multi_gpu and train:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()   
        sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank, shuffle=shuffle)
        shuffle = False
    else:
        sampler = None

    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                batch_size=batch_size,
                                                sampler=sampler,
                                                shuffle=shuffle,
                                                num_workers=num_workers,
                                                pin_memory=True,
                                                collate_fn=collate_fn,
                                                drop_last=train,
                                                )
    return data_loader


def get_train_loader(opt, data_path, tokenizer, batch_size, workers, split='train'):
    
    train_loader = get_loader(opt, data_path, split, tokenizer,
                              batch_size, True, workers, train=True)
    return train_loader


def get_test_loader(opt, data_path, tokenizer, batch_size, workers, split='test'):

    test_loader = get_loader(opt, data_path, split, tokenizer,
                             batch_size, False, workers, train=False) 
    return test_loader


if __name__ == '__main__':

    pass
