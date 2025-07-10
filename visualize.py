
# %%
from Visualizer.visualizer import get_local
get_local.activate()
import torch
import torchvision.transforms as T
from timm.models.vision_transformer import vit_base_patch16_224
import json
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt

# %%
def grid_show(to_shows, cols):
    rows = (len(to_shows)-1) // cols + 1
    it = iter(to_shows)
    fig, axs = plt.subplots(rows, cols, figsize=(rows*8.5, cols*2))
    for i in range(rows):
        for j in range(cols):
            try:
                image, title = next(it)
            except StopIteration:
                image = np.zeros_like(to_shows[0][0])
                title = 'pad'
            axs[i, j].imshow(image)
            axs[i, j].set_title(title)
            axs[i, j].set_yticks([])
            axs[i, j].set_xticks([])
    plt.show()

def visualize_head(att_map):
    ax = plt.gca()
    # Plot the heatmap
    im = ax.imshow(att_map)
    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    plt.show()
    
def visualize_heads(att_map, cols):
    to_shows = []
    att_map = att_map.squeeze()
    for i in range(att_map.shape[0]):
        to_shows.append((att_map[i], f'Head {i}'))
    average_att_map = att_map.mean(axis=0)
    to_shows.append((average_att_map, 'Head Average'))
    grid_show(to_shows, cols=cols)

def gray2rgb(image):
    return np.repeat(image[...,np.newaxis],3,2)
    
def cls_padding(image, mask, cls_weight, grid_size):
    if not isinstance(grid_size, tuple):
        grid_size = (grid_size, grid_size)
        
    image = np.array(image)

    H, W = image.shape[:2]
    delta_H = int(H/grid_size[0])
    delta_W = int(W/grid_size[1])
    
    padding_w = delta_W
    padding_h = H
    padding = np.ones_like(image) * 255
    padding = padding[:padding_h, :padding_w]
    
    padded_image = np.hstack((padding,image))
    padded_image = Image.fromarray(padded_image)
    draw = ImageDraw.Draw(padded_image)
    draw.text((int(delta_W/4),int(delta_H/4)),'CLS', fill=(0,0,0)) # PIL.Image.size = (W,H) not (H,W)

    mask = mask / max(np.max(mask),cls_weight)
    cls_weight = cls_weight / max(np.max(mask),cls_weight)
    
    if len(padding.shape) == 3:
        padding = padding[:,:,0]
        padding[:,:] = np.min(mask)
    mask_to_pad = np.ones((1,1)) * cls_weight
    mask_to_pad = Image.fromarray(mask_to_pad)
    mask_to_pad = mask_to_pad.resize((delta_W, delta_H))
    mask_to_pad = np.array(mask_to_pad)

    padding[:delta_H,  :delta_W] = mask_to_pad
    padded_mask = np.hstack((padding, mask))
    padded_mask = padded_mask
    
    meta_mask = np.zeros((padded_mask.shape[0], padded_mask.shape[1],4))
    meta_mask[delta_H:,0: delta_W, :] = 1 
    
    return padded_image, padded_mask, meta_mask
    

def visualize_grid_to_grid_with_cls(att_map, grid_index, image, grid_size=14, alpha=0.6):
    if not isinstance(grid_size, tuple):
        grid_size = (grid_size, grid_size)
    
    attention_map = att_map[grid_index]
    cls_weight = attention_map[0]
    
    mask = attention_map[1:].reshape(grid_size[0], grid_size[1])
    mask = Image.fromarray(mask).resize((image.size))
    
    padded_image ,padded_mask, meta_mask = cls_padding(image, mask, cls_weight, grid_size)
    
    if grid_index != 0: # adjust grid_index since we pad our image
        grid_index = grid_index + (grid_index-1) // grid_size[1]
        
    grid_image = highlight_grid(padded_image, [grid_index], (grid_size[0], grid_size[1]+1))
    
    fig, ax = plt.subplots(1, 2, figsize=(10,7))
    fig.tight_layout()
    
    ax[0].imshow(grid_image)
    ax[0].axis('off')
    
    ax[1].imshow(grid_image)
    ax[1].imshow(padded_mask, alpha=alpha, cmap='rainbow')
    ax[1].imshow(meta_mask)
    ax[1].axis('off')
    

def visualize_grid_to_grid(att_map, grid_index, image, grid_size=14, alpha=0.6):
    if not isinstance(grid_size, tuple):
        grid_size = (grid_size, grid_size)
    
    H,W = att_map.shape
    with_cls_token = False
      
    grid_image = highlight_grid(image, [grid_index], grid_size)
    
    mask = att_map[grid_index].reshape(grid_size[0], grid_size[1])
    mask = Image.fromarray(mask).resize((image.size))
    
    fig, ax = plt.subplots(1, 2, figsize=(10,7))
    fig.tight_layout()
    
    ax[0].imshow(grid_image)
    ax[0].axis('off')
    
    ax[1].imshow(image)
    ax[1].imshow(mask/np.max(mask), alpha=alpha, cmap='rainbow')
    ax[1].axis('off')
    plt.show()
    
def highlight_grid(image, grid_indexes, grid_size=14):
    if not isinstance(grid_size, tuple):
        grid_size = (grid_size, grid_size)
    
    W, H = image.size
    h = H / grid_size[0]
    w = W / grid_size[1]
    image = image.copy()
    for grid_index in grid_indexes:
        x, y = np.unravel_index(grid_index, (grid_size[0], grid_size[1]))
        a= ImageDraw.ImageDraw(image)
        a.rectangle([(y*w,x*h),(y*w+w,x*h+h)],fill =None,outline ='red',width =2)
    return image
# %%

# image_path = "/home/sculiuyang/data/f30k/flickr30k-images/4146886427.jpg"  # 使用提供的图像路径
image_path = "/home/sculiuyang/data/f30k/flickr30k-images/1071201387.jpg"

image = Image.open(image_path)

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

transforms = T.Compose([
    T.Resize((224, 224) , interpolation=Image.BICUBIC),
    T.ToTensor(),
    T.Normalize(mean=mean, std=std),
])
# %%
# load model and options
from lib.vse import VSEModel

checkpoint = torch.load("./runs/f30k/vit384_distill/model_best.pth", map_location='cuda')
opt = checkpoint['opt']
model = VSEModel(opt)
model.load_state_dict(checkpoint['model'], strict=False)
# model.eval()
# %%
input_tensor = transforms(image).unsqueeze(0)
input_tensor.shape

with torch.no_grad():
    _, attention_map = model.img_enc(input_tensor, return_atttention=True)
    
# %%
attention_map[0].shape
a_map = attention_map[-1].numpy()
a_map[0,0,1:,1:]
print(a_map.shape)
# %%
# for i in range(12):

#     print(f"第{i}层")

npatch= 77
print(npatch)
a_map = attention_map[-1].numpy()
for j in range(12):
    print(f"第{j}个头")
    visualize_grid_to_grid(a_map[0,j,1:,1:], npatch, image)

b_map = attention_map[-2].numpy()
for j in range(12):
    print(f"第{j}个头")
    visualize_grid_to_grid(b_map[0,j,1:,1:], npatch, image)

c_map = attention_map[-3].numpy()
for j in range(12):
    print(f"第{j}个头")
    visualize_grid_to_grid(c_map[0,j,1:,1:], npatch, image)
# %%
def softmax(x, axis=-1):
    """
    数值稳定的 Softmax 函数
    参数：
        x : 输入数组（支持任意维度）
        axis : 指定计算的维度（默认最后一个维度）
    返回：
        与输入形状相同的概率分布数组
    """
    # 数值稳定性处理：减去最大值
    x_max = np.max(x, axis=axis, keepdims=True)
    x_exp = np.exp(x - x_max)
    
    # 计算概率分布
    sum_exp = np.sum(x_exp, axis=axis, keepdims=True)
    probs = x_exp / sum_exp
    
    return probs

# %%

# attn = np.random.randn(1,1,197,197)
# attn = softmax(attn, -1)
from scipy.ndimage import gaussian_filter

def generate_focused_attention(shape, center_bias=0.5, focus_strength=2.0, sigma=3.0):
    """
    生成具有中心聚集特性的注意力矩阵
    参数：
        shape : 目标形状 (H, W)
        center_bias : 中心偏置强度（0-1）
        focus_strength : 聚焦强度系数（>1）
        sigma : 高斯平滑参数
    返回：
        归一化的注意力矩阵
    """
    # 基础随机矩阵
    attn = np.random.randn(*shape)
    
    # 添加中心偏置
    h, w = shape
    y_center = h//2 + np.random.randint(-h//8, h//8)
    x_center = w//2 + np.random.randint(-w//8, w//8)
    
    # 生成中心高斯权重
    y, x = np.ogrid[:h, :w]
    center_mask = np.exp(-((x - x_center)**2 + (y - y_center)**2) / (0.2*min(h,w)))
    attn += center_bias * center_mask
    
    # 增强局部连续性
    attn = gaussian_filter(attn, sigma=sigma)
    
    # 非线性增强聚焦区域
    attn = np.power(attn, focus_strength)
    return attn / attn.sum()

# 生成聚集型注意力矩阵 (197x197)
raw_attn = generate_focused_attention((197, 197), 
                                    center_bias=1.2, 
                                    focus_strength=3.0,
                                    sigma=4.0)
attn = softmax(raw_attn.reshape(1,1,197,197), axis=-1)
print(attn.shape)
# %%
npatch=90
visualize_grid_to_grid(attn[0,0,1:,1:], npatch, image)
# %%
print(attn)
# a_map = attention_map[4].numpy()
# visualize_grid_to_grid_with_cls(a_map[0,1,:,:], 0, image)
# %%
# %%
# %%

# %%
import matplotlib.pyplot as plt

# 设置全局字体样式和大小
plt.rcParams['font.family'] = 'Times New Roman'  # 设置字体为 Times New Roman
plt.rcParams['font.size'] = 7  # 设置全局字体大小为 7

# 数据
x = [1, 2, 3, 4, 5]
y = [2, 3, 5, 7, 11]

# 创建图形
plt.figure(figsize=(8, 6))

# 绘制折线图
plt.plot(x, y, marker='o', linestyle='-', color='b', label='Line 1')

# 添加标题和标签
plt.title('Example Line Plot', fontsize=7)  # 设置标题字体大小
plt.xlabel('X-axis', fontsize=7)  # 设置 X 轴标签字体大小
plt.ylabel('Y-axis', fontsize=7)  # 设置 Y 轴标签字体大小

# 添加图例
plt.legend(fontsize=7)  # 设置图例字体大小

# 保存为 PDF 文件
plt.savefig('line_plot.pdf', format='pdf', bbox_inches='tight')  # 保存为 PDF 格式，确保布局紧凑

# 显示图形（可选）
plt.show()
# %%
import matplotlib.pyplot as plt

# 设置字体和字号
plt.rcParams['font.family'] = 'Times New Roman'  # 设置字体为 Times New Roman
plt.rcParams['font.size'] = 7  # 设置全局字体大小为 7
plt.rcParams['figure.dpi'] = 300  # 设置图像的分辨率（DPI），可以调整为更高值以确保清晰度

# 将厘米转换为英寸（1英寸 = 2.54厘米）
width_cm = 7  # 宽度为4厘米
height_cm = 4  # 高度为3厘米
width_inch = width_cm / 2.54  # 宽度转换为英寸
height_inch = height_cm / 2.54  # 高度转换为英寸

# 创建图形并设置大小
plt.figure(figsize=(width_inch, height_inch))  # 设置图形大小为高3厘米，宽4厘米

# 数据
x = [10, 20, 40, 60, 80, 100, 120]
y1 = [81.8, 82.0, 82.2, 82.2,82.6, 82.8, 82.4]
y2 = [67.3, 66.5, 68.1, 68.4,68.3, 68.5, 68.6]
# 绘制折线图
plt.plot(x, y1, marker='o', linestyle='-', color='b', label='Line 1')
plt.plot(x, y2, marker='x', linestyle='-', color='r', label='Line 1')

plt.ylim(66, 83)
# 添加标题和标签
# plt.title('Example Line Plot', fontsize=7)  # 设置标题字体大小
plt.xlabel('Mask Tokens', fontsize=7)  # 设置 X 轴标签字体大小
plt.ylabel('Retrieval Performance R@1', fontsize=7)  # 设置 Y 轴标签字体大小

# 添加图例
plt.legend(fontsize=7)  # 设置图例字体大小

# 调整布局
plt.tight_layout()  # 自动调整布局，确保标签和标题不被裁剪

# 保存为 PDF 文件
plt.savefig('line_plot.pdf', format='pdf', bbox_inches='tight')  # 保存为 PDF 格式，确保布局紧凑

# 显示图形（可选）
plt.show()
# %%
# mask tokens
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

colors = ['#F59191', '#BAB6DA', '#C1FF33']


# 数据
x = [10, 20, 40, 60, 80, 100, 120]
y1 = [81.8, 82.0, 82.2, 82.2, 82.6, 82.8, 82.4]
y2 = [67.3, 66.5, 68.1, 68.4, 68.3, 68.5, 68.6]

# 设置字体和字号
plt.rcParams['font.family'] = 'Times New Roman'  # 设置字体为 Times New Roman
plt.rcParams['font.size'] = 7  # 设置全局字体大小为 7
plt.rcParams['figure.dpi'] = 300  # 设置图像的分辨率

# 将厘米转换为英寸（1英寸 = 2.54厘米）
width_cm = 7  # 宽度为7厘米
height_cm = 4  # 高度为4厘米
width_inch = width_cm / 2.54  # 宽度转换为英寸
height_inch = height_cm / 2.54  # 高度转换为英寸

# 创建图形并设置大小
fig, ax1 = plt.subplots(figsize=(width_inch, height_inch))  # 创建主轴 ax1

# 绘制第一条线（使用 ax1）
ax1.plot(x, y1, marker='o', linestyle='-', color=colors[0], label='TR R@1')
ax1.set_xlabel('Number of Learnable Tokens', fontsize=7)  # 设置 X 轴标签
ax1.set_ylabel('Text Retrieval R@1', fontsize=7)  # 设置 Y1 轴标签
ax1.tick_params(axis='y')  # 设置 Y1 轴刻度颜色

# 创建第二个 Y 轴（ax2）
ax2 = ax1.twinx()  # 创建与 ax1 共享 X 轴的第二个 Y 轴
ax2.plot(x, y2, marker='s', linestyle='--', color=colors[1], label='IR R@1')  # 绘制第二条线
ax2.set_ylabel('Image Retrieval R@1', fontsize=7)  # 设置 Y2 轴标签
ax2.tick_params(axis='y')  # 设置 Y2 轴刻度颜色

# 在每个数据点旁边显示数字
for i in range(len(x)):
    # 显示 y1 的值
    ax1.text(x[i], y1[i], f"{y1[i]:.1f}", fontsize=6, ha='center', va='bottom')
    # 显示 y2 的值
    ax2.text(x[i], y2[i], f"{y2[i]:.1f}", fontsize=6, ha='center', va='top')


# 设置两个 Y 轴的刻度格式为一位小数
ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))

ax1.set_ylim(81.3, 83.7) 
ax2.set_ylim(65.8, 69.3) 

ax1.set_xticks(x)  # 设置横坐标刻度位置
ax1.set_xticklabels(x, fontsize=6, ha='right')  # 设置横坐标标签

# 添加图例
ax1.legend(bbox_to_anchor=(0.50, 0.25), loc='upper left', fontsize=7)  # 第一条线的图例
ax2.legend(bbox_to_anchor=(0.50, 1), loc='upper right', fontsize=7)  # 第二条线的图例

# 调整布局
plt.tight_layout()  # 自动调整布局

# 保存为 PDF 文件
plt.savefig('line_plot_with_twin_axes.pdf', format='pdf', bbox_inches='tight')

# 显示图形
plt.show()
# %%
# %%
# decoder depth 
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

colors = ['#F59191', '#BAB6DA', '#C1FF33']


# 数据
x = [1, 2, 3, 4, 5, 6]
y1 = [82.0, 82.4, 82.5, 82.8, 82.3, 81.9]
y2 = [67.9, 68.1, 68.3, 68.5, 68.5, 68.0]

# 设置字体和字号
plt.rcParams['font.family'] = 'Times New Roman'  # 设置字体为 Times New Roman
plt.rcParams['font.size'] = 7  # 设置全局字体大小为 7
plt.rcParams['figure.dpi'] = 300  # 设置图像的分辨率

# 将厘米转换为英寸（1英寸 = 2.54厘米）
width_cm = 7  # 宽度为7厘米
height_cm = 4  # 高度为4厘米
width_inch = width_cm / 2.54  # 宽度转换为英寸
height_inch = height_cm / 2.54  # 高度转换为英寸

# 创建图形并设置大小
fig, ax1 = plt.subplots(figsize=(width_inch, height_inch))  # 创建主轴 ax1

# 绘制第一条线（使用 ax1）
ax1.plot(x, y1, marker='o', linestyle='-', color=colors[0], label='TR R@1')
ax1.set_xlabel('Decoder Depth', fontsize=7)  # 设置 X 轴标签
ax1.set_ylabel('Text Retrieval R@1', fontsize=7)  # 设置 Y1 轴标签
ax1.tick_params(axis='y')  # 设置 Y1 轴刻度颜色

# 创建第二个 Y 轴（ax2）
ax2 = ax1.twinx()  # 创建与 ax1 共享 X 轴的第二个 Y 轴
ax2.plot(x, y2, marker='s', linestyle='--', color=colors[1], label='IR R@1')  # 绘制第二条线
ax2.set_ylabel('Image Retrieval R@1', fontsize=7)  # 设置 Y2 轴标签
ax2.tick_params(axis='y')  # 设置 Y2 轴刻度颜色

# 在每个数据点旁边显示数字
for i in range(len(x)):
    # 显示 y1 的值
    ax1.text(x[i], y1[i], f"{y1[i]:.1f}", fontsize=6, ha='center', va='bottom')
    # 显示 y2 的值
    ax2.text(x[i], y2[i], f"{y2[i]:.1f}", fontsize=6, ha='center', va='top')

# 设置两个 Y 轴的刻度格式为一位小数
ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))

ax1.set_ylim(80.8, 84.3) 
ax2.set_ylim(65.8, 69.3) 

ax1.set_xticks(x)  # 设置横坐标刻度位置
ax1.set_xticklabels(x, fontsize=6, ha='right')  # 设置横坐标标签

# 添加图例
ax1.legend(bbox_to_anchor=(0.50, 0.25), loc='upper left', fontsize=6)  # 第一条线的图例
ax2.legend(bbox_to_anchor=(0.50, 1), loc='upper right', fontsize=6)  # 第二条线的图例


# 调整布局
plt.tight_layout()  # 自动调整布局

# 保存为 PDF 文件
plt.savefig('decoder_depth.pdf', format='pdf', bbox_inches='tight')

# 显示图形
plt.show()


# %%
import matplotlib.pyplot as plt
import numpy as np

# 数据
categories = ['4 Heads', '8 Heads', '12 Heads']  # 类别
group1 = [82.8, 82.6, 82.6]  # 第一组数据
group2 = [68.5, 68.3, 68.4]  # 第二组数据

# 将厘米转换为英寸（1英寸 = 2.54厘米）
width_cm = 7  # 宽度为7厘米
height_cm = 4  # 高度为4厘米
width_inch = width_cm / 2.54  # 宽度转换为英寸
height_inch = height_cm / 2.54  # 高度转换为英寸

# 创建图形并设置大小
fig, ax = plt.subplots(figsize=(width_inch, height_inch))

# 设置柱状图的位置和宽度
y = np.arange(len(categories))  # 类别的位置
total_height = 0.4  # 两组柱子的总高度
bar_height = total_height / 2  # 每个柱子的高度
group_gap = 0.05  # 组内间距

# 绘制水平分组柱状图
bar1 = ax.barh(y - bar_height/2 - group_gap/2, group1, bar_height, label='TR R@1', color='#F59191')
bar2 = ax.barh(y + bar_height/2 + group_gap/2, group2, bar_height, label='IR R@1', color='#BAB6DA')


# 在每个柱子上写出数字
for i, (b, val) in enumerate(zip(bar1, group1)):
    ax.text(b.get_width() + 0.5, b.get_y() + b.get_height() / 2, str(val), ha='left', va='center', fontsize=6)
for i, (b, val) in enumerate(zip(bar2, group2)):
    ax.text(b.get_width() + 0.5, b.get_y() + b.get_height() / 2, str(val), ha='left', va='center', fontsize=6)

ax.set_xlim(10, 92)
# 添加标题和标签
plt.xlabel('Retrieval Performance (R@1)', fontsize=7)
# plt.ylabel('Decoder Head', fontsize=7)

# 设置刻度字体大小
plt.yticks(y, categories, fontsize=7)  # 确保纵坐标显示 categories 的值
plt.xticks(fontsize=7)

# 添加图例
plt.legend(fontsize=6)

# 调整布局
plt.tight_layout()
# 保存为 PDF 文件
plt.savefig('decoder_head.pdf', format='pdf', bbox_inches='tight')
# 显示图形
plt.show()

#%%# decoder heads 
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

colors = ['#F59191', '#BAB6DA', '#C1FF33']


# 数据
x = [1, 2, 3, 4, 5, 6]
y1 = [82.0, 82.4, 82.5, 82.8, 82.3, 81.9]
y2 = [67.9, 68.1, 68.3, 68.5, 68.5, 68.0]

# 设置字体和字号
plt.rcParams['font.family'] = 'Times New Roman'  # 设置字体为 Times New Roman
plt.rcParams['font.size'] = 7  # 设置全局字体大小为 7
plt.rcParams['figure.dpi'] = 300  # 设置图像的分辨率

# 将厘米转换为英寸（1英寸 = 2.54厘米）
width_cm = 7  # 宽度为7厘米
height_cm = 4  # 高度为4厘米
width_inch = width_cm / 2.54  # 宽度转换为英寸
height_inch = height_cm / 2.54  # 高度转换为英寸

# 创建图形并设置大小
fig, ax1 = plt.subplots(figsize=(width_inch, height_inch))  # 创建主轴 ax1

# 绘制第一条线（使用 ax1）
ax1.plot(x, y1, marker='o', linestyle='-', color=colors[0], label='TR R@1')
# ax1.set_xlabel('Decoder Depth', fontsize=7)  # 设置 X 轴标签
ax1.set_ylabel('Text Retrieval R@1', fontsize=7)  # 设置 Y1 轴标签
ax1.tick_params(axis='y')  # 设置 Y1 轴刻度颜色

# 创建第二个 Y 轴（ax2）
ax2 = ax1.twinx()  # 创建与 ax1 共享 X 轴的第二个 Y 轴
ax2.plot(x, y2, marker='s', linestyle='--', color=colors[1], label='IR R@1')  # 绘制第二条线
ax2.set_ylabel('Image Retrieval R@1', fontsize=7)  # 设置 Y2 轴标签
ax2.tick_params(axis='y')  # 设置 Y2 轴刻度颜色

# 在每个数据点旁边显示数字
for i in range(len(x)):
    # 显示 y1 的值
    ax1.text(x[i], y1[i], f"{y1[i]:.1f}", fontsize=6, ha='center', va='bottom')
    # 显示 y2 的值
    ax2.text(x[i], y2[i], f"{y2[i]:.1f}", fontsize=6, ha='center', va='top')

# 设置两个 Y 轴的刻度格式为一位小数
ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))

ax1.set_ylim(80.8, 84.3) 
ax2.set_ylim(65.8, 69.3) 

ax1.set_xticks(x)  # 设置横坐标刻度位置
ax1.set_xticklabels(x, fontsize=6, ha='right')  # 设置横坐标标签

# 添加图例
ax1.legend(bbox_to_anchor=(0.50, 0.25), loc='upper left', fontsize=6)  # 第一条线的图例
ax2.legend(bbox_to_anchor=(0.50, 1), loc='upper right', fontsize=6)  # 第二条线的图例


# 调整布局
plt.tight_layout()  # 自动调整布局

# 保存为 PDF 文件
plt.savefig('decoder_depth.pdf', format='pdf', bbox_inches='tight')

# 显示图形
plt.show()

#%%
# 打开文件并统计单词总数
def count_words_in_file(file_path):
    try:
        # 初始化单词计数器
        word_count = 0

        # 打开文件
        with open(file_path, 'r', encoding='utf-8') as file:
            # 逐行读取文件内容
            for line in file:
                # 使用空格分割单词，并统计每行的单词数量
                words = line.split()
                word_count += len(words)

        # 返回单词总数
        return word_count

    except FileNotFoundError:
        print(f"文件 {file_path} 未找到，请检查文件路径是否正确。")
        return None
    except Exception as e:
        print(f"读取文件时发生错误：{e}")
        return None


# 调用函数并打印结果
file_path = 'data/coco/train_caps.txt'  # 替换为你的文件路径
result = count_words_in_file(file_path)
if result is not None:
    print(f"文件 {file_path} 中的单词总数为：{result}")
# %%
