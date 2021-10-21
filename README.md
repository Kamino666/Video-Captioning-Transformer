# Video Captioning Transformer

## 引言

这是一个使用Pytorch实现的**视频描述生成**深度学习模型。

视频描述生成任务就是：输入一个视频，可以输出一句描述整个视频内容的文字（当视频太长或较复杂时效果可能就很差了，针对长视频，目前有密集视频描述生成任务，即Dense Video Captioning，本项目暂时不涉及）。本项目主要目的是帮助视力障碍者欣赏网络视频、感知周围环境，促进“无障碍视频”的发展。

本项目实现了：

- [x] 从视频到英文文字描述的全流程
- [x] 预训练模型的提供
- [x] 预训练模型的训练参数、代码等
- [x] 一些样例
- [x] 视频描述生成的微信小程序Demo（由于缺乏算力和预算，暂不开放）

本项目尚未实现但预计实现的：

- [ ] 从视频到中文文字及语音播报的全流程
- [ ] 密集视频描述生成
- [ ] 模型训练的可视化
- [ ] 论文
- [ ] 项目主页

**注意！：本项目拥有版权，遵守Apache-2.0 License，详情请看库内LICENSE文件。但是不包括使用的数据集版权、submodule子目录下任何文件的版权。**

## 模型总结

模型基于Transformer架构，对结构暂未做太大的改变（如下图）。视频输入经过抽帧之后，按帧提取CLIP特征，同时文本侧使用BERT的Embedding层权重，之后两路特征送入Transformer，输出概率使用SCE loss计算损失。

![中文baseline](https://kamino-img.oss-cn-beijing.aliyuncs.com/20211016133947.png)

## 开始

### 环境

本文实验运行环境为 Ubuntu 16.04，平台为Python3.7.0，使用 Pytorch 深度学习框架，使用一张 GTX TITAN X显卡进行训练（假如要使用coco eval评估则需要按照java jre）。使用的Python库如下：

```
torch
transformers
tensorboardX
tqdm
timeit
mmcv
numpy
pathlib
```

### 预训练模型

[百度网盘: juyy](https://pan.baidu.com/s/1LCRpK_HCxWxqNkY6KsRdvw) 	 [Google Drive](https://drive.google.com/drive/folders/1YDEJ0hxlj3F887jaOQ2ymSc-HskSqOFQ?usp=sharing)

模型命名方式为`[数据集名字]-[一些训练配置]-[模型大小base]`，base模型大概400MB。

| 模型名字                 | 模型描述              | METEOR指标 |
| ------------------------ | --------------------- | ---------- |
| M-CLIP-SCEloss-BERT-base | 使用SCEloss和BERT权重 | 28.7       |
| M-CLIP-SCEloss-base      | 使用SCEloss           | **28.8**   |

### 数据集准备

我们使用MSR-VTT数据集，由于版权原因不放出原视频，我们提供提取好的[CLIP特征 oyrt](https://pan.baidu.com/s/1mNFhymugYV58Z55F--e9cA)。

提取CLIP特征既可以使用[官方CLIP库](https://github.com/openai/CLIP)，也可以使用[Kamino666/video_features](https://github.com/Kamino666/video_features)，提取特征抽帧为fps3，resize为224*224，使用CLIP的CLIP-ViT-B/32版本。

## 训练

```bash
CUDA_VISIBLE_DEVICES=0 python train.py
```

目前配置文件存储在train.py的Opt类中。详情查看代码注释。

## 评估

```bash
CUDA_VISIBLE_DEVICES=2 python eval_batch_video.py  # 需要java
```

同样，配置在eval_batch_video.py的EvalOpt类中。

## 单视频预测

```bash
python test_video.py \
	-v 预测视频路径 \
    -m ./checkpoint/模型名字 \
    --feat CLIP \
    --fps 2 \
    --gpu
```

 除了上述参数，在test_video.py里还可以找到更多说明。

## 效果展示

MSR-VTT数据集内视频：

| ![1](https://kamino-img.oss-cn-beijing.aliyuncs.com/20211016150236.png) | ![2](https://kamino-img.oss-cn-beijing.aliyuncs.com/20211016150241.png) |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![3](https://kamino-img.oss-cn-beijing.aliyuncs.com/20211016150245.png) | ![4](https://kamino-img.oss-cn-beijing.aliyuncs.com/20211016150246.png) |
| ![6](https://kamino-img.oss-cn-beijing.aliyuncs.com/20211016150258.png) | ![7](https://kamino-img.oss-cn-beijing.aliyuncs.com/20211016150306.png) |

Bilibili上随机搜索得到的视频：

暂无，正在测试中。

## 致谢

[openai/CLIP: Contrastive Language-Image Pretraining (github.com)](https://github.com/openai/CLIP)

[v-iashin/video_features: Extract video features from raw videos using multiple GPUs. We support RAFT and PWC flow frames as well as I3D, R(2+1)D, VGGish, ResNet features. (github.com)](https://github.com/v-iashin/video_features)

[salaniz/pycocoevalcap: Python 3 support for the MS COCO caption evaluation tools (github.com)](https://github.com/salaniz/pycocoevalcap)



