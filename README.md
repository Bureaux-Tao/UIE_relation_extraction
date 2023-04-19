# 基于UIE的小样本中文肺部CT病历实体关系抽取方法

## Dataset

### 数据集格式

本文中标注完的数据可以是以下几种

#### 命名实体识别数据格式

- BIO格式

```
胡 B-PER
锦 I-PER
涛 I-PER
强 O
调 O
， O
中 B-ORG
国 I-ORG
共 I-ORG
产 I-ORG
党 I-ORG
愿 O
在 O
党 O
际 O
关 O
系 O
四 O
项 O
原 O
则 O
的 O
基 O
础 O
上 O
```

#### 关系数据集格式

- SPO数据格式

```json
{
    "text": "右肺中叶实性结节，考虑良性，随访。",
    "spo_list": [{
        "object": "右肺中叶",
        "object_type": "部位与器官",
        "predicate": "部位与器官",
        "subject": "实性结节",
        "subject_type": "疾病与症状"
    }, {
        "object": "良性",
        "object_type": "性质",
        "predicate": "性质",
        "subject": "实性结节",
        "subject_type": "疾病与症状"
    }]
}
```

- Doccano数据

```json
{
    "id": 33834,
    "text": "右肺中叶实性结节，考虑良性，随访。",
    "entities": [{
        "id": 101477,
        "label": "部位与器官",
        "start_offset": 0,
        "end_offset": 4
    }, {
        "id": 101478,
        "label": "疾病与症状",
        "start_offset": 4,
        "end_offset": 8
    }, {
        "id": 101479,
        "label": "性质",
        "start_offset": 11,
        "end_offset": 13
    }],
    "relations": [{
        "id": 13326,
        "from_id": 101478,
        "to_id": 101477,
        "type": "部位与器官"
    }, {
        "id": 13327,
        "from_id": 101478,
        "to_id": 101479,
        "type": "性质"
    }]
}
```

- 标注精灵导出格式

```json
{
    "path": "D:\\研究生相关\\平常工作相关\\关系抽取\\原始数据+新增数据.txt",
    "outputs": {
        "annotation": {
        	"T": [{
                "type": "T",
                "name": "部位与器官",
                "value": "右肺下叶外基底段",
                "start": 20,
                "end": 28,
                "attributes": [],
                "id": 21
            }, {
                "type": "T",
                "name": "部位与器官",
                "value": "左肺下叶外基底段",
                "start": 29,
                "end": 37,
                "attributes": [],
                "id": 22
            },	...	],
            "E": [""],
            "R": [{
                "name": "部位与器官",
                "from": 23,
                "to": 21,
                "arg1": "Arg1",
                "arg2": "Arg2"
            }, {
                "name": "部位与器官",
                "from": 23,
                "to": 22,
                "arg1": "Arg1",
                "arg2": "Arg2"
            }, {
                "name": "直径",
                "from": 23,
                "to": 24,
                "arg1": "Arg1",
                "arg2": "Arg2"
            },	...	],
            "A": [""]
        }
    },
    "time_labeled": 1648025867327,
    "labeled": true,
    "content": "PICC置管中。两肺纹理清晰，走向自然，右肺下叶外基底段、左肺下叶外基底段均可见密度增高结节影，最大径分别为8.5mm、7.5mm，所见各支气管腔通畅 ... "
}
```

- Label Studio格式

```
略
```

## Project Structure

```
./
├── README.md
├── __pycache__
├── bert4torch                      bert4torch包，也可以pip下载
│   ├── __init__.py
│   ├── __pycache__
│   ├── activations.py
│   ├── layers.py
│   ├── losses.py
│   ├── models.py
│   ├── optimizers.py
│   ├── snippets.py
│   └── tokenizers.py
├── checkpoints                     保存的权重
│   └── placeholder.pt
├── convert.py                      权重转换脚本
├── convert_weights                 转换的权重
│   └── uie_medical_base_pytorch
│       ├── config.json
│       ├── pytorch_model.bin
│       ├── special_tokens_map.json
│       ├── tokenizer_config.json
│       └── vocab.txt
├── data
│   ├── doccano_data                Doccano格式数据
│   │   └── placeholder.jsonl
│   ├── ori_data                    原始格式数据
│   └── target_data                 转换后目标格式数据
│       └── placeholder.json
├── data_convert                    数据格式转换脚本
│   ├── __init__.py
│   ├── bio2doccano.py
│   ├── colabeler2doccano.py
│   ├── doccano2prompt.py
│   ├── doccano2spo.py
│   ├── doccano_test.py
│   ├── labelstudio2doccano.py
│   └── spo2doccano.py
├── evaluate.py                     模型性能评估脚本
├── finetune.sh                     流程脚本
├── history                         训练历史数据
│   └── lung_uie_mcc_vat.csv
├── log
│   ├── F1                          各类F1值
│   │   └── lung_uie_mcc_vat.out
│   ├── logs                        训练日志
│   │   └── lung_uie_mcc_vat.log
│   └── outs                        nohup输出文件
│       └── lung_uie_mcc_vat.out
├── model.py                        模型文件
├── predict.py
├── statistic.py                    数据集文本长度统计脚本
├── tensorboard                     Tensorboard输出脚本
│   └── lung_uie_mcc_vat
│       ├── events.out.tfevents.1668431748.wQeGaW
│       ├── events.out.tfevents.1668431789.wQeGaW
│       ├── events.out.tfevents.1668431833.wQeGaW
│       └── events.out.tfevents.1668486982.P42V0J
├── train.py                        训练文件
└── utils.py                        工具文件
```

## Requirements

```
colorama==0.4.5
colorlog==6.7.0
ipykernel==6.15.2
matplotlib==3.5.3
numpy==1.23.3
pandas==1.4.4
Python==3.9.13
tensorboardX==2.5.1
torch==1.12.1
torchinfo==1.7.0
tqdm==4.64.1
```

## Model

用pytorch改写的paddle版UIE，不同于论文原版，采用抽取式框架，上游底座是ERNIE 3.0，下游为MRC-Span式指针抽取框架的简化版，只有首尾指针，没有span matrix的首尾指针匹配预测，所以可以解决组间嵌套和组内嵌套问题，但无法解决组内嵌套。上下游都有预训练权重，所以不需要分层学习率。

![167236006-66ed845d-21b8-4647-908b-e1c6e7613eb1.png](images%2F167236006-66ed845d-21b8-4647-908b-e1c6e7613eb1.png)

该模型可以实现零样本快速冷启动，并具备优秀的小样本微调能力，快速适配特定的抽取目标，大幅度降低标注数据依赖，在降低成本的同时，还提升了效果。

该模型为通用抽取框架，MRC模型具备知识迁移能力，使用同一套权重可以同时抽取实体、关系、事件等。

## Steps

### 数据标注

使用Docker方式进行部署

拉取镜像

```shell
docker pull doccano/doccano
```

创建容器并初始化

```shell
docker container create --name doccano \\
  -e "ADMIN_USERNAME=[xxx]" \\
  -e "ADMIN_EMAIL=[xxx@xxx.com]" \\
  -e "ADMIN_PASSWORD=[password]" \\
  -v doccano-db:/data \\
  -p 8000:8000 doccano/doccano
```

启动容器

```shell
docker container start doccano
```

打开 `http://127.0.0.1:8000/` 即可看到

Doccano的使用方法：[链接](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/model_zoo/uie/doccano.md)

**输出请选择`JSONL`格式。**

### 数据转换

需要将上文中所提到的其他格式的数据先转换成Doccano的输出格式的数据

- BIO格式转Doccano格式：`data_convert/bio2doccano.py`

```python
preprocess(["xxx.txt", "xxx.txt",...],    # 多个BIO格式文件的路径
           "../data/doccano_data/",       # 导出文件路径
           "xxx.jsonl")                   # 导出文件名
```

- 标注精灵格式转Doccano格式：`data_convert/colabeler2doccano.py`

```shell
python colabeler2doccano.py \
--colabeler_file ../data/export.json \    # 标注精灵文件路径
--doccano_file ../data/doccano.jsonl      # Doccano输出文件路径
```

- Label Studio格式转Doccano格式：`data_convert/labelstudio2doccano.py`

```shell
python labelstudio2doccano.py \
--labelstudio_file ../data/export.json \  # 标注精灵文件路径
--doccano_file ../data/doccano.jsonl      # Doccano输出文件路径
--task_type ext                           # 任务类型，抽取类任务选择"ext"
```

-SPO格式转Doccano格式：`data_convert/spo2doccano.py`

```shell
python spo2doccano.py \
--spo_file ../data/export.json \          # SPO文件路径
--doccano_file ../data/doccano.jsonl      # Doccano输出文件路径
```

### 构造prompt输入数据

#### 实体抽取

例：当前数据集包含4种实体类型：出发地、目的地、费用、时间，若待抽取文本中包含时间、费用两个实体类型，则需要构造其他两个类型的负例数据（出发地、目的地）。

```
# 文本
1月22日交通费33元拜访客户

# 正样本
prompt: 时间
result_list: [{"text": "1月22日", "start": 0, "end": 5}]
prompt: 费用
result_list: [{"text": "33", "start": 8, "end": 10}]

# 负样本
prompt: 出发地
result_list: []
prompt: 目的地
result_list: []
```

#### 关系抽取

UIE中关系抽取任务分为两阶段。第一阶段的负例构造方式和实体抽取相同，主要说明第二阶段的负例构造方法。

若第二阶段的prompt为A的B，负例包含以下三种形式：

1. 反关系负例（如：b的B，b为A的B的抽取结果）
2. 随机替换A负例（如：C的B，C为文本中不包含的实体）
3. 随机替换B负例（如：A的D，D为文本中不包含的实体类型）

```
# 文本
写申玲，就一定要写申玲的老公——王玉平

# 正样本
prompt: 申玲的老公
result_list: [{"text": "王玉平", "end": 19, "start": 16}]
prompt: 王玉平的老婆
result_list: [{"text": "申玲", "end": 3, "start": 1}]

# 负样本
1) 反关系负例
prompt: 王玉平的老公
result_list: []

2）随机替换A负例
prompt: 刘英俊的老公
result_list: []

3）随机替换B负例
prompt: 申玲的毕业学校
result_list: []
```

在UIE训练定制中，训练时需要加入一定比例的负例，验证时不加入。正负样本的比例十分重要，如果数据集只有正样本容易造成错误召回，而如果负样本所占比例过多也可能造成模型不召回，因此数据集构造过程中需要控制好正负样本的比例。

转换脚本：`data_convert/doccano2prompt.py`

```shell
python doccano2prompt.py \
--doccano_file ../data/doccano_data/xxx.jsonl \   # doccano_file文件路径
--task_type ext \                                 # 任务类型，抽取类任务选择"ext"
--save_dir ../data/target_data/ \                 # 保存目录
--negative_ratio 5 \                              # 负样本比例
--splits 0.6 0.2 0.2                              # 训练集、验证集、测试集划分比例
```

### 转换权重

需要将原paddle权重转换成pytorch权重

```shell
python convert.py\
--input_model uie-micro \                         # 可以下载paddle的uie模型权重进行转换，此项为paddle的uie模型路径，默认[uie-base/uie-tiny]
--output_model convert_weights/uie-micro_pytorch  # 输出转换后的权重路径
```

支持自动下载的模型，不用路径，可直接在`input_model`里填名称

- uie-base
- uie-medium
- uie-mini
- uie-micro
- uie-nano
- uie-medical-base
- uie-tiny


### 微调训练

```shell
python train.py \
--batch_size 32 \                                         # batch size
--learning_rate 1e-5 \                                    # 学习率，建议小于等于1e-5
--train_path ./data/target_data/lungCT/train.txt \        # 转换后的训练集路径
--dev_path ./data/target_data/lungCT/dev.txt \            # 转换后的测试集路径
--max_seq_len 256 \                                       # max len，一个批次内小于max_seq_len则该批次填充到批次内最长，若一个批次内大于max_seq_len则填充或截取到max_seq_len
--num_epochs 20 \                                         # 最大训练epoch
--shots 0 \                                               # 取多少条训练集数据进行训练，0为全部
--early_stop_patience 5 \                                 # 早停轮次
--best_weight_path ./checkpoints/lung_uie_untrained.pt \  # 保存最佳权重路径
--train_history_path ./history/lung_uie_untrained.csv \   # 训练损失、F1记录路径
--log_path ./log/logs/lung_uie_untrained.log \            # 训练日志路径
--tensorboard_dir ./tensorboard/lung_uie_untrained/ \     # tensorboard路径
--steps_per_epoch 500                                     # 每轮训练步数，默认训练集条数/batch size
```

## 优化策略

### 损失函数

损失函数使用[多标签交叉熵](https://kexue.fm/archives/7359)，可以标签平衡0,1不平衡问题，增加模型性能

```python
class MultilabelCategoricalCrossentropy(nn.Module):
    """多标签分类的交叉熵
    说明：y_true和y_pred的shape一致，y_true的元素非0即1， 1表示对应的类为目标类，0表示对应的类为非目标类。
    警告：y_pred是输出的得分，不是概率
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def forward(self, y_pred, y_true):
        y_pred = (1 - 2 * y_true) * y_pred
        y_pred_pos = y_pred - (1 - y_true) * 1e12
        y_pred_neg = y_pred - y_true * 1e12
        
        y_pred_pos = torch.cat([y_pred_pos, torch.zeros_like(y_pred_pos[..., :1])], dim = -1)
        y_pred_neg = torch.cat([y_pred_neg, torch.zeros_like(y_pred_neg[..., :1])], dim = -1)
        pos_loss = torch.logsumexp(y_pred_pos, dim = -1)
        neg_loss = torch.logsumexp(y_pred_neg, dim = -1)
        return (pos_loss + neg_loss).mean()
```

### 对抗训练

对抗训练使用[虚拟对抗的方式（VAT）](https://arxiv.org/abs/1704.03976v1)，VAT更适合在标签嘈杂、数据较少，且还有很多未标签数据的情况下使用，而且比传统的对抗训练更具优势。

实验表明VAT对模型有巨大的提升，并且可以平衡precision和recall相差过大的问题。

## Task Introduction

本课题数据集来源于企业实践中所做的开放性肺结节医学影像标准化数据库建设项目，此项目由上海长征医院、华东医院、上海市公共卫生临床中心、中国食品药品检定研究院、上海市计算技术研究所等机构参与，不开源。

本文主要是针对肺结节电子病例中的结构化，利用实体识别技术从影像所见和诊断描述中获取描述肺结节的关键信息，本任务中，可能为主实体类别包括：

1. 疾病与症状：指导致患者处于非健康状态的原因，或者医生根据患者的身体状况作出的诊断，与疾病导致的不适或异常感觉和异常检查结果。如：右肺上叶后段见一实性结节（疾病为“实性结节”），边缘不清。
2. 部位与器官：影像所见描述的异常或病变的身体具体部位或器官。如：左肺上叶（部位为“左肺上叶”）见一磨玻璃密度结节影。

主对应的父子关系包括：

1. 部位与器官：即异常与疾病所在的部位。主体为疾病与症状。如：“右肺下叶后基底段见实变影”，疾病“实变影”的器官即为“右肺下叶后基底段”。
2. 数量：肺部疾病、异常所见的数目信息。主体为疾病与症状。如：“两下肺少量间质炎症”，疾病“间质炎症”的数量为“少量”。
3. 纹理：对部位表面样子的描述。主体为病灶部位。如：“两肺纹理增粗紊乱”，器官“两肺”的纹理为“增粗紊乱”。
4. 疾病密度：描述病灶物质的CT值高低。主体为疾病与症状。如：“右残肺结节、团块灶，大者长径约5.4cm，密度不均”，疾病“结节”、“团块灶”的密度为“不均”。
5. 部位密度：检查所见中各器官的CT值高低，均匀、不均匀等。主体为病灶部位。如：“甲状腺右叶密度不均”，部位“甲状腺右叶”的密度为“不均”。
6. 边界：病变和周边的组织之间的界限，清楚或不清楚。主体为疾病与症状。如：“左肺下叶见磨玻璃密度小结节，边界欠清”，疾病“磨玻璃密度小结节”的边界为“欠清”。
7. 边缘：异常部分边缘异常情况。主体为疾病与症状。如：“实性结节，边缘不清”，疾病“实性结节”的边缘为“不清”。
8. 直径：异常部分的直径大小。主体为疾病与症状。如：“实性结节，边缘不清，直径6.0mm”，疾病“实性结节”的直径为“6.0mm”。
9. 疾病形态：描述病灶的形状、形态。主体为疾病与症状。如：“右肺见数枚结节影，部分边界不清，形态不规则”，异常“结节影”的形态为“不规则”。
10. 部位形态：描述部位或器官的形状、形态。主体为病灶部位。如：“两侧肱骨头形态欠光整”，部位“两侧肱骨头”的形态为“欠光整”。
11. 性质：肺结节性质，恶行或者良性。主体为疾病与症状。如：“左下肺结节与前片CT相仿，考虑良性”，病灶“结节”的性质为“良性”。
12. 异常所见：关于病灶的一些其他描述。主体为疾病与症状。如：“左肺下叶前内基底段肺门旁见一枚结节灶，边缘光整，增强可见不均匀轻度强化”，疾病“结节灶”的异常所见为“不均匀轻度强化”。

可结构化成树形结构：

```json
{
    "疾病与症状": ["部位与器官", "数量", "密度", "边界", "边缘", "直径", "病灶形态", "性质", "异常所见"],
    "部位与器官": ["纹理", "部位形态", "部位密度"]
}
```

数据示例：

![screenshot2023-01-03-18.37.10.png](images%2Fscreenshot2023-01-03-18.37.10.png)

本数据集中，样本一共1094例。模型的截断比例设置为256。数据分布如下图所示：

![dataset.jpg](images%2Fdataset.jpg)

## Train

```
=============Start Training=============
2022-11-15 04:36:22 - Epoch: 1/99
[2022-11-15 04:36:22,589] [    INFO] - model runs on cuda:0
600/600 [==============================] - 728s 1s/step - loss: 3.5382 - loss_sup: 1.2261 - loss_unsup: 2.3122
[2022-11-15 04:50:47,639] [    INFO] - [val-entity level] f1: 0.18665, p: 0.94306 r: 0.10358

2022-11-15 04:50:47 - Epoch: 2/99
600/600 [==============================] - 732s 1s/step - loss: 1.8561 - loss_sup: 0.6524 - loss_unsup: 1.2037
[2022-11-15 05:05:16,576] [    INFO] - [val-entity level] f1: 0.53071, p: 0.96203 r: 0.36643
2022-11-15 05:05:16 - Epoch: 3/99

600/600 [==============================] - 733s 1s/step - loss: 1.5525 - loss_sup: 0.5408 - loss_unsup: 1.0116
[2022-11-15 05:19:47,174] [    INFO] - [val-entity level] f1: 0.88664, p: 0.92139 r: 0.85441

        ... ...
        
2022-11-15 07:43:47 - Epoch: 14/99
600/600 [==============================] - 732s 1s/step - loss: 0.7400 - loss_sup: 0.2449 - loss_unsup: 0.4951
[2022-11-15 07:58:06,295] [    INFO] - [val-entity level] f1: 0.91748, p: 0.91261 r: 0.92242

2022-11-15 07:58:06 - Epoch: 15/99
600/600 [==============================] - 726s 1s/step - loss: 0.6375 - loss_sup: 0.2086 - loss_unsup: 0.4289
[2022-11-15 08:12:18,689] [    INFO] - [val-entity level] f1: 0.91770, p: 0.90625 r: 0.92945

2022-11-15 08:12:18 - Epoch: 16/99
600/600 [==============================] - 728s 1s/step - loss: 0.6534 - loss_sup: 0.2102 - loss_unsup: 0.4432
[2022-11-15 08:26:33,096] [    INFO] - [val-entity level] f1: 0.91578, p: 0.91039 r: 0.92124
============Finish Training=============
Epoch 16: early stopping
```

## Evaluate

评估方式：生成的prompt测试集数据，实体和关系一起进行评估，原来一条样本评估的数量为`实体类别数+subject*关系类别数`，并非传统SPO三元组评估方式。

`evaluate.py`文件

```shell
python predict.py \
--model_path ./checkpoints/lung_uie_mcc_shots100.pt \ # 权重路径
--test_path ./data/target_data/LungCT/dev.txt \       # 测试集路径
--batch_size 128 \                                    # batch size
--max_seq_len 512 \                                   # max len
--category \                                          # 是否评估每类F1
--no_neg                                              # 评估时不使用负样本
```

## Performance

### Zero-shot

```
[2022-11-10 02:12:05,781] [    INFO] - zero_shot performance - f1:0.1089, precision:0.1089, recall:0.1089
```

### Few-shots

- 10 shots

```
2022-11-18 06:58:05 - Epoch: 12/20
23/23 [==============================] - 16s 682ms/step - loss: 0.0294
[2022-11-18 07:00:41,366] [    INFO] - [val-entity level] f1: 0.74601, p: 0.69983 r: 0.79871
```

- 20 shots

```
2022-11-18 07:38:00 - Epoch: 7/20
44/44 [==============================] - 32s 722ms/step - loss: 0.1140
[2022-11-18 07:40:49,500] [    INFO] - [val-entity level] f1: 0.77342, p: 0.70660 r: 0.85421
```

- 30 shota

```
2022-11-18 08:31:48 - Epoch: 13/20
72/72 [==============================] - 53s 734ms/step - loss: 0.0244
[2022-11-18 08:34:59,625] [    INFO] - [val-entity level] f1: 0.80984, p: 0.75640 r: 0.87141
```

- 100 shots

```
2022-11-18 09:17:53 - Epoch: 7/20
192/192 [==============================] - 146s 758ms/step - loss: 0.1080
[2022-11-18 09:22:37,770] [    INFO] - [val-entity level] f1: 0.85642, p: 0.82732 r: 0.88763
```

- full shots best

```
[2022-11-15 16:49:25,360] [    INFO] - Class Name: all_classes
[2022-11-15 16:49:25,360] [    INFO] - Precision: 0.93932 | Recall: 0.93180 | F1: 0.93554
```

### 每类F1

- 实体F1

```
[2022-11-15 10:26:21,507] [    INFO] - Entities Categories F1:
[2022-11-15 10:26:24,847] [    INFO] - -----------------------------
[2022-11-15 10:26:24,847] [    INFO] - Class Name: 部位与器官
[2022-11-15 10:26:24,847] [    INFO] - Evaluation Correct: 1192 | Infer: 1307 | Label: 1270 | Precision: 0.91201 | Recall: 0.93858 | F1: 0.92511
[2022-11-15 10:26:27,762] [    INFO] - -----------------------------
[2022-11-15 10:26:27,762] [    INFO] - Class Name: 疾病与症状
[2022-11-15 10:26:27,763] [    INFO] - Evaluation Correct: 1136 | Infer: 1246 | Label: 1244 | Precision: 0.91172 | Recall: 0.91318 | F1: 0.91245
[2022-11-15 10:26:29,460] [    INFO] - -----------------------------
[2022-11-15 10:26:29,461] [    INFO] - Class Name: 数量
[2022-11-15 10:26:29,461] [    INFO] - Evaluation Correct: 259 | Infer: 272 | Label: 263 | Precision: 0.95221 | Recall: 0.98479 | F1: 0.96822
[2022-11-15 10:26:29,644] [    INFO] - -----------------------------
[2022-11-15 10:26:29,644] [    INFO] - Class Name: 边缘
[2022-11-15 10:26:29,644] [    INFO] - Evaluation Correct: 18 | Infer: 18 | Label: 22 | Precision: 1.00000 | Recall: 0.81818 | F1: 0.90000
[2022-11-15 10:26:30,633] [    INFO] - -----------------------------
[2022-11-15 10:26:30,634] [    INFO] - Class Name: 直径
[2022-11-15 10:26:30,634] [    INFO] - Evaluation Correct: 139 | Infer: 141 | Label: 140 | Precision: 0.98582 | Recall: 0.99286 | F1: 0.98932
[2022-11-15 10:26:30,904] [    INFO] - -----------------------------
[2022-11-15 10:26:30,905] [    INFO] - Class Name: 密度
[2022-11-15 10:26:30,905] [    INFO] - Evaluation Correct: 25 | Infer: 25 | Label: 31 | Precision: 1.00000 | Recall: 0.80645 | F1: 0.89286
[2022-11-15 10:26:31,261] [    INFO] - -----------------------------
[2022-11-15 10:26:31,261] [    INFO] - Class Name: 形态
[2022-11-15 10:26:31,261] [    INFO] - Evaluation Correct: 25 | Infer: 30 | Label: 45 | Precision: 0.83333 | Recall: 0.55556 | F1: 0.66667
[2022-11-15 10:26:31,461] [    INFO] - -----------------------------
[2022-11-15 10:26:31,461] [    INFO] - Class Name: 边界
[2022-11-15 10:26:31,461] [    INFO] - Evaluation Correct: 25 | Infer: 25 | Label: 25 | Precision: 1.00000 | Recall: 1.00000 | F1: 1.00000
[2022-11-15 10:26:31,897] [    INFO] - -----------------------------
[2022-11-15 10:26:31,897] [    INFO] - Class Name: 纹理
[2022-11-15 10:26:31,897] [    INFO] - Evaluation Correct: 42 | Infer: 42 | Label: 46 | Precision: 1.00000 | Recall: 0.91304 | F1: 0.95455
[2022-11-15 10:26:31,932] [    INFO] - -----------------------------
[2022-11-15 10:26:31,932] [    INFO] - Class Name: 性质
[2022-11-15 10:26:31,932] [    INFO] - Evaluation Correct: 7 | Infer: 7 | Label: 8 | Precision: 1.00000 | Recall: 0.87500 | F1: 0.93333
[2022-11-15 10:26:31,932] [    INFO] - Relations Categories F1:
[2022-11-15 10:30:47,933] [    INFO] - -----------------------------
```

- 关系F1

```
[2022-11-15 10:30:47,933] [    INFO] - Class Name: X的部位与器官
[2022-11-15 10:30:47,933] [    INFO] - Evaluation Correct: 31425 | Infer: 32836 | Label: 33236 | Precision: 0.95703 | Recall: 0.94551 | F1: 0.95124
[2022-11-15 10:30:52,456] [    INFO] - -----------------------------
[2022-11-15 10:30:52,456] [    INFO] - Class Name: X的数量
[2022-11-15 10:30:52,456] [    INFO] - Evaluation Correct: 452 | Infer: 462 | Label: 465 | Precision: 0.97835 | Recall: 0.97204 | F1: 0.97519
[2022-11-15 10:30:52,849] [    INFO] - -----------------------------
[2022-11-15 10:30:52,849] [    INFO] - Class Name: X的边缘
[2022-11-15 10:30:52,849] [    INFO] - Evaluation Correct: 39 | Infer: 40 | Label: 46 | Precision: 0.97500 | Recall: 0.84783 | F1: 0.90698
[2022-11-15 10:31:01,514] [    INFO] - -----------------------------
[2022-11-15 10:31:01,514] [    INFO] - Class Name: X的直径
[2022-11-15 10:31:01,514] [    INFO] - Evaluation Correct: 889 | Infer: 906 | Label: 964 | Precision: 0.98124 | Recall: 0.92220 | F1: 0.95080
[2022-11-15 10:31:01,603] [    INFO] - -----------------------------
[2022-11-15 10:31:01,603] [    INFO] - Class Name: X的部位密度
[2022-11-15 10:31:01,603] [    INFO] - Evaluation Correct: 9 | Infer: 9 | Label: 9 | Precision: 1.00000 | Recall: 1.00000 | F1: 1.00000
[2022-11-15 10:31:01,603] [    INFO] - -----------------------------
[2022-11-15 10:31:01,603] [    INFO] - Class Name: X的密度
[2022-11-15 10:31:01,603] [    INFO] - Evaluation Correct: 9 | Infer: 9 | Label: 9 | Precision: 1.00000 | Recall: 1.00000 | F1: 1.00000
[2022-11-15 10:31:02,235] [    INFO] - -----------------------------
[2022-11-15 10:31:02,235] [    INFO] - Class Name: X的病灶形态
[2022-11-15 10:31:02,235] [    INFO] - Evaluation Correct: 71 | Infer: 71 | Label: 91 | Precision: 1.00000 | Recall: 0.78022 | F1: 0.87654
[2022-11-15 10:31:02,689] [    INFO] - -----------------------------
[2022-11-15 10:31:02,689] [    INFO] - Class Name: X的边界
[2022-11-15 10:31:02,689] [    INFO] - Evaluation Correct: 56 | Infer: 56 | Label: 56 | Precision: 1.00000 | Recall: 1.00000 | F1: 1.00000
[2022-11-15 10:31:02,778] [    INFO] - -----------------------------
[2022-11-15 10:31:02,778] [    INFO] - Class Name: X的异常所见
[2022-11-15 10:31:02,779] [    INFO] - Evaluation Correct: 1 | Infer: 1 | Label: 9 | Precision: 1.00000 | Recall: 0.11111 | F1: 0.20000
[2022-11-15 10:31:03,194] [    INFO] - -----------------------------
[2022-11-15 10:31:03,194] [    INFO] - Class Name: X的纹理
[2022-11-15 10:31:03,194] [    INFO] - Evaluation Correct: 40 | Infer: 41 | Label: 41 | Precision: 0.97561 | Recall: 0.97561 | F1: 0.97561
[2022-11-15 10:31:03,208] [    INFO] - -----------------------------
[2022-11-15 10:31:03,208] [    INFO] - Class Name: X的性质
[2022-11-15 10:31:03,209] [    INFO] - Evaluation Correct: 1 | Infer: 1 | Label: 2 | Precision: 1.00000 | Recall: 0.50000 | F1: 0.66667
[2022-11-15 10:31:03,248] [    INFO] - -----------------------------
[2022-11-15 10:31:03,248] [    INFO] - Class Name: X的部位形态
[2022-11-15 10:31:03,248] [    INFO] - Evaluation Correct: 2 | Infer: 2 | Label: 3 | Precision: 1.00000 | Recall: 0.66667 | F1: 0.80000
```

## Predict

`predict.py`文件，根据传入的不同schema，可预测实体或关系

### 实体预测：

实体的schema为一个列表：`['部位与器官', '疾病与症状', '数量', '纹理', '密度', '边界', '边缘', '直径', '形态','性质']`

```python
sample = "右肺下叶可见两枚实性结节，大者位于右肺下叶后基底段，长径约7mm，界清，边缘模糊，密度为纯毛玻璃样。"
    
schema = ['部位与器官', '疾病与症状', '数量', '纹理', '密度', '边界', '边缘', '直径', '形态', '性质']  # Define the schema for entity extraction
ie = UIEPredictor(schema = schema, model_weight = "./checkpoints/lung_uie_mcc_vat.pt")
result = ie(sample)
print(json.dumps(result, ensure_ascii = False))
```

输出：

```json
[{
    "部位与器官": [{
        "text": "右肺下叶",
        "start": 0,
        "end": 4,
        "probability": 27.975500106811523
    }, {
        "text": "右肺下叶后基底段",
        "start": 17,
        "end": 25,
        "probability": 26.680252075195312
    }],
    "疾病与症状": [{
        "text": "实性结节",
        "start": 8,
        "end": 12,
        "probability": 18.970672607421875
    }],
    "数量": [{
        "text": "两枚",
        "start": 6,
        "end": 8,
        "probability": 20.36829376220703
    }],
    "密度": [{
        "text": "纯毛玻璃样",
        "start": 44,
        "end": 49,
        "probability": 4.406230449676514
    }],
    "边界": [{
        "text": "清",
        "start": 34,
        "end": 35,
        "probability": 21.579200744628906
    }],
    "边缘": [{
        "text": "模糊",
        "start": 38,
        "end": 40,
        "probability": 21.036479949951172
    }],
    "直径": [{
        "text": "7mm",
        "start": 29,
        "end": 32,
        "probability": 28.67426872253418
    }]
}]
```

### 关系预测

subject为key，object为value，object为list形式：`{'疾病与症状': ['部位与器官', '数量', '密度', '边界', '边缘', '直径', '病灶形态', '性质', '异常所见'], '部位与器官': ['纹理', '部位形态', '部位密度']}`

```python
schema = {'疾病与症状': ['部位与器官', '数量', '密度', '边界', '边缘', '直径', '病灶形态', '性质', '异常所见'], '部位与器官': ['纹理', '部位形态', '部位密度']}
ie = UIEPredictor(schema = schema, model_weight = "./checkpoints/lung_uie_mcc_vat.pt")
ie.set_schema(schema)  # Reset schema
result = ie(sample)
print(json.dumps(result, ensure_ascii = False))
```

输出
```json
[{
    "疾病与症状": [{
        "text": "实性结节",
        "start": 8,
        "end": 12,
        "probability": 18.970672607421875,
        "relations": {
            "部位与器官": [{
                "text": "右肺下叶",
                "start": 0,
                "end": 4,
                "probability": 18.747642517089844
            }, {
                "text": "右肺下叶后基底段",
                "start": 17,
                "end": 25,
                "probability": 19.03058624267578
            }],
            "数量": [{
                "text": "两枚",
                "start": 6,
                "end": 8,
                "probability": 29.957298278808594
            }],
            "密度": [{
                "text": "纯毛玻璃样",
                "start": 44,
                "end": 49,
                "probability": 5.639398574829102
            }],
            "边界": [{
                "text": "清",
                "start": 34,
                "end": 35,
                "probability": 16.658353805541992
            }],
            "边缘": [{
                "text": "模糊",
                "start": 38,
                "end": 40,
                "probability": 19.482820510864258
            }],
            "直径": [{
                "text": "7mm",
                "start": 29,
                "end": 32,
                "probability": 23.60215950012207
            }]
        }
    }],
    "部位与器官": [{
        "text": "右肺下叶",
        "start": 0,
        "end": 4,
        "probability": 27.975500106811523
    }, {
        "text": "右肺下叶后基底段",
        "start": 17,
        "end": 25,
        "probability": 26.680252075195312
    }]
}]
```

## 训练配置

```
                    'c.          bureaux@BureauxdeMacBook-Air.local
                 ,xNMM.          ----------------------------------
               .OMMMMo           OS: macOS 13.1 22C65 arm64
               OMMM0,            Host: Mac14,2
     .;loddo:' loolloddol;.      Kernel: 22.2.0
   cKMMMMMMMMMMNWMMMMMMMMMM0:    Uptime: 6 hours, 38 mins
 .KMMMMMMMMMMMMMMMMMMMMMMMWd.    Packages: 141 (brew)
 XMMMMMMMMMMMMMMMMMMMMMMMX.      Shell: zsh 5.8.1
;MMMMMMMMMMMMMMMMMMMMMMMM:       Resolution: 1470x956
:MMMMMMMMMMMMMMMMMMMMMMMM:       DE: Aqua
.MMMMMMMMMMMMMMMMMMMMMMMMX.      WM: Quartz Compositor
 kMMMMMMMMMMMMMMMMMMMMMMMMWd.    WM Theme: Blue (Dark)
 .XMMMMMMMMMMMMMMMMMMMMMMMMMMk   Terminal: iTerm2
  .XMMMMMMMMMMMMMMMMMMMMMMMMK.   Terminal Font: FiraMonoForPowerline-Medium 15
    kMMMMMMMMMMMMMMMMMMMMMMd     CPU: Apple M2
     ;KMMMMMMMWXXWMMMMMMMk.      GPU: Apple M2
       .cooc,.    .,coo:.        Memory: 2453MiB / 16384MiB
```