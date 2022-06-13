# MagicData-AccentedMandarin-ASR-Challenge

## 规则说明
1. 数据：训练数据只能使用180小时的[MagicData-RAMC](https://magichub.com/datasets/magicdata-ramc/)或[SLR123](https://www.openslr.org/123/)和MagicData提供14小时的重口音普通话对话数据下载见邮件。
允许使用公开的噪声数据集 (如 MUSAN (openslr-17), RIRNoise (openslr-28)) 进行数据增广，但需要注明来源。禁止使用其他来源的数据(包括无监督数据)训练出的预训练模型。

2. 方法：使用ASR建模方法进行建模，允许包括模型融合，预训练-finetune，无监督自适应在内的所有方法，但需要符合1中的数据使用规范。

3. 测试：测试数据与MagicData提供14小时的重口音普通话对话数据同源，数据的发布请关注官方渠道，本次任务测试集会提供对应的时间标注信息，测试集中不存在噪音符号。

4. 打分：标点符号、非语言符不参与最终 WER(此处WER即指CER，字错误率) 计算。

## DATA Preparation
```bash
# magicdata-ramc
extract_magicdata_ramc.py

# accented mandarin dev 
extract_magicdata_accented_dev.py

# accented mandarin test
extract_magicdata_accented_test_noref.py
# extract_magicdata_accented_test_ref.py
```

##  关于Accented data的数据格式
样例
```jaon
{
    "dataset":"magicdata-accented-dev",
    "language":"ZH",
    "version":"0.1.0",
    "creation_time":"2022-05-20",
    "update_time":"2022-05-20",
    "description":"Chinese accent Mandarin",
    "copyright":"MagicData",
    "audios":[
        {
            "aid":"S0001_0_0_0_10000640003_2_1651589423000.wav",
            "path":"magicdata-accented-data/dev/audios/S0001_0_0_0_10000640003_2_1651589423000.wav",
            "total":1234.603,
            "valid":430.197,
            "topic":[
                "entertainment"
            ],
            "device":"iPhone",
            "scene":"indoor",
            "channel":"C0",
            "segments":[
                {
                    "uttid":"S0001_0_0_0_10000640003_2_1651589423000.wav-0001047-0002054",
                    "spkid":"",
                    "begin_time":1.047,
                    "end_time":2.054,
                    "text":"[ENS]"
                } ]
        }
    ],
    "speakes":[
        {
            "spkid":"G10000001",
            "gender":"Female",
            "age":16,
            "place_birth":"CHINA,Guangdong",
            "place_residence":"CHINA,Guangdong"
        }
    ]
}
```

### Test data
* 测试集文件结构
```
../magicdata-accented-data/test
├── audios
├── magicdata_test_noref.json
└── magicdata_test_uttid_submit.csv  
```

* 最终提交文件：`magicdata_test_uttid_submit.csv`,确保uttid一致
```
uttid,hyp
S0001_0_0_0_10000640002_2_1651589422000.wav-0005758-0008260,北京 爱 数 智慧 语音 采集
```

* 压缩文件大小：
```bash
$du -l magicdata-accented-data-test.tar.gz
1478496 magicdata-accented-data-test.tar.gz
```
* 测试集文件格式示例
```json
{
    "dataset":"magicdata-accented-test",
    "language":"ZH",
    "version":"0.1.0",
    "creation_time":"2022-05-20",
    "update_time":"2022-05-20",
    "description":"Chinese accent Mandarin",
    "copyright":"MagicData",
    "audios":[
        {
            "aid":"S0001_0_0_0_10000640002_2_1651589422000.wav",
            "path":"magicdata-accented-data/test/audios/S0001_0_0_0_10000640002_2_1651589422000.wav",
            "total":1235.051,
            "valid":672.612,
            "topic":[
                "entertainment"
            ],
            "device":"iPhone",
            "scene":"indoor",
            "channel":"C0",
            "segments":[
                {
                    "uttid":"S0001_0_0_0_10000640002_2_1651589422000.wav-0005758-0008260",
                    "spkid":"",
                    "begin_time":5.758,
                    "end_time":8.26,
                    "text":""
                },
                {
                    "uttid":"S0001_0_0_0_10000640002_2_1651589422000.wav-0012260-0012880",
                    "spkid":"",
                    "begin_time":12.26,
                    "end_time":12.88,
                    "text":""
                }
            ]
        },
        {
            "aid":"S0001_0_0_0_10000640963_2_1652196918000.wav",
            "path":"magicdata-accented-data/test/audios/S0001_0_0_0_10000640963_2_1652196918000.wav",
            "total":1810.901,
            "valid":729.072,
            "topic":[
                "food"
            ],
            "device":"iPhone",
            "scene":"indoor",
            "channel":"C0",
            "segments":[
                {
                    "uttid":"S0001_0_0_0_10000640963_2_1652196918000.wav-0001130-0002140",
                    "spkid":"",
                    "begin_time":1.13,
                    "end_time":2.14,
                    "text":""
                },
                {
                    "uttid":"S0001_0_0_0_10000640963_2_1652196918000.wav-0004220-0007360",
                    "spkid":"",
                    "begin_time":4.22,
                    "end_time":7.36,
                    "text":""
                }
            ]
        }
    ]
}
```


## Training
```bsah
./run.sh
```

## Submit
submit csv file
```
uttid,hyp
...

```

## Baseline result
| Model| Corr | Sub  | Del  |Ins   | WER  |
|------|------|------|------|------|------|
|CNN+TDNNF|76.596|18.35|5.049|0.875|24.28|
|CNN+TDNNF+finetune|81.876|15.053|3.07|1.12|19.24|

## Baseline model
采用传统的Hybrid的建模方式，基于Kaldi开源工具搭建了简易的重口音对话ASR 赛道的基线系统。首先用chain模型对北京爱数智慧提供的160小时中文对话数据训练了一个CNN+TDNN-F的基础模型，然后使用14小时的重口音普通话对话数据集进行了声学模型的自适应。

[CNN+TDNNF+finetune](https://freedata.oss-cn-beijing.aliyuncs.com/MagicData-AccentedMandarin-ASR-Challenge.tar.gz)

