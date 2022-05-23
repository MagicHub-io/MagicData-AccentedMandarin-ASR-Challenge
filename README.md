# MagicData-AccentedMandarin-ASR-Challenge

## 规则说明
1. 数据：训练数据只能使用180小时的[MagicData-RAMC](https://magichub.com/datasets/magicdata-ramc/)和MagicData提供14小时的重口音普通话对话数据[重口音数据]()。
允许使用公开的噪声数据集 (如 MUSAN (openslr-17), RIRNoise (openslr-28)) 进行数据增广，但需要注明来源。禁止使用其他来源的数据(包括无监督数据)训练出的预训练模型。

2. 方法：使用ASR建模方法进行建模，允许包括模型融合，预训练-finetune，无监督自适应在内的所有方法，但需要符合1中的数据使用规范。

3. 测试：测试数据与MagicData提供14小时的重口音普通话对话数据同源，数据的发布请关注官方渠道，本次任务测试集会提供对应的时间标注信息，测试集中不存在噪音符号。

4. 打分：标点符号、非语言符不参与最终 WER 计算。

## DATA Preparation
```
# magicdata-ramc
extract_magicdata_ramc.py

# accented mandarin dev 
extract_magicdata_accented_dev.py

# accented mandarin test
extract_magicdata_accented_test_noref.py
# extract_magicdata_accented_test_ref.py
```
