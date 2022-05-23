#!/usr/bin/env python2.7
# -*- encoding: utf-8 -*-
"""
根据所提供的的数据，生成wav.scp、segments、utt2spk、text
MagicData-RAMC
├── DataPartition
│   ├── dev.tsv
│   ├── test.tsv
│   └── train.tsv
├── KeywordList
│   ├── kwlist.txt
│   ├── ref_dev_kwlist.txt
│   └── ref_test_kwlist.txt
└── MDT2021S003
    ├── README.txt
    ├── SPKINFO.txt
    ├── TXT
    ├── UTTERANCEINFO.txt
    └── WAV
"""
import os,sys
import re
import string
import wave
from itertools import islice
import argparse


LABEL = ["[+]","[++]","[*]","[SONANT]","[MUSIC]","[LAUGHTER]"]

def get_args():
    parser = argparse.ArgumentParser(description="""data preparation long style dataset""")
    parser.add_argument('-i','--magicdata-ramc', type=str, metavar='magicdata_ramc', required=True,
                        help='magicdata ramc path')
    parser.add_argument('-r','--is-remove-tags', type=bool, default=False,
                        help="remove tags, for examples:[*]")
    args = parser.parse_args()
    return args


def remove_punctuation(text):
    """移除标点符号"""
    PUNCTIUATION = u"[！？。，＂＃＄％＆＇（）－／：；＜＝＞＠＼＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏"  + string.punctuation+']+'
    rule = re.compile(PUNCTIUATION)
    text = re.sub(rule, " ", text)
    return text


def remove_tags(text):
    """移除特殊标识符
    如：[ENS]、[LAUGH] ......
    """
    rule = re.compile('(\[[^\]|^\[]*\])')
    text = re.sub(rule, " ", text)
    return text


def find_by_pattern(path, pattern='.wav'):
    result_dict = {}
    for root, dirs, files in os.walk(path):
        for f in files:
            if f.endswith(pattern):
                result_dict[os.path.splitext(f)[0]] = os.path.abspath(os.path.join(root, f))
    return result_dict


def preprocess_data(wav_dir, txt_dir, output_dir, tsv_dir, is_remove_tags=False):
    """
    根据数据集的数据结构（文件结构），生成wav.scp、segments、utt2spk、text
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    partition_list = []
    with open(tsv_dir, 'r') as fin:
        # 跳过第一行的路径信息
        for line in islice(fin, 1, None):
            line_info = line.strip().split()
            partition_list.append(os.path.splitext(line_info[0])[0])

    wavs_dict = find_by_pattern(wav_dir, pattern='.wav')
    txts_dict = find_by_pattern(txt_dir, pattern='.txt')

    wav_fout = open(os.path.join(output_dir, 'wav.scp') ,'w')
    utt_fout = open(os.path.join(output_dir, 'utt2spk') ,'w')
    seg_fout = open(os.path.join(output_dir, 'segments') ,'w')
    text_fout = open(os.path.join(output_dir, 'text') ,'w')
    
    for txt_item in txts_dict:
        file_name = os.path.splitext(txt_item)[0]
        if file_name in wavs_dict and file_name in partition_list:
            wav_fout.write('{}.wav\t{}\n'.format(file_name, wavs_dict[file_name]))
            with open(txts_dict[txt_item], 'r') as fin:
                for line in fin:
                    line_split = line.strip().split('\t')
                    assert len(line_split) == 4, "%s"%line_split
                    beg, end = eval(line_split[0])
                    if beg < end:
                        seg_text = line_split[3]
                        spkid = line_split[1]
                        if spkid == "G00000000": continue 
                        # 移除[]标识符和标点
                        if is_remove_tags:
                            seg_text = remove_tags(seg_text)
                        seg_text = remove_punctuation(seg_text)
                        if seg_text not in LABEL:
                            uttid = '%s-%s.wav-%s-%s' % (spkid, file_name, '%07d' % round(beg*1000), '%07d' % round(end*1000))
                            utt_fout.write('{}\t{}\n'.format(uttid, spkid))
                            seg_fout.write('{}\t{}.wav\t{}\t{}\n'.format(uttid, file_name, str(beg), str(end)))
                            text_fout.write('{}\t{}\n'.format(uttid, seg_text))
                        else:
                            print("Info: {%s} in %s"%(line.strip(), txt_item))
                            pass
                    else:
                        print("Error: {%s} in %s"%(line.strip(), txt_item))

    wav_fout.close()
    utt_fout.close()
    seg_fout.close()
    text_fout.close()


def main():
    args = get_args()
    print(args.__dict__)
    DataPartition_path = os.path.join(args.magicdata_ramc, 'DataPartition')
    MDT2021S003_path = os.path.join(args.magicdata_ramc,'MDT2021S003')
    wavs_path = os.path.join(MDT2021S003_path, 'WAV') 
    txts_path = os.path.join(MDT2021S003_path, 'TXT')
    if not os.path.exists(wavs_path) or not os.path.exists(txts_path):
        raise IOError("not found WAV or TXT")
    
    preprocess_data(wavs_path, txts_path, 'data/magicdata_ramc/train', os.path.join(DataPartition_path, 'train.tsv'), is_remove_tags=args.is_remove_tags)
    preprocess_data(wavs_path, txts_path, 'data/magicdata_ramc/dev', os.path.join(DataPartition_path, 'dev.tsv'), is_remove_tags=args.is_remove_tags)
    preprocess_data(wavs_path, txts_path, 'data/magicdata_ramc/test', os.path.join(DataPartition_path, 'test.tsv'), is_remove_tags=args.is_remove_tags)
    print("Prepare MAGICDATA-RAMC done")
    
    
if __name__ == '__main__':
    main()

