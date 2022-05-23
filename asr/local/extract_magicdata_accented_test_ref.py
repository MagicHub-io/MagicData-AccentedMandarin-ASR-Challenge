# -*- encoding: utf-8 -*-
"""
根据所提供的的数据，生成wav.scp、segments、utt2spk、text
"""
import os
import re
import string
import csv
import json
import wave
from itertools import islice
import argparse

from tqdm import tqdm


LABEL = ["[+]","[++]","[*]","[SONANT]","[MUSIC]","[LAUGHTER]", "[ENS]", "[SYSTEM]"]

def get_args():
    parser = argparse.ArgumentParser(description="""data preparation long style dataset""")
    parser.add_argument('-i','--infile', type=str, metavar='infile', required=True,
                        help='magicdata accented json')
    parser.add_argument('--submit', type=str, metavar='insubmit', required=True,
                        help='magicdata accented test submit csv')
    parser.add_argument('--dataroot', type=str, metavar='dataroot', required=True,
                        help='magicdata accented root')
    parser.add_argument('--output', type=str, metavar='doutput', required=True,
                        help='magicdata accented output')
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

def extract_uttidinfo(csv_file):
    uttid_dict = {}
    with open(csv_file, 'r') as f:
        f_csv = csv.reader(f)
        headers = next(f_csv)
        print(headers)
        for idx, row in enumerate(f_csv):
            uttid = row[0]
            if uttid not in uttid_dict:
                uttid_dict[uttid] = row
    return uttid_dict

def preprocess_data(json_file, submit_csv, data_root, output_dir, is_remove_tags=False):
    """
    根据数据集的数据结构（文件结构），生成wav.scp、segments、utt2spk、text
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    wav_fout = open(os.path.join(output_dir, 'wav.scp') ,'w')
    utt_fout = open(os.path.join(output_dir, 'utt2spk') ,'w')
    seg_fout = open(os.path.join(output_dir, 'segments') ,'w')
    text_fout = open(os.path.join(output_dir, 'text') ,'w')

    uttid_dict = extract_uttidinfo(submit_csv)
    
    with open(json_file, 'r', encoding="utf-8") as fin:
        test_data = json.load(fin)

    for item in tqdm(test_data['audios']):
        wav_id = item['aid']
        path = os.path.join(data_root, item['path'])
        wav_fout.write('{}\t{}\n'.format(wav_id, path))

        segments_info =item['segments']
        for seg_info in segments_info:
            beg, end = seg_info["begin_time"], seg_info["end_time"]
            spkid = seg_info["spkid"]
            text = remove_punctuation(seg_info['text'])
            uttid= seg_info['uttid']

            if spkid and uttid in uttid_dict:
                uttid= spkid+'-'+seg_info['uttid']
                utt_fout.write('{}\t{}\n'.format(uttid, spkid))
                seg_fout.write('{}\t{}\t{}\t{}\n'.format(uttid, wav_id, str(beg), str(end)))
                text_fout.write('{}\t{}\n'.format(uttid, text))

    wav_fout.close()
    utt_fout.close()
    seg_fout.close()
    text_fout.close()


def main():
    args = get_args()
    print(args.__dict__)
    preprocess_data(args.infile, args.submit, args.dataroot, args.output)
    print("Prepare done")
    
    
if __name__ == '__main__':
    main()
