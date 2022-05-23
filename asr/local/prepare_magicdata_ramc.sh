#!/usr/bin/env bash

set -e
set -o pipefail

stage=0
do_segmentation=false

. ./utils/parse_options.sh || exit 1;

if [ $# -ne 1 ]; then
  echo "Usage: $0 [options] <magicdata-ramc-dataset-dir>"
  echo " e.g.: $0 /mnt/data/yanfazu/MagicData-RAMC"
  echo ""
  echo "This script takes the MAGICDATA-RAMC source directory, and prepares the"
  echo "Kaldi format data directory."
  echo "  --stage <stage>                  # Processing stage."
  echo "  --do-segmentation <true|false>   # segment the text or not."
  exit 1
fi

magicdata_ramc_root=$1
corpus_dir=data/magicdata_ramc

if [ $stage -le 1 ]; then
  echo "$0: Extract meta into $corpus_dir"
  # Sanity check.
  [ ! -d ${magicdata_ramc_root}/DataPartition ] &&\
    echo "$0: Please download ${magicdata_ramc_root}/DataPartition!" && exit 1;
  [ ! -d ${magicdata_ramc_root}/MDT2021S003 ] &&\
    echo "$0: Please download ${magicdata_ramc_root}/MDT2021S003!" && exit 1;

  [ ! -d $corpus_dir ] && mkdir -p $corpus_dir

  # Files to be created:
  # wav.scp text segments utt2spk
  (
    export LC_ALL="zh_CN.UTF-8"
    python3 local/extract_magicdata_ramc.py -i ${magicdata_ramc_root} || exit 1;

    if [ "$do_segmentation" == "true" ]; then
      python3 -c '''import jieba''' 2>/dev/null || \
        (echo "jieba is not found. Use tools/extra/install_jieba.sh to install it." && exit 1;)
      if [ ! -f data/local/dict/word_seg_lexicon.txt ]; then
        (echo "Run local/prepare_dict.sh in advance." && exit 1;)
      fi

      for x in train dev test; do
        echo $corpus_dir/$x/
        mv $corpus_dir/$x/text $corpus_dir/$x/text.non_seg
        python3 local/word_segmentation.py data/local/dict/word_seg_lexicon.txt \
          $corpus_dir/$x/text.non_seg > $corpus_dir/$x/text
      done
    fi
    
    for x in train dev test; do
      
      for file in wav.scp utt2spk text segments; do
        sort -u $corpus_dir/$x/$file -o $corpus_dir/$x/$file
      done
      utils/utt2spk_to_spk2utt.pl $corpus_dir/$x/utt2spk > $corpus_dir/$x/spk2utt
      utils/fix_data_dir.sh  $corpus_dir/$x || exit 1;
      utils/data/validate_data_dir.sh --no-feats $corpus_dir/$x || exit 1;
    done
  )
fi

echo "$0: Done"
exit 0
