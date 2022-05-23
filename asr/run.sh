#!/usr/bin/env bash
# you might not want to do this for interactive shells.
set -e
set -o pipefail

# ---- Configurations ----
stage=7
stage_gmm=0
stage_chain=0

nj=30
ngram_order=3

lm_dir=data/local/lm
dict_dir=data/local/dict

magicdata_ramc_root=/mnt/data/yanfazu/MagicData-RAMC
train_ramc=train_ramc
test_ramc=test_ramc
# ---- Configurations End ----

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh


# Download DaCiDian raw resources, convert to Kaldi lexicon format
if [ $stage -le 0 ]; then
  local/prepare_dict.sh ${dict_dir} || exit 1;
fi

# wav.scp, text(word-segmented), segments，utt2spk, spk2utt
if [ $stage -le 1 ]; then
  local/prepare_magicdata_ramc.sh --do-segmentation true  ${magicdata_ramc_root} || exit 1;

  ./utils/combine_data.sh data/${train_ramc} data/magicdata_ramc/train  data/magicdata_ramc/dev || exit 1;
  ./utils/data/copy_data_dir.sh data/magicdata_ramc/test data/${test_ramc} || exit 1
fi

# arpa LM
if [ $stage -le 2 ]; then
  mkdir -p  data/corpus
  sed 's|\t| |' data/${train_ramc}/text |  cut -d " " -f 2- > data/corpus/corpus.txt || exit 1;  
  
  local/train_lm.sh  --ngram-order $ngram_order \
    $dict_dir/lexicon.txt data/corpus/corpus.txt $lm_dir || exit 1;
fi

# L
if [ $stage -le 3 ]; then
  utils/prepare_lang.sh --position-dependent-phones false \
    ${dict_dir} "<UNK>" data/local/lang_nosp data/lang_nosp || exit 1;
fi

# G compilation, check LG composition
if [ $stage -le 4 ]; then
  utils/format_lm.sh data/lang ${lm_dir}/lm_${ngram_order}gram.arpa.gz \
    $dict_dir/lexicon.txt data/lang_test || exit 1;
fi

# GMM 
if [ $stage -le 5 ]; then
  ./local/run_gmm_common.sh --stage ${stage_gmm} --nj $nj \
      --train ${train_ramc} --test-sets ${test_ramc} || exit 1;
fi


# CNN-TDNN-f, chain 
if [ $stage -le 6 ]; then
  ./local/chain/run_cnn_tdnn.sh --stage ${stage_chain} --nj $nj \
      --num-epochs 6  --num-jobs-initial 9 --num-jobs-final 10 \
      --train ${train_ramc} --test-sets ${test_ramc} || exit 1;
fi

################
# accented data
###############
accented_data_root=/mnt/data/yanfazu/MagicData-AccentedData/
accented_dev_json=${accented_data_root}/magicdata-accented-data/dev/magicdata_dev.json
accented_test_json=${accented_data_root}/magicdata-accented-data/test/magicdata_test.json
#accented_test_noref_json=${accented_data_root}/magicdata-accented-data/test/magicdata_test_noref.json
accented_test_uttid_submit=${accented_data_root}/magicdata-accented-data/test/magicdata_test_uttid_submit.csv

# Prepare accented data
if [ $stage -le 7 ]; then
  python3  local/extract_magicdata_accented_dev.py -i ${accented_dev_json} \
           --dataroot ${accented_data_root} --output data/magicdata_accented/dev

  python3  local/extract_magicdata_accented_test_ref.py -i ${accented_test_json}  --submit ${accented_test_uttid_submit}\
          --dataroot ${accented_data_root} --output data/magicdata_accented/test

  #python3  local/extract_magicdata_accented_test_ref.py -i ${accented_test_noref_json}  --submit ${accented_test_uttid_submit}\
  #        --dataroot ${accented_data_root} --output data/magicdata_accented/test
  
  do_segmentation=true
  if [ "$do_segmentation" == "true" ]; then
      python3 -c '''import jieba''' 2>/dev/null || \
        (echo "jieba is not found. Use tools/extra/install_jieba.sh to install it." && exit 1;)
      if [ ! -f data/local/dict/word_seg_lexicon.txt ]; then
        (echo "Run local/prepare_dict.sh in advance." && exit 1;)
      fi

      for x in dev test; do
        echo data/magicdata_accented/$x/
        mv data/magicdata_accented/$x/text data/magicdata_accented/$x/text.non_seg
        python3 local/word_segmentation.py data/local/dict/word_seg_lexicon.txt \
          data/magicdata_accented/$x/text.non_seg > data/magicdata_accented/$x/text
      done
  fi

    for x in  dev test; do
      for file in wav.scp utt2spk text segments; do
        sort -u data/magicdata_accented/$x/$file -o data/magicdata_accented/$x/$file
      done
      utils/utt2spk_to_spk2utt.pl data/magicdata_accented/$x/utt2spk > data/magicdata_accented/$x/spk2utt
      utils/fix_data_dir.sh  data/magicdata_accented/$x || exit 1;
      utils/data/validate_data_dir.sh --no-feats data/magicdata_accented/$x || exit 1;
    done
fi

if [ $stage -le 8 ]; then
  # Create high-resolution MFCC features (with 40 cepstra instead of 13).
  # this shows how you can split across multiple file-systems.  we'll split the
  # MFCC dir across multiple locations.  You might want to be careful here, if you
  # have multiple copies of Kaldi checked out and run the same recipe, not to let
  # them overwrite each other.
  echo "$0: creating high-resolution MFCC features"

  for part in dev test; do
    utils/copy_data_dir.sh data/magicdata_accented/$part data/${part}_accented_decode

    nspk=$(wc -l <data/${part}_accented_decode/spk2utt)
    if [ $nspk -gt $nj ]; then
      nspk=$nj
    fi
    steps/make_mfcc.sh --nj $nspk --mfcc-config conf/mfcc_hires.conf \
      --cmd "$train_cmd" data/${part}_accented_decode || exit 1;
    steps/compute_cmvn_stats.sh data/${part}_accented_decode || exit 1;
    utils/fix_data_dir.sh data/${part}_accented_decode
  done
fi

if [ $stage -le 9 ]; then
  ivector_dir=exp/nnet3_ramc/extractor
  echo "$0: extracting iVectors for dev test data"
  for part in dev test; do
    nspk=$(wc -l <data/${part}_accented_decode/spk2utt)
    if [ $nspk -gt $nj ]; then
      nspk=$nj
    fi
    steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj $nspk \
      data/${part}_accented_decode  ${ivector_dir}\
      exp/accented/ivectors_${part}_accented_decode || exit 1;
  done
fi

am_dir=exp/chain_ramc/cnn_tdnn_cnn_1a_sp
src_graph_dir=exp/chain_ramc/cnn_tdnn_cnn_1a_sp/graph
if [ $stage -le 10 ]; then
  for part in dev test; do
      nspk=$(wc -l <data/${part}_accented_decode/spk2utt)
      if [ $nspk -gt $nj ]; then
        nspk=$nj
      fi
      steps/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 \
        --nj $nspk --cmd "$decode_cmd" \
        --online-ivector-dir exp/accented/ivectors_${part}_accented_decode \
        ${src_graph_dir} data/${part}_accented_decode ${am_dir}/decode_${part}_accented_decode || exit 1
  done
fi


# train Finetune 
if [ $stage -le 11 ]; then
  for part in dev test; do
      utils/copy_data_dir.sh data/magicdata_accented/$part data/${part}_accented
  done
fi

# Finetune 
if [ $stage -le 12 ]; then
  ./local/chain/run_cnn_tdnn_finetune.sh --nj 30  --train dev_accented --test-sets test_accented \
    --stage ${stage_chain} --num-epochs 2 --num-jobs-initial 9 --num-jobs-final 10
fi


# Decode accented test
if [ $stage -le 13 ]; then
  cat exp/chain_finetune/cnn_tdnn_cnn_1a_finetune_sp/decode_test_accented/log/decode* | grep -a "\.wav" | grep -v -a LOG | grep -v -a WARNING > exp/decode/test_accented_decode_submit.txt
fi

exit 0;
