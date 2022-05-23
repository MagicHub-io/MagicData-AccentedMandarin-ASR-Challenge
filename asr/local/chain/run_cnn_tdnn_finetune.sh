#!/bin/bash

# This is uses weight transfer as transfer learning method to transfer
# already trained AM on magicdata openslr to magicdata conversation data(160h)
# The cnn-tdnn-f (tdnn_cnn_1a_sp) outperforms the tdnn-f (tdnn_1d_sp).

set -e -o pipefail

# configs for 'chain'
stage=0
nj=20

train=train
test_sets="test"

gmm=tri3b

# The rest are configs specific to this script.  Most of the parameters
# are just hardcoded at this level, in the commands below.
train_stage=-10
affix=cnn_1a_finetune
tree_affix=_finetune
nnet3_affix=_finetune

# CNN-TDNN options
frames_per_eg=150,110,100
remove_egs=true
common_egs_dir=
xent_regularize=0.1
dropout_schedule='0,0@0.20,0.5@0.50,0'
get_egs_stage=-10
num_epochs=2
num_jobs_initial=9
num_jobs_final=10

test_online_decoding=false  # if true, it will run the last decoding stage.

#TODO: 具体路径根据基线模型训练后的为准，再做修改
src_mdl=exp/chain_ramc/cnn_tdnn_cnn_1a_sp/final.mdl # Input chain model
                                                    # trained on source dataset (magicdata openslr).
                                                    # This model is transfered to the target domain.

src_ivec_extractor_dir=exp/nnet3_ramc/extractor  # Source ivector extractor dir used to extract ivector for
                         # source data. The ivector for target data is extracted using this extractor.
                         # It should be nonempty, if ivector is used in the source model training.

src_tree_dir=exp/chain_ramc/tree_sp__ramc # chain tree-dir for src data;
                                         # the alignment in target domain is
                                         # converted using src-tree

primary_lr_factor=0.25 # The learning-rate factor for transferred layers from source
                       # model. e.g. if 0, the paramters transferred from source model
                       # are fixed.
                       # The learning-rate factor for new added layers is 1.0.


# End configuration section.
echo "$0 $@"  # Print the command line for logging

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

if ! cuda-compiled; then
  cat <<EOF && exit 1
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF
fi

nvidia-smi -c 3

# The iVector-extraction and feature-dumping parts are the same as the standard
# nnet3 setup, and you can skip them by setting "--stage 11" if you have already
# run those things.
if [ $stage -le 10 ]; then
  local/chain/run_ivector_common_finetune.sh --stage $stage \
      --nj $nj \
      --train $train \
      --gmm $gmm \
      --extractor $src_ivec_extractor_dir \
      --nnet3-affix "$nnet3_affix" || exit 1;
fi

gmm_dir=exp/$gmm
ali_dir=exp/${gmm}_ali_${train}_sp
lang=data/lang_chain
lat_dir=exp/chain${nnet3_affix}/${gmm}_${train}_sp_lats
dir=exp/chain${nnet3_affix}/cnn_tdnn${affix:+_$affix}_sp
train_data_dir=data/${train}_sp_hires
lores_train_data_dir=data/${train}_sp
train_ivector_dir=exp/nnet3${nnet3_affix}/ivectors_${train}_sp_hires


for f in $gmm_dir/final.mdl $train_data_dir/feats.scp $train_ivector_dir/ivector_online.scp \
    $lores_train_data_dir/feats.scp; do
  [ ! -f $f ] && echo "$0: expected file $f to exist" && exit 1
done


if [ $stage -le 11 ]; then
  steps/align_fmllr_lats.sh --nj $nj --cmd "$train_cmd" \
    --generate-ali-from-lats true \
    $lores_train_data_dir  $lang $gmm_dir $lat_dir
  rm $lat_dir/fsts.*.gz
fi


if [ $stage -le 12 ]; then
  # Set the learning-rate-factor for all transferred layers but the last output
  # layer to primary_lr_factor.
  $train_cmd $dir/log/generate_input_mdl.log \
    nnet3-am-copy --raw=true --edits="set-learning-rate-factor name=* learning-rate-factor=$primary_lr_factor; set-learning-rate-factor name=output* learning-rate-factor=1.0" \
      $src_mdl $dir/input.raw || exit 1;
fi


if [ $stage -le 13 ]; then
  steps/nnet3/chain/train.py --stage $train_stage \
    --use-gpu true  \
    --cmd "$train_cmd" \
    --trainer.input-model $dir/input.raw \
    --feat.online-ivector-dir $train_ivector_dir \
    --feat.cmvn-opts "--norm-means=false --norm-vars=false" \
    --chain.xent-regularize $xent_regularize \
    --chain.leaky-hmm-coefficient 0.1 \
    --chain.l2-regularize 0.0 \
    --chain.apply-deriv-weights false \
    --chain.lm-opts="--num-extra-lm-states=2000" \
    --egs.dir "$common_egs_dir" \
    --egs.stage $get_egs_stage \
    --egs.opts "--frames-overlap-per-eg 0 --constrained false --max-shuffle-jobs-run 40" \
    --egs.chunk-width $frames_per_eg \
    --trainer.dropout-schedule $dropout_schedule \
    --trainer.add-option="--optimization.memory-compression-level=2" \
    --trainer.num-chunk-per-minibatch 128,64 \
    --trainer.frames-per-iter 1500000 \
    --trainer.num-epochs $num_epochs \
    --trainer.optimization.num-jobs-initial ${num_jobs_initial} \
    --trainer.optimization.num-jobs-final ${num_jobs_final} \
    --trainer.optimization.initial-effective-lrate 0.00025 \
    --trainer.optimization.final-effective-lrate 0.000025 \
    --trainer.max-param-change 2.0 \
    --cleanup.remove-egs $remove_egs \
    --feat-dir $train_data_dir \
    --tree-dir $src_tree_dir \
    --lat-dir $lat_dir \
    --dir $dir  || exit 1;
fi


nnet3_affix=_ramc  # run_cnn_tdnn.sh 脚本中已经提取过特征，直接拿来使用即可
graph_dir=$dir/graph
if [ $stage -le 14 ]; then
  # Note: it's not important to give mkgraph.sh the lang directory with the
  # matched topology (since it gets the topology file from the model).
  utils/mkgraph.sh --self-loop-scale 1.0  data/lang_test $dir $graph_dir
fi

if [ $stage -le 15 ]; then
  # Create high-resolution MFCC features (with 40 cepstra instead of 13).
  # this shows how you can split across multiple file-systems.  we'll split the
  # MFCC dir across multiple locations.  You might want to be careful here, if you
  # have multiple copies of Kaldi checked out and run the same recipe, not to let
  # them overwrite each other.
  echo "$0: creating high-resolution MFCC features"

  for part in $test_sets; do
    utils/copy_data_dir.sh data/$part data/${part}_hires

    nspk=$(wc -l <data/${part}_hires/spk2utt)
    if [ $nspk -gt $nj ]; then
      nspk=$nj
    fi
    steps/make_mfcc.sh --nj $nspk --mfcc-config conf/mfcc_hires.conf \
      --cmd "$train_cmd" data/${part}_hires || exit 1;
    steps/compute_cmvn_stats.sh data/${part}_hires || exit 1;
    utils/fix_data_dir.sh data/${part}_hires
  done
fi

if [ $stage -le 16 ]; then
  echo "$0: extracting iVectors for test data"
  for part in $test_sets; do
    nspk=$(wc -l <data/${part}_hires/spk2utt)
    if [ $nspk -gt $nj ]; then
      nspk=$nj
    fi
    steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj $nspk \
      data/${part}_hires exp/nnet3${nnet3_affix}/extractor \
      exp/nnet3${nnet3_affix}/ivectors_${part}_hires || exit 1;
  done
fi

if [ $stage -le 17 ]; then
  rm $dir/.error 2>/dev/null || true
  for part in $test_sets; do
    (
      nspk=$(wc -l <data/$part/spk2utt)
      if [ $nspk -gt $nj ]; then
        nspk=$nj
      fi	
      steps/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 \
        --nj $nspk --cmd "$decode_cmd" \
        --online-ivector-dir exp/nnet3${nnet3_affix}/ivectors_${part}_hires \
        $graph_dir data/${part}_hires $dir/decode_${part} || exit 1
    ) || touch $dir/.error &
  done
  wait
  [ -f $dir/.error ] && echo "$0: there was a problem while decoding" && exit 1
fi

if $test_online_decoding && [ $stage -le 18 ]; then
  # note: if the features change (e.g. you add pitch features), you will have to
  # change the options of the following command line.
  steps/online/nnet3/prepare_online_decoding.sh \
    --mfcc-config conf/mfcc_hires.conf \
    $lang exp/nnet3${nnet3_affix}/extractor $dir ${dir}_online

  rm $dir/.error 2>/dev/null || true
  for part in $test_sets; do
    (
      nspk=$(wc -l <data/$part/spk2utt)
      if [ $nspk -gt $nj ]; then
        nspk=$nj
      fi	
      steps/online/nnet3/decode.sh \
        --acwt 1.0 --post-decode-acwt 10.0 \
        --nj $nspk --cmd "$decode_cmd" \
        $graph_dir data/$part ${dir}_online/decode_${part} || exit 1
    ) || touch $dir/.error &
  done
  wait
  [ -f $dir/.error ] && echo "$0: there was a problem while decoding" && exit 1
fi

exit 0;
