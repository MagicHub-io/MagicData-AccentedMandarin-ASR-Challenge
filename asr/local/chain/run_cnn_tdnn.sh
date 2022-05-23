#!/bin/bash

# This script is copied from librispeech/s5
# In a previous version, pitch is used with hires mfcc, however,
# removing pitch does not cause regression, and helps online
# decoding, so pitch is removed in this recipe.

# This is based on tdnn_1d_sp, but adding cnn as the front-end.
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
affix=cnn_1a
tree_affix=_ramc
nnet3_affix=_ramc

# TDNN options
frames_per_eg=150,110,100
remove_egs=true
common_egs_dir=
xent_regularize=0.1
dropout_schedule='0,0@0.20,0.5@0.50,0'
get_egs_stage=-10

num_epochs=6
num_jobs_initial=9 
num_jobs_final=10

test_online_decoding=false  # if true, it will run the last decoding stage.

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
  local/chain/run_ivector_common.sh --stage $stage \
                                    --nj $nj \
                                    --train $train \
                                    --gmm $gmm \
                                    --nnet3-affix "$nnet3_affix" || exit 1;
fi

gmm_dir=exp/$gmm
ali_dir=exp/${gmm}_ali_${train}_sp
tree_dir=exp/chain${nnet3_affix}/tree_sp${tree_affix:+_$tree_affix}
lang=data/lang_chain
lat_dir=exp/chain${nnet3_affix}/${gmm}_${train}_sp_lats
dir=exp/chain${nnet3_affix}/cnn_tdnn${affix:+_$affix}_sp
train_data_dir=data/${train}_sp_hires
lores_train_data_dir=data/${train}_sp
train_ivector_dir=exp/nnet3${nnet3_affix}/ivectors_${train}_sp_hires


for f in $gmm_dir/final.mdl $train_data_dir/feats.scp $train_ivector_dir/ivector_online.scp \
    $lores_train_data_dir/feats.scp $ali_dir/ali.1.gz; do
  [ ! -f $f ] && echo "$0: expected file $f to exist" && exit 1
done


if [ $stage -le 11 ]; then
  # Please take this as a reference on how to specify all the options of
  # local/chain/run_chain_common.sh
  local/chain/run_chain_common.sh --stage $stage \
                                  --gmm-dir $gmm_dir \
                                  --ali-dir $ali_dir \
                                  --lores-train-data-dir ${lores_train_data_dir} \
                                  --lang $lang \
                                  --lat-dir $lat_dir \
                                  --num-leaves 5000 \
                                  --tree-dir $tree_dir || exit 1;
fi


if [ $stage -le 12 ]; then
  echo "$0: creating neural net configs using the xconfig parser";

  num_targets=$(tree-info $tree_dir/tree | grep num-pdfs | awk '{print $2}')
  learning_rate_factor=$(echo "print (0.5/$xent_regularize)" | python)
  cnn_opts="l2-regularize=0.01"
  ivector_affine_opts="l2-regularize=0.0"
  affine_opts="l2-regularize=0.008 dropout-proportion=0.0 dropout-per-dim=true dropout-per-dim-continuous=true"
  tdnnf_first_opts="l2-regularize=0.008 dropout-proportion=0.0 bypass-scale=0.0"
  tdnnf_opts="l2-regularize=0.008 dropout-proportion=0.0 bypass-scale=0.75"
  linear_opts="l2-regularize=0.008 orthonormal-constraint=-1.0"
  prefinal_opts="l2-regularize=0.008"
  output_opts="l2-regularize=0.005"

  mkdir -p $dir/configs

  cat <<EOF > $dir/configs/network.xconfig
  input dim=100 name=ivector
  input dim=40 name=input

  # MFCC to filterbank
  idct-layer name=idct input=input dim=40 cepstral-lifter=22 affine-transform-file=$dir/configs/idct.mat

  linear-component name=ivector-linear $ivector_affine_opts dim=200 input=ReplaceIndex(ivector, t, 0)
  batchnorm-component name=ivector-batchnorm target-rms=0.025
  batchnorm-component name=idct-batchnorm input=idct

  combine-feature-maps-layer name=combine_inputs input=Append(idct-batchnorm, ivector-batchnorm) num-filters1=1 num-filters2=5 height=40
  conv-relu-batchnorm-layer name=cnn1 $cnn_opts height-in=40 height-out=40 time-offsets=-1,0,1 height-offsets=-1,0,1 num-filters-out=64
  conv-relu-batchnorm-layer name=cnn2 $cnn_opts height-in=40 height-out=40 time-offsets=-1,0,1 height-offsets=-1,0,1 num-filters-out=64
  conv-relu-batchnorm-layer name=cnn3 $cnn_opts height-in=40 height-out=20 height-subsample-out=2 time-offsets=-1,0,1 height-offsets=-1,0,1 num-filters-out=128
  conv-relu-batchnorm-layer name=cnn4 $cnn_opts height-in=20 height-out=20 time-offsets=-1,0,1 height-offsets=-1,0,1 num-filters-out=128
  conv-relu-batchnorm-layer name=cnn5 $cnn_opts height-in=20 height-out=10 height-subsample-out=2 time-offsets=-1,0,1 height-offsets=-1,0,1 num-filters-out=256
  conv-relu-batchnorm-layer name=cnn6 $cnn_opts height-in=10 height-out=10 time-offsets=-1,0,1 height-offsets=-1,0,1 num-filters-out=256

  # the first TDNN-F layer has no bypass
  tdnnf-layer name=tdnnf7 $tdnnf_first_opts dim=1536 bottleneck-dim=256 time-stride=0
  tdnnf-layer name=tdnnf8 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
  tdnnf-layer name=tdnnf9 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
  tdnnf-layer name=tdnnf10 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
  tdnnf-layer name=tdnnf11 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
  tdnnf-layer name=tdnnf12 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
  tdnnf-layer name=tdnnf13 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
  tdnnf-layer name=tdnnf14 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
  tdnnf-layer name=tdnnf15 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
  tdnnf-layer name=tdnnf16 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
  tdnnf-layer name=tdnnf17 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
  tdnnf-layer name=tdnnf18 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
  linear-component name=prefinal-l dim=256 $linear_opts

  ## adding the layers for chain branch
  prefinal-layer name=prefinal-chain input=prefinal-l $prefinal_opts big-dim=1536 small-dim=256
  output-layer name=output include-log-softmax=false dim=$num_targets $output_opts

  # adding the layers for xent branch
  prefinal-layer name=prefinal-xent input=prefinal-l $prefinal_opts big-dim=1536 small-dim=256
  output-layer name=output-xent dim=$num_targets learning-rate-factor=$learning_rate_factor $output_opts
EOF
  steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig --config-dir $dir/configs/
fi

if [ $stage -le 13 ]; then
  steps/nnet3/chain/train.py --stage $train_stage \
    --use-gpu true \
    --cmd "$train_cmd" \
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
    --trainer.num-epochs 6 \
    --trainer.optimization.num-jobs-initial 9 \
    --trainer.optimization.num-jobs-final 10 \
    --trainer.optimization.initial-effective-lrate 0.00015 \
    --trainer.optimization.final-effective-lrate 0.000015 \
    --trainer.max-param-change 2.0 \
    --cleanup.remove-egs $remove_egs \
    --feat-dir $train_data_dir \
    --tree-dir $tree_dir \
    --lat-dir $lat_dir \
    --dir $dir  || exit 1;
fi

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
