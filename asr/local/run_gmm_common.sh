#!/usr/bin/env bash
# This script runs all common stages for training the GMM models

set -euo pipefail

stage=0
nj=20

exp=exp
train=train
test_sets="test"

tri3_lda=false  # If true, the final system (tri3) will be LDA
                  # and no SAT will be trained.

. ./cmd.sh
. ./path.sh
. utils/parse_options.sh


echo "$0 $@"  # Print the command line for logging


if [ $stage -le 0 ]; then
  echo "Extract mfcc feat"
  mfccdir=mfcc
  for part in $train $test_sets; do
    (
    if [ -f data/$part/feats.scp ]; then
        printf "\nNote: data/$part/feats.scp exists. skipping...\n\n"
    else
        steps/make_mfcc_pitch_online.sh --cmd "$train_cmd" --nj $nj \
          --online-pitch-config conf/online_pitch.conf \
          data/$part $exp/make_mfcc/$part $mfccdir || exit 1;
        steps/compute_cmvn_stats.sh \
          data/$part $exp/make_mfcc/$part $mfccdir || exit 1;
        utils/fix_data_dir.sh data/$part
    fi
    ) &
  done
  wait
fi

if [ $stage -le 1 ]; then
  num_utts=$(wc -l < data/$train/utt2spk)
   # 50k utterances for monophone
  subset_num=50000
  if [ $subset_num -ge $num_utts ]; then
    ln -sf data/$train data/${train}_mono
  else
    utils/subset_data_dir.sh --shortest \
      data/$train $subset_num data/${train}_mono
  fi

  # 150k utterances for deltas
  subset_num=150000
  if [ $subset_num -ge $num_utts ]; then
    ln -sf data/$train data/${train}_deltas
  else
    utils/subset_data_dir.sh \
      data/$train $subset_num data/${train}_deltas
  fi

  # 500k utterances for deltas
  subset_num=500000
  if [ $subset_num -ge $num_utts ]; then
    ln -sf $train data/${train}_lda
  else
    utils/subset_data_dir.sh \
      data/$train $subset_num data/${train}_lda
  fi

  # 1000k utterances for deltas
  subset_num=1000000
  if [ $subset_num -ge $num_utts ]; then
    ln -sf $train data/${train}_tri3
  else
    utils/subset_data_dir.sh \
      data/$train $subset_num data/${train}_tri3
  fi
fi

if [ $stage -le 2 ]; then
  steps/train_mono.sh --boost-silence 1.25 --nj $nj --cmd "$train_cmd" \
    data/${train}_mono data/lang $exp/mono

  {
    utils/mkgraph.sh data/lang_test $exp/mono $exp/mono/graph || exit 1

    for x in $test_sets; do
      nspk=$(wc -l <data/$x/spk2utt)
      if [ $nspk -gt $nj ]; then
        nspk=$nj
      fi
      steps/decode.sh --nj $nspk --cmd "$decode_cmd" \
        $exp/mono/graph data/$x \
        $exp/mono/decode_$x || exit 1
    done
  }

  steps/align_si.sh --boost-silence 1.25 --nj $nj --cmd "$train_cmd" \
    data/${train}_deltas data/lang $exp/mono $exp/mono_ali_${train}_deltas
fi

if [ $stage -le 3 ]; then
  steps/train_deltas.sh --boost-silence 1.25 --cmd "$train_cmd" \
    2500 20000 data/${train}_deltas data/lang $exp/mono_ali_${train}_deltas $exp/tri1a

  {
    utils/mkgraph.sh data/lang_test $exp/tri1a $exp/tri1a/graph || exit 1

    for x in $test_sets; do
      nspk=$(wc -l <data/$x/spk2utt)
      if [ $nspk -gt $nj ]; then
        nspk=$nj
      fi
      steps/decode.sh --nj $nspk --cmd "$decode_cmd" \
        $exp/tri1a/graph data/$x \
        $exp/tri1a/decode_$x || exit 1
    done
  }

  steps/align_si.sh --nj $nj --cmd "$train_cmd" \
    data/${train}_deltas data/lang $exp/tri1a $exp/tri1a_ali_${train}_deltas
fi

if [ $stage -le 4 ]; then
  steps/train_deltas.sh --boost-silence 1.25 --cmd "$train_cmd" \
    4500 36000 data/${train}_deltas data/lang $exp/tri1a_ali_${train}_deltas $exp/tri1b

  {
    utils/mkgraph.sh data/lang_test $exp/tri1b $exp/tri1b/graph || exit 1

    for x in $test_sets; do
      nspk=$(wc -l <data/$x/spk2utt)
      if [ $nspk -gt $nj ]; then
        nspk=$nj
      fi
      steps/decode.sh --nj $nspk --cmd "$decode_cmd" \
        $exp/tri1b/graph data/$x \
        $exp/tri1b/decode_$x || exit 1
    done
  }

  steps/align_si.sh --nj $nj --cmd "$train_cmd" \
    data/${train}_lda data/lang $exp/tri1b $exp/tri1b_ali_${train}_lda
fi

if [ $stage -le 5 ]; then
  steps/train_lda_mllt.sh --cmd "$train_cmd" \
    --splice-opts "--left-context=3 --right-context=3" 5000 90000 \
    data/${train}_lda data/lang $exp/tri1b_ali_${train}_lda $exp/tri2a

  {
    utils/mkgraph.sh data/lang_test $exp/tri2a $exp/tri2a/graph || exit 1

    for x in $test_sets; do
      nspk=$(wc -l <data/$x/spk2utt)
      if [ $nspk -gt $nj ]; then
        nspk=$nj
      fi
      steps/decode.sh --nj $nspk --cmd "$decode_cmd" \
        $exp/tri2a/graph data/$x \
        $exp/tri2a/decode_$x || exit 1
    done
  }
  steps/align_fmllr.sh --nj $nj --cmd "$train_cmd" \
    data/${train}_tri3 data/lang $exp/tri2a $exp/tri2a_ali_${train}_tri3
fi

# Train tri3b, which is LDA+MLLT+SAT
if [ $stage -le 6 ]; then
  if $tri3_lda; then
    steps/train_lda_mllt.sh --cmd "$train_cmd" \
      --splice-opts "--left-context=3 --right-context=3" 7000 110000 \
      data/${train}_tri3 data/lang $exp/tri2a_ali_${train}_tri3 $exp/tri3b
  else
    steps/train_sat.sh --cmd "$train_cmd" 7000 110000 \
      data/${train}_tri3 data/lang $exp/tri2a_ali_${train}_tri3 $exp/tri3b
  fi

  {
    utils/mkgraph.sh data/lang_test $exp/tri3b $exp/tri3b/graph || exit 1

    for x in $test_sets; do
      nspk=$(wc -l <data/$x/spk2utt)
      if [ $nspk -gt $nj ]; then
        nspk=$nj
      fi
      steps/decode.sh --nj "$nspk" --cmd "$decode_cmd" \
        $exp/tri3b/graph data/$x \
        $exp/tri3b/decode_$x || exit 1
    done
  }
fi

exit 0;

