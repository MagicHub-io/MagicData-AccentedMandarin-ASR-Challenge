#!/bin/bash

set -euo pipefail

# This script is copied from librispeech

# This script is called from local/nnet3/run_tdnn.sh and local/chain/run_tdnn.sh (and may eventually
# be called by more scripts).  It contains the common feature preparation and iVector-related parts
# of the script.  See those scripts for examples of usage.

stage=0
nj=20
train=train      # you might set this to e.g. train_all
gmm=tri3b       # This specifies a GMM-dir from the features of the type you're training the system on;
num_threads_ubm=4
num_processes=2
nnet3_affix=_cleaned     # affix for exp/nnet3 directory to put iVector stuff in, so it
                         # becomes exp/nnet3_cleaned or whatever.
extractor=exp/nnet3${nnet3_affix}/extractor

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

gmm_dir=exp/${gmm}
ali_dir=exp/${gmm}_ali_${train}_sp

#for f in data/${train}/feats.scp ${gmm_dir}/final.mdl; do
for f in ${gmm_dir}/final.mdl; do
  if [ ! -f $f ]; then
    echo "$0: expected file $f to exist"
    exit 1
  fi
done

if [ $stage -le 1 ]; then
  #Although the nnet will be trained by high resolution data, we still have to
  # perturb the normal data to get the alignment.  _sp stands for speed-perturbed
  echo "$0: preparing directory for low-resolution speed-perturbed data (for alignment)"
  utils/data/perturb_data_dir_speed_3way.sh data/${train} data/${train}_sp
  echo "$0: making MFCC features for low-resolution speed-perturbed data"
  # steps/make_mfcc.sh --cmd "$train_cmd" --nj $nj data/${train}_sp || exit 1;
  steps/make_mfcc_pitch_online.sh --cmd "$train_cmd" --nj $nj \
    --online-pitch-config conf/online_pitch.conf data/${train}_sp || exit 1
  steps/compute_cmvn_stats.sh data/${train}_sp || exit 1;
  echo "$0: fixing input data-dir to remove nonexistent features, in case some "
  echo ".. speed-perturbed segments were too short."
  utils/fix_data_dir.sh data/${train}_sp
fi

if [ $stage -le 3 ]; then
  # Create high-resolution MFCC features (with 40 cepstra instead of 13).
  # this shows how you can split across multiple file-systems.  we'll split the
  # MFCC dir across multiple locations.  You might want to be careful here, if you
  # have multiple copies of Kaldi checked out and run the same recipe, not to let
  # them overwrite each other.
  echo "$0: creating high-resolution MFCC features"
  mfccdir=data/${train}_sp_hires/data

  utils/copy_data_dir.sh data/${train}_sp data/${train}_sp_hires

  # do volume-perturbation on the training data prior to extracting hires
  # features; this helps make trained nnets more invariant to test data volume.
  utils/data/perturb_data_dir_volume.sh data/${train}_sp_hires
  steps/make_mfcc.sh --nj $nj --mfcc-config conf/mfcc_hires.conf \
    --cmd "$train_cmd" data/${train}_sp_hires || exit 1;
  steps/compute_cmvn_stats.sh data/${train}_sp_hires || exit 1;
  utils/fix_data_dir.sh data/${train}_sp_hires
fi

if [ $stage -le 4 ]; then
  echo "$0: extracting iVectors for training data"
  ivectordir=exp/nnet3${nnet3_affix}/ivectors_${train}_sp_hires
  # We extract iVectors on the speed-perturbed training data after combining
  # short segments, which will be what we train the system on.  With
  # --utts-per-spk-max 2, the script pairs the utterances into twos, and treats
  # each of these pairs as one speaker. this gives more diversity in iVectors..
  # Note that these are extracted 'online'.

  # having a larger number of speakers is helpful for generalization, and to
  # handle per-utterance decoding well (iVector starts at zero).
  utils/data/modify_speaker_info.sh --utts-per-spk-max 2 \
    data/${train}_sp_hires data/${train}_sp_hires_max2

  steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj $nj \
    data/${train}_sp_hires_max2  $extractor $ivectordir || exit 1;
fi

exit 0;
