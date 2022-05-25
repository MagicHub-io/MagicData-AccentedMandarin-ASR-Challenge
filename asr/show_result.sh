#!/bin/bash


for mdl in mono tri1a tri1b tri2a tri3b;do
    grep WER exp/$mdl/decode_*/cer_* | utils/best_wer.sh; 
done

echo
echo "# fintune model for magicdata-ramc test、 accented dev、accented test"
for x in exp/chain_ramc/cnn_tdnn_*/decode_*; do [ -d $x ] && grep WER $x/cer_* | utils/best_wer.sh; done 2>/dev/null

echo
echo "# fintune model for accented test"
for x in exp/chain_finetune/cnn_tdnn_cnn_1a_finetune_sp/decode_*; do [ -d $x ] && grep WER $x/cer_* | utils/best_wer.sh; done 2>/dev/null

