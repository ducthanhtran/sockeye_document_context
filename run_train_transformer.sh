#!/usr/bin/env bash
###########################################################
DIR=./data_preparation/
OUT=./testing_model/
TRAIN_S="${DIR}/news-commentary.pre.tok.post.bpe.boundary.10k.de-en.en.gz"
TRAIN_T="${DIR}/news-commentary.pre.tok.post.bpe.boundary.10k.de-en.de.gz"
DEV_S="${DIR}/newstest2015.pre.tok.post.bpe.boundary.de-en.en.gz"
DEV_T="${DIR}/newstest2015.pre.tok.post.bpe.boundary.de-en.de.gz"

ls ${TRAIN_S} ${TRAIN_T} ${DEV_S} ${DEV_T}

rm -rf ${OUT}


source /u/bahar/settings/python3-sockeye1.18.85-mxnet-cu90mkl.1.3.1--2019-03-18/bin/activate


python -m sockeye.train -s "${TRAIN_S}" \
                        -t "${TRAIN_T}" \
                        -vs "${DEV_S}" \
                        -vt "${DEV_T}" \
                        --batch-type word \
                        --batch-size 1000 \
                        --embed-dropout 0:0 \
                        --checkpoint-frequency 200 \
                        --encoder transformer \
                        --num-layers 2:2 \
                        --transformer-model-size 4 \
                        --transformer-attention-heads 2 \
                        --transformer-feed-forward-num-hidden 8 \
                        --transformer-preprocess n \
                        --transformer-postprocess dr \
                        --transformer-dropout-prepost 0.1 \
                        --transformer-dropout-act 0.1 \
                        --transformer-dropout-attention 0.1 \
                        --transformer-positional-embedding-type fixed \
                        --label-smoothing 0.1 \
                        --num-embed 4:4 \
                        --learning-rate-reduce-num-not-improved 3 \
                        --max-num-checkpoint-not-improved 4 \
                        --seed 1 \
                        --max-seq-len 99:99 \
                        --decode-and-evaluate 0 \
                        --use-cpu \
                        --method outside-decoder \
                        --src-pre 2 \
                        --src-nxt 1 \
                        --tar-pre 0 \
                        --tar-nxt 0 \
                        --source-train-doc ${TRAIN_S} \
                        --source-valid-doc ${DEV_S} \
                        --target-train-doc ${TRAIN_T} \
                        --target-valid-doc ${DEV_T} \
                        --output ${OUT}

deactivate
