#!/usr/bin/env bash
###########################################################
DIR=./data_preparation/
MODEL=./testing_model/
OUT=./testing_model/hyp.test18
TEST_S="${DIR}/newstest2018.pre.tok.post.bpe.boundary.200.de-en.en.gz"


ls ${TEST_S}


source /u/bahar/settings/python3-sockeye1.18.85-mxnet-cu90mkl.1.3.1--2019-03-18/bin/activate


zcat ${TEST_S} | python -m sockeye.translate --beam-size 5 \
                                             --model ${MODEL} \
                                             --use-cpu \
                                             --method inside-decoder-parallel-attention \
                                             --input-source-doc ${TEST_S} \
                                             --output ${OUT}

deactivate
