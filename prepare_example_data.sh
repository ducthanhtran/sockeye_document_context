#!/bin/bash


BPE_MERGES=32000
VOC_TRESH=16


function pre_tokenization {
    # 1. replace unicode character 'SOFT HYPHEN' (U+00AD) http://www.fileformat.info/info/unicode/char/ad/index.htm
    # 2. Unicode Character 'ZERO WIDTH SPACE' (U+200B) http://www.fileformat.info/info/unicode/char/200B/index.htm
    # 3. Seperate double dash
    zcat ${1} \
        | sed -e 's/\uc2ad/-/g' \
        | sed -e 's/\u200b/-/g' \
        | sed -e 's/--/ -- /g' \
        | gzip \
        > ${2}
}

function post_tokenization {
    # 1. De-escape special characters
    # 2. Squeeze repeating whitespaces
    zcat ${1} \
        | ${TMP_DIR}/mosesdecoder/scripts/tokenizer/deescape-special-chars.perl \
        | tr -s ' ' \
        > ${2}
}

function extract_from_sgml {
    grep -oP '(?<=>).*?(?=</seg>)' ${1} | gzip > ${2}
}


TMP_DIR=./data_preparation/
mkdir -p ${TMP_DIR}


# Download news-commentary.v14 data from WMT 2019 - used for training
wget -nc http://data.statmt.org/news-commentary/v14/training/news-commentary-v14.de-en.tsv.gz -P ${TMP_DIR}

# Download newstest data
wget -nc http://data.statmt.org/wmt19/translation-task/dev.tgz -P ${TMP_DIR}

# Clone mosesdecoder which includes basic text preprocessing tools
if [ ! -d ${TMP_DIR}/mosesdecoder/ ]; then
    git clone git@github.com:moses-smt/mosesdecoder.git ${TMP_DIR}/mosesdecoder/
fi

# Download Byte-pair encoding scripts
wget -nc https://raw.githubusercontent.com/rsennrich/subword-nmt/master/subword_nmt/learn_joint_bpe_and_vocab.py -P ${TMP_DIR}
wget -nc https://raw.githubusercontent.com/rsennrich/subword-nmt/master/subword_nmt/learn_bpe.py -P ${TMP_DIR}
wget -nc https://raw.githubusercontent.com/rsennrich/subword-nmt/master/subword_nmt/apply_bpe.py -P ${TMP_DIR}


# Obtain bilingual training data
zcat ${TMP_DIR}/news-commentary-v14.de-en.tsv.gz | cut -f1 | gzip > ${TMP_DIR}/news-commentary.de-en.de.gz
zcat ${TMP_DIR}/news-commentary-v14.de-en.tsv.gz | cut -f2 | gzip > ${TMP_DIR}/news-commentary.de-en.en.gz

# Obtain bilingual development (newstest2015) and test data (newstest2018)
tar zxf ${TMP_DIR}/dev.tgz -C ${TMP_DIR}/

for y in {5,8}; do
    extract_from_sgml \
        ${TMP_DIR}/dev/newstest201${y}-ende-src.en.sgm \
        ${TMP_DIR}/newstest201${y}.de-en.en.gz
    extract_from_sgml \
        ${TMP_DIR}/dev/newstest201${y}-ende-ref.de.sgm \
        ${TMP_DIR}/newstest201${y}.de-en.de.gz
done


# Tokenize data with including basic text preprocessing
for t in {en,de}; do
    ## Training data
    pre_tokenization \
        ${TMP_DIR}/news-commentary.de-en.${t}.gz \
        ${TMP_DIR}/news-commentary.pre.de-en.${t}.gz

    zcat ${TMP_DIR}/news-commentary.pre.de-en.${t}.gz \
        | ${TMP_DIR}/mosesdecoder/scripts/tokenizer/tokenizer.perl -l ${t} \
        | gzip \
        > ${TMP_DIR}/news-commentary.pre.tok.de-en.${t}.gz

    post_tokenization \
        ${TMP_DIR}/news-commentary.pre.tok.de-en.${t}.gz \
        ${TMP_DIR}/news-commentary.pre.tok.post.de-en.${t}

    ## Newstest data
    for y in {5,8}; do
        pre_tokenization \
            ${TMP_DIR}/newstest201${y}.de-en.${t}.gz \
            ${TMP_DIR}/newstest201${y}.pre.de-en.${t}.gz

        zcat ${TMP_DIR}/newstest201${y}.pre.de-en.${t}.gz \
            | ${TMP_DIR}/mosesdecoder/scripts/tokenizer/tokenizer.perl -l ${t} \
            | gzip \
            > ${TMP_DIR}/newstest201${y}.pre.tok.de-en.${t}.gz

        post_tokenization \
            ${TMP_DIR}/newstest201${y}.pre.tok.de-en.${t}.gz \
            ${TMP_DIR}/newstest201${y}.pre.tok.post.de-en.${t}

    done
done


# Learn and apply byte-pair encoding (BPE)
python ${TMP_DIR}/learn_joint_bpe_and_vocab.py \
    --symbols ${BPE_MERGES} \
    --input ${TMP_DIR}/news-commentary.pre.tok.post.de-en.en ${TMP_DIR}/news-commentary.pre.tok.post.de-en.de \
    --write-vocabulary ${TMP_DIR}/voc.de-en.en ${TMP_DIR}/voc.de-en.de \
    --output ${TMP_DIR}/bpe_joint_codes

for t in {en,de}; do
    ## Training data
    python ${TMP_DIR}/apply_bpe.py \
        --input ${TMP_DIR}/news-commentary.pre.tok.post.de-en.${t} \
        --codes ${TMP_DIR}/bpe_joint_codes \
        --vocabulary ${TMP_DIR}/voc.de-en.${t} \
        --vocabulary-threshold ${VOC_TRESH} \
        --output ${TMP_DIR}/news-commentary.pre.tok.post.bpe.de-en.${t}

    ## Newstest data
    for y in {5,8}; do
        python ${TMP_DIR}/apply_bpe.py \
            --input ${TMP_DIR}/newstest201${y}.pre.tok.post.de-en.${t} \
            --codes ${TMP_DIR}/bpe_joint_codes \
            --vocabulary ${TMP_DIR}/voc.de-en.${t} \
            --vocabulary-threshold ${VOC_TRESH} \
            --output ${TMP_DIR}/newstest201${y}.pre.tok.post.bpe.de-en.${t}
    done
done


# Replace empty lines with a special token _CONTEXT_BREAK_LINE_ to denote document boundaries.
for t in {en,de}; do
    ## Training data
    cat ${TMP_DIR}/news-commentary.pre.tok.post.bpe.de-en.${t} \
        | sed 's/^$/${BOUNDARY_TOKEN}/' \
        | gzip \
        > ${TMP_DIR}/news-commentary.pre.tok.post.bpe.boundary.de-en.${t}.gz

    ## Newstest data
    for y in {5,8}; do
        cat ${TMP_DIR}/newstest201${y}.pre.tok.post.bpe.de-en.${t} \
            | sed 's/^$/${BOUNDARY_TOKEN}/' \
            | gzip \
            > ${TMP_DIR}/newstest201${y}.pre.tok.post.bpe.boundary.de-en.${t}.gz
    done
done

# Dummy data set (10K training samples) for debugging
for t in {en,de}; do
zcat ${TMP_DIR}/news-commentary.pre.tok.post.bpe.boundary.de-en.${t}.gz \
    | head -n 10000 \
    | gzip \
    > ${TMP_DIR}/news-commentary.pre.tok.post.bpe.boundary.10k.de-en.${t}.gz
done