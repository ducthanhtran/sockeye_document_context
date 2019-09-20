# Copyright 2017, 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You may not
# use this file except in compliance with the License. A copy of the License
# is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is distributed on
# an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

"""
Implements data iterators and I/O related functions for sequence-to-sequence models.
"""
import bisect
import logging
import math
import os
import pickle
import random
from abc import ABC, abstractmethod
from collections import OrderedDict
from contextlib import ExitStack
from typing import Any, cast, Dict, Iterator, Iterable, List, Optional, Sequence, Sized, Tuple, Set, Union

import mxnet as mx
import numpy as np

from . import config
from . import constants as C
from . import doc_context
from . import vocab
from .utils import check_condition, smart_open, get_tokens, OnlineMeanAndVariance

logger = logging.getLogger(__name__)


class BucketDoc:
    def __init__(self,
                 src_bkt: int,
                 tar_bkt: int,
                 src_pre_bkts: Optional[List[int]] = None,
                 src_nxt_bkts: Optional[List[int]] = None,
                 tar_pre_bkts: Optional[List[int]] = None,
                 tar_nxt_bkts: Optional[List[int]] = None) -> None:
        self.src_bkt = src_bkt
        self.tar_bkt = tar_bkt
        self.src_pre_bkts = src_pre_bkts if src_pre_bkts is not None else []
        self.src_nxt_bkts = src_nxt_bkts if src_nxt_bkts is not None else []
        self.tar_pre_bkts = tar_pre_bkts if tar_pre_bkts is not None else []
        self.tar_nxt_bkts = tar_nxt_bkts if tar_nxt_bkts is not None else []

    @staticmethod
    def create_from_tuple(tup: Tuple[int, ...], window_config: doc_context.WindowConfig) -> 'BucketDoc':
        tup_entries = list(tup)

        src_pre_size, src_nxt_size, tar_pre_size, tar_nxt_size = window_config.sizes
        sizes = [src_pre_size, 1, src_nxt_size, tar_pre_size, 1, tar_nxt_size]

        bkts = [[] for _ in sizes]
        for size, bkt in zip(sizes, bkts):
            bkt[:] = tup_entries[:size]
            tup_entries[:] = tup_entries[size:]

        return BucketDoc(src_bkt=bkts[1][0],
                         tar_bkt=bkts[4][0],
                         src_pre_bkts=bkts[0],
                         src_nxt_bkts=bkts[2],
                         tar_pre_bkts=bkts[3],
                         tar_nxt_bkts=bkts[5])

    def to_tuple(self) -> Tuple[int, ...]:
        combined = self.src_pre_bkts + [self.src_bkt] + self.src_nxt_bkts \
                   + self.tar_pre_bkts + [self.tar_bkt] + self.tar_nxt_bkts
        return tuple(combined)

    def fits(self,
             length_source: int,
             length_target: int,
             lengths_src_pre: Optional[List[int]],
             lengths_src_nxt: Optional[List[int]],
             lengths_tar_pre: Optional[List[int]],
             lengths_tar_nxt: Optional[List[int]]) -> bool:
        """
        Checks whether bucket can fit given lengths.

        :param length_source: length of current source sentence
        :param length_target: length of current target sentence
        :param lengths_src_pre: lengths of previous source sentences
        :param lengths_src_nxt: lengths of next source sentences
        :param lengths_tar_pre: lengths of previous target sentences
        :param lengths_tar_nxt: lengths of next target sentences
        :return:
        """
        if self.src_bkt >= length_source and self.tar_bkt >= length_target:
            if (self.is_greater_equal(self.src_pre_bkts, lengths_src_pre)
                    and self.is_greater_equal(self.src_nxt_bkts, lengths_src_nxt)
                    and self.is_greater_equal(self.tar_pre_bkts, lengths_tar_pre)
                    and self.is_greater_equal(self.tar_nxt_bkts, lengths_tar_nxt)):
                return True
        return False

    @staticmethod
    def is_greater_equal(bucket: List[int], lengths: Optional[List[int]]) -> bool:
        if lengths is None:
            assert len(bucket) == 0
            return True

        assert len(bucket) == len(lengths)
        for size, length in zip(bucket, lengths):
            if size < length:
                return False
        return True

    @property
    def src_len(self) -> int:
        return self.src_bkt

    @property
    def tar_len(self) -> int:
        return self.tar_bkt

    @tar_len.setter
    def tar_len(self, value) -> None:
        self.tar_bkt = value

    @property
    def src_pre_lens(self) -> Tuple[int, ...]:
        return tuple(self.src_pre_bkts)

    @property
    def src_nxt_lens(self) -> Tuple[int, ...]:
        return tuple(self.src_nxt_bkts)

    @property
    def tar_pre_lens(self) -> Tuple[int, ...]:
        return tuple(self.tar_pre_bkts)

    @property
    def tar_nxt_lens(self) -> Tuple[int, ...]:
        return tuple(self.tar_nxt_bkts)

    def __lt__(self, other):
        return self.to_tuple() < other.to_tuple()

    def __str__(self):
        return str(self.to_tuple())


def define_buckets(max_seq_len: int, step=10) -> List[int]:
    """
    Returns a list of integers defining bucket boundaries.
    Bucket boundaries are created according to the following policy:
    We generate buckets with a step size of step until the final bucket fits max_seq_len.
    We then limit that bucket to max_seq_len (difference between semi-final and final bucket may be less than step).

    :param max_seq_len: Maximum bucket size.
    :param step: Distance between buckets.
    :return: List of bucket sizes.
    """
    buckets = [bucket_len for bucket_len in range(step, max_seq_len + step, step)]
    buckets[-1] = max_seq_len
    return buckets


def define_parallel_buckets(max_seq_len_source: int,
                            max_seq_len_target: int,
                            bucket_width: int = 10,
                            length_ratio: float = 1.0,
                            doc_context_config: Optional[doc_context.DocumentContextConfig] = None) -> \
        Union[List[Tuple[int, int]],
              List[BucketDoc]]:
    """
    Returns (source, target) buckets up to (max_seq_len_source, max_seq_len_target).  The longer side of the data uses
    steps of bucket_width while the shorter side uses steps scaled down by the average target/source length ratio.  If
    one side reaches its max_seq_len before the other, width of extra buckets on that side is fixed to that max_seq_len.

    If we're using additional document context, then TODO doc

    :param max_seq_len_source: Maximum source bucket size.
    :param max_seq_len_target: Maximum target bucket size.
    :param bucket_width: Width of buckets on longer side.
    :param length_ratio: Length ratio of data (target/source).
    :param doc_context_config: Document context config.
    """
    source_step_size = bucket_width
    target_step_size = bucket_width
    if length_ratio >= 1.0:
        # target side is longer -> scale source
        source_step_size = max(1, int(round(bucket_width / length_ratio)))
    else:
        # source side is longer, -> scale target
        target_step_size = max(1, int(round(bucket_width * length_ratio)))
    source_buckets = define_buckets(max_seq_len_source, step=source_step_size)
    target_buckets = define_buckets(max_seq_len_target, step=target_step_size)
    # Extra buckets
    if len(source_buckets) < len(target_buckets):
        source_buckets += [source_buckets[-1] for _ in range(len(target_buckets) - len(source_buckets))]
    elif len(target_buckets) < len(source_buckets):
        target_buckets += [target_buckets[-1] for _ in range(len(source_buckets) - len(target_buckets))]
    # minimum bucket size is 2 (as we add BOS symbol to target side)
    source_buckets = [max(2, b) for b in source_buckets]
    target_buckets = [max(2, b) for b in target_buckets]
    parallel_buckets = list(zip(source_buckets, target_buckets))

    # deduplicate for return
    buckets = list(OrderedDict.fromkeys(parallel_buckets))
    buckets.sort()

    # additional data buckets
    if doc_context_config is not None:
        window_config = doc_context_config.window_config
        source_buckets_additional = []  # type: List[int]
        if window_config.use_source_side:
            source_buckets_additional = define_buckets(
                    max_seq_len_source + 1,
                    step=doc_context_config.bucket_width[0]
            )

        target_buckets_additional = []  # type: List[int]
        if window_config.use_target_side:
            target_buckets_additional = define_buckets(
                    max_seq_len_target + 1,
                    step=doc_context_config.bucket_width[1])

        if source_buckets_additional and len(source_buckets_additional) < len(source_buckets):
            source_buckets_additional += [source_buckets_additional[-1]
                                          for _ in range(len(source_buckets) - len(source_buckets_additional))]
        if target_buckets_additional and len(target_buckets_additional) < len(target_buckets):
            target_buckets_additional += [target_buckets_additional[-1]
                                          for _ in range(len(target_buckets) - len(target_buckets_additional))]

        source_buckets_additional = [max(2, b) for b in source_buckets_additional]
        target_buckets_additional = [max(2, b) for b in target_buckets_additional]

        source_pre_buckets = [source_buckets_additional for _ in range(window_config.src_pre)]
        source_nxt_buckets = [source_buckets_additional for _ in range(window_config.src_nxt)]
        target_pre_buckets = [target_buckets_additional for _ in range(window_config.tar_pre)]
        target_nxt_buckets = [target_buckets_additional for _ in range(window_config.tar_nxt)]

        parallel_buckets = list(zip(*source_pre_buckets, source_buckets, *source_nxt_buckets,
                                    *target_pre_buckets, target_buckets, *target_nxt_buckets))
        bucket_with_context = list(OrderedDict.fromkeys(parallel_buckets))
        bucket_with_context.sort()

        bucket_with_context = [BucketDoc.create_from_tuple(b, window_config) for b in bucket_with_context]
        return bucket_with_context
    return buckets


def define_empty_source_parallel_buckets(max_seq_len_target: int,
                                         bucket_width: int = 10) -> List[Tuple[int, int]]:
    """
    Returns (source, target) buckets up to (None, max_seq_len_target). The source
    is empty since it is supposed to not contain data that can be bucketized.
    The target is used as reference to create the buckets.

    :param max_seq_len_target: Maximum target bucket size.
    :param bucket_width: Width of buckets on longer side.
    """
    target_step_size = max(1, bucket_width)
    target_buckets = define_buckets(max_seq_len_target, step=target_step_size)
    # source buckets are always 0 since there is no text
    source_buckets = [0 for b in target_buckets]
    target_buckets = [max(2, b) for b in target_buckets]
    parallel_buckets = list(zip(source_buckets, target_buckets))
    # deduplicate for return
    buckets = list(OrderedDict.fromkeys(parallel_buckets))
    buckets.sort()
    return buckets


def get_bucket(seq_len: int, buckets: List[int]) -> Optional[int]:
    """
    Given sequence length and a list of buckets, return corresponding bucket.

    :param seq_len: Sequence length.
    :param buckets: List of buckets.
    :return: Chosen bucket.
    """
    bucket_idx = bisect.bisect_left(buckets, seq_len)
    if bucket_idx == len(buckets):
        return None
    return buckets[bucket_idx]


class BucketBatchSize:
    """
    :param bucket: The corresponding bucket.
    :param batch_size: Number of sequences in each batch.
    :param average_words_per_batch: Approximate number of non-padding tokens in each batch.
    """

    def __init__(self, bucket: Tuple[int, int], batch_size: int, average_words_per_batch: float) -> None:
        self.bucket = bucket
        self.batch_size = batch_size
        self.average_words_per_batch = average_words_per_batch


class BucketBatchSizeDoc:
    """
    :param bucket: The corresponding bucket.
    :param batch_size: Number of sequences in each batch.
    :param average_words_per_batch: Approximate number of non-padding tokens in each batch.
    :param average_words_src_pre_per_batch: Approximate number of non-padding tokens in all previous source sentences.
    :param average_words_src_nxt_per_batch: Approximate number of non-padding tokens in all next source sentences.
    :param average_words_tar_pre_per_batch: Approximate number of non-padding tokens in all previous target sentences.
    :param average_words_tar_nxt_per_batch: Approximate number of non-padding tokens in all next target sentences.
    """

    def __init__(self,
                 bucket: BucketDoc,
                 batch_size: int,
                 average_words_per_batch: float,
                 average_words_src_pre_per_batch: float,
                 average_words_src_nxt_per_batch: float,
                 average_words_tar_pre_per_batch: float,
                 average_words_tar_nxt_per_batch: float) -> None:
        self.bucket = bucket
        self.batch_size = batch_size
        self.average_words_per_batch = average_words_per_batch
        self.average_words_src_pre_per_batch = average_words_src_pre_per_batch
        self.average_words_src_nxt_per_batch = average_words_src_nxt_per_batch
        self.average_words_tar_pre_per_batch = average_words_tar_pre_per_batch
        self.average_words_tar_nxt_per_batch = average_words_tar_nxt_per_batch


def define_bucket_batch_sizes(buckets: List[Tuple[int, int]],
                              batch_size: int,
                              batch_by_words: bool,
                              batch_num_devices: int,
                              data_target_average_len: List[Optional[float]]) -> List[BucketBatchSize]:
    """
    Computes bucket-specific batch sizes (sentences, average_words).

    If sentence-based batching: number of sentences is the same for each batch, determines the
    number of words. Hence all batch sizes for each bucket are equal.

    If word-based batching: number of sentences for each batch is set to the multiple of number
    of devices that produces the number of words closest to the target batch size.  Average
    target sentence length (non-padding symbols) is used for word number calculations.

    :param buckets: Bucket list.
    :param batch_size: Batch size.
    :param batch_by_words: Batch by words.
    :param batch_num_devices: Number of devices.
    :param data_target_average_len: Optional average target length for each bucket.
    """
    check_condition(len(data_target_average_len) == len(buckets),
                    "Must provide None or average target length for each bucket")
    data_target_average_len = list(data_target_average_len)
    bucket_batch_sizes = []  # type: List[BucketBatchSize]
    largest_total_num_words = 0
    for buck_idx, bucket in enumerate(buckets):
        # Target/label length with padding
        padded_seq_len = bucket[1]
        # Average target/label length excluding padding
        if data_target_average_len[buck_idx] is None:
            data_target_average_len[buck_idx] = padded_seq_len
        average_seq_len = data_target_average_len[buck_idx]

        # Word-based: num words determines num sentences
        # Sentence-based: num sentences determines num words
        if batch_by_words:
            check_condition(padded_seq_len <= batch_size, "Word batch size must cover sequence lengths for all"
                                                          " buckets: (%d > %d)" % (padded_seq_len, batch_size))
            # Multiple of number of devices (int) closest to target number of words, assuming each sentence is of
            # average length
            batch_size_seq = batch_num_devices * max(1, round((batch_size / average_seq_len) / batch_num_devices))
            batch_size_word = batch_size_seq * average_seq_len
        else:
            batch_size_seq = batch_size
            batch_size_word = batch_size_seq * average_seq_len
        bucket_batch_sizes.append(BucketBatchSize(bucket, batch_size_seq, batch_size_word))
        # Track largest number of source or target word samples in a batch
        largest_total_num_words = max(largest_total_num_words, batch_size_seq * max(*bucket))

    # Final step: guarantee that largest bucket by sequence length also has a batch size so that it covers any
    # (batch_size, len_source) and (batch_size, len_target) matrix from the data iterator to allow for memory sharing.
    # When batching by sentences, this will already be the case.
    if batch_by_words:
        padded_seq_len = max(*buckets[-1])
        average_seq_len = data_target_average_len[-1]
        while bucket_batch_sizes[-1].batch_size * padded_seq_len < largest_total_num_words:
            bucket_batch_sizes[-1] = BucketBatchSize(
                bucket_batch_sizes[-1].bucket,
                bucket_batch_sizes[-1].batch_size + batch_num_devices,
                bucket_batch_sizes[-1].average_words_per_batch + batch_num_devices * average_seq_len)
    return bucket_batch_sizes


def define_bucket_batch_sizes_doc(buckets: List[BucketDoc],
                                  batch_size: int,
                                  batch_by_words: bool,
                                  batch_num_devices: int,
                                  data_target_average_len: List[Optional[float]],
                                  data_src_pre_average_len: List[List[Optional[float]]],
                                  data_src_nxt_average_len: List[List[Optional[float]]],
                                  data_tar_pre_average_len: List[List[Optional[float]]],
                                  data_tar_nxt_average_len: List[List[Optional[float]]]) \
        -> List[BucketBatchSizeDoc]:
    """
    Computes bucket-specific batch sizes (sentences, average_words).

    If sentence-based batching: number of sentences is the same for each batch, determines the
    number of words. Hence all batch sizes for each bucket are equal.

    If word-based batching: number of sentences for each batch is set to the multiple of number
    of devices that produces the number of words closest to the target batch size.  Average
    target sentence length (non-padding symbols) is used for word number calculations. In addition we also utilize
    average additional sentence lengths (non-padding symbols) for word number calculations.

    :param buckets: Bucket list.
    :param batch_size: Batch size.
    :param batch_by_words: Batch by words.
    :param batch_num_devices: Number of devices.
    :param data_target_average_len: Optional average target length for each bucket.
    :param data_src_pre_average_len: Optional average previous source lengths for each bucket.
    :param data_src_nxt_average_len: Optional average next source lengths for each bucket.
    :param data_tar_pre_average_len: Optional average previous target lengths for each bucket.
    :param data_tar_nxt_average_len: Optional average next target lengths for each bucket.
    """
    check_condition(len(data_target_average_len) == len(buckets),
                    "Must provide None or average target length for each bucket")
    data_target_average_len = list(data_target_average_len)
    bucket_batch_sizes = []  # type: List[BucketBatchSizeDocLevel]
    largest_total_num_words = 0
    for buck_idx, bucket in enumerate(buckets):
        # Target/label length with padding
        padded_seq_len = bucket.tar_bkt
        # Average target/label length excluding padding
        if data_target_average_len[buck_idx] is None:
            data_target_average_len[buck_idx] = padded_seq_len

        for data_average_lens, padded_seq_lens in zip([data_src_pre_average_len[buck_idx],
                                                       data_src_nxt_average_len[buck_idx],
                                                       data_tar_pre_average_len[buck_idx],
                                                       data_tar_nxt_average_len[buck_idx]],
                                                      [bucket.src_pre_bkts, bucket.src_nxt_bkts,
                                                       bucket.tar_pre_bkts, bucket.tar_nxt_bkts]):
            if padded_seq_lens:
                assert len(data_average_lens) == len(padded_seq_lens)
                if all(avg is None for avg in data_average_lens):
                    data_average_lens[:] = padded_seq_lens  # assign padded_seq_len as values

        average_seq_len = data_target_average_len[buck_idx]
        average_src_pre_seq_lens = data_src_pre_average_len[buck_idx]
        average_src_nxt_seq_lens = data_src_nxt_average_len[buck_idx]
        average_tar_pre_seq_lens = data_tar_pre_average_len[buck_idx]
        average_tar_nxt_seq_lens = data_tar_nxt_average_len[buck_idx]

        # adjust average length to take into account the additional lengths
        average_seq_len += sum(average_src_pre_seq_lens + average_src_nxt_seq_lens
                               + average_tar_pre_seq_lens + average_tar_nxt_seq_lens)

        # Word-based: num words determines num sentences
        # Sentence-based: num sentences determines num words
        if batch_by_words:
            check_condition(padded_seq_len <= batch_size, "Word batch size must cover sequence lengths for all"
                                                          " buckets: (%d > %d)" % (padded_seq_len, batch_size))
            # Multiple of number of devices (int) closest to target number of words, assuming each sentence is of
            # average length
            batch_size_seq = batch_num_devices * max(1,
                                                     round(
                                                             (batch_size / average_seq_len) / batch_num_devices
                                                     ))
            batch_size_word = batch_size_seq * average_seq_len
        else:
            batch_size_seq = batch_size
            batch_size_word = batch_size_seq * average_seq_len
        bucket_batch_sizes.append(BucketBatchSizeDoc(bucket,
                                                     batch_size_seq,
                                                     batch_size_word,
                                                     batch_size_seq * sum(average_src_pre_seq_lens),
                                                     batch_size_seq * sum(average_src_nxt_seq_lens),
                                                     batch_size_seq * sum(average_tar_pre_seq_lens),
                                                     batch_size_seq * sum(average_tar_nxt_seq_lens)))
        # Track largest number of source or target word samples in a batch
        largest_total_num_words = max(largest_total_num_words, batch_size_seq * max(*bucket.to_tuple()))

    # Final step: guarantee that largest bucket by sequence length also has a batch size so that it covers any
    # (batch_size, len_source) and (batch_size, len_target) matrix from the data iterator to allow for memory sharing.
    # Moreover, we consider the additional data from the document-level context window as well.
    # When batching by sentences, this will already be the case.
    if batch_by_words:
        padded_seq_len = max(*buckets[-1].to_tuple())

        average_seq_len = data_target_average_len[-1]
        average_src_pre_seq_lens = data_src_pre_average_len[-1]
        average_src_nxt_seq_lens = data_src_nxt_average_len[-1]
        average_tar_pre_seq_lens = data_tar_pre_average_len[-1]
        average_tar_nxt_seq_lens = data_tar_nxt_average_len[-1]

        average_seq_len += sum(average_src_pre_seq_lens + average_src_nxt_seq_lens
                               + average_tar_pre_seq_lens + average_tar_nxt_seq_lens)

        increased_counter = 0
        while bucket_batch_sizes[-1].batch_size * padded_seq_len < largest_total_num_words:
            increased_counter += 1
            bucket_batch_sizes[-1] = BucketBatchSizeDoc(
                    bucket_batch_sizes[-1].bucket,
                    bucket_batch_sizes[-1].batch_size + batch_num_devices,
                    bucket_batch_sizes[-1].average_words_per_batch + batch_num_devices * average_seq_len,
                    bucket_batch_sizes[-1].average_words_src_pre_per_batch
                    + batch_num_devices * sum(average_src_pre_seq_lens),
                    bucket_batch_sizes[-1].average_words_src_pre_per_batch
                    + batch_num_devices * sum(average_src_nxt_seq_lens),
                    bucket_batch_sizes[-1].average_words_src_pre_per_batch
                    + batch_num_devices * sum(average_tar_pre_seq_lens),
                    bucket_batch_sizes[-1].average_words_src_pre_per_batch
                    + batch_num_devices * sum(average_tar_nxt_seq_lens),
            )

        logger.info("Largest total number of tokens: {}. Increased last bucket batch size by {} to "
                    "ensure memory sharing is handled.".format(largest_total_num_words, increased_counter))
    return bucket_batch_sizes


def calculate_length_statistics(source_iterables: Sequence[Iterable[Any]],
                                target_iterable: Iterable[Any],
                                max_seq_len_source: int,
                                max_seq_len_target: int) -> 'LengthStatistics':
    """
    Returns mean and standard deviation of target-to-source length ratios of parallel corpus.

    :param source_iterables: Source sequence readers.
    :param target_iterable: Target sequence reader.
    :param max_seq_len_source: Maximum source sequence length.
    :param max_seq_len_target: Maximum target sequence length.
    :return: The number of sentences as well as the mean and standard deviation of target to source length ratios.
    """
    mean_and_variance = OnlineMeanAndVariance()

    for sources, target in parallel_iter(source_iterables, target_iterable):
        source_len = len(sources[0])
        target_len = len(target)
        if source_len > max_seq_len_source or target_len > max_seq_len_target:
            continue

        length_ratio = target_len / source_len
        mean_and_variance.update(length_ratio)

    num_sents = mean_and_variance.count
    mean = mean_and_variance.mean
    if not math.isnan(mean_and_variance.variance):
        std = math.sqrt(mean_and_variance.variance)
    else:
        std = 0.0
    return LengthStatistics(num_sents, mean, std)


def analyze_sequence_lengths(sources: List[str],
                             target: str,
                             vocab_sources: List[vocab.Vocab],
                             vocab_target: vocab.Vocab,
                             max_seq_len_source: int,
                             max_seq_len_target: int) -> 'LengthStatistics':
    train_sources_sentences, train_target_sentences = create_sequence_readers(sources, target, vocab_sources,
                                                                              vocab_target)

    length_statistics = calculate_length_statistics(train_sources_sentences, train_target_sentences,
                                                    max_seq_len_source,
                                                    max_seq_len_target)

    logger.info("%d sequences of maximum length (%d, %d) in '%s' and '%s'.",
                length_statistics.num_sents, max_seq_len_source, max_seq_len_target, sources[0], target)
    logger.info("Mean training target/source length ratio: %.2f (+-%.2f)",
                length_statistics.length_ratio_mean,
                length_statistics.length_ratio_std)
    return length_statistics


def are_none(sequences: Sequence[Sized]) -> bool:
    """
    Returns True if all sequences are None.
    """
    if not sequences:
        return True
    return all(s is None for s in sequences)


def are_token_parallel(sequences: Sequence[Sized]) -> bool:
    """
    Returns True if all sequences in the list have the same length.
    """
    if not sequences or len(sequences) == 1:
        return True
    return all(len(s) == len(sequences[0]) for s in sequences)


class DataStatisticsAccumulator:

    def __init__(self,
                 buckets: List[Tuple[int, int]],
                 vocab_source: Optional[Dict[str, int]],
                 vocab_target: Dict[str, int],
                 length_ratio_mean: float,
                 length_ratio_std: float) -> None:
        self.buckets = buckets
        num_buckets = len(buckets)
        self.length_ratio_mean = length_ratio_mean
        self.length_ratio_std = length_ratio_std
        if vocab_source is not None:
            self.unk_id_source = vocab_source[C.UNK_SYMBOL]
            self.size_vocab_source = len(vocab_source)
        else:
            self.unk_id_source = None
            self.size_vocab_source = 0
        self.unk_id_target = vocab_target[C.UNK_SYMBOL]
        self.size_vocab_target = len(vocab_target)
        self.num_sents = 0
        self.num_discarded = 0
        self.num_tokens_source = 0
        self.num_tokens_target = 0
        self.num_unks_source = 0
        self.num_unks_target = 0
        self.max_observed_len_source = 0
        self.max_observed_len_target = 0
        self._mean_len_target_per_bucket = [OnlineMeanAndVariance() for _ in range(num_buckets)]

    def sequence_pair(self,
                      source: List[int],
                      target: List[int],
                      bucket_idx: Optional[int]):
        if bucket_idx is None:
            self.num_discarded += 1
            return

        source_len = len(source)
        target_len = len(target)

        self._mean_len_target_per_bucket[bucket_idx].update(target_len)

        self.num_sents += 1
        self.num_tokens_source += source_len
        self.num_tokens_target += target_len
        self.max_observed_len_source = max(source_len, self.max_observed_len_source)
        self.max_observed_len_target = max(target_len, self.max_observed_len_target)

        if self.unk_id_source is not None:
            self.num_unks_source += source.count(self.unk_id_source)
        self.num_unks_target += target.count(self.unk_id_target)

    @property
    def mean_len_target_per_bucket(self) -> List[Optional[float]]:
        return [mean_and_variance.mean if mean_and_variance.count > 0 else None
                for mean_and_variance in self._mean_len_target_per_bucket]

    @property
    def statistics(self):
        num_sents_per_bucket = [mean_and_variance.count for mean_and_variance in self._mean_len_target_per_bucket]
        return DataStatistics(num_sents=self.num_sents,
                              num_discarded=self.num_discarded,
                              num_tokens_source=self.num_tokens_source,
                              num_tokens_target=self.num_tokens_target,
                              num_unks_source=self.num_unks_source,
                              num_unks_target=self.num_unks_target,
                              max_observed_len_source=self.max_observed_len_source,
                              max_observed_len_target=self.max_observed_len_target,
                              size_vocab_source=self.size_vocab_source,
                              size_vocab_target=self.size_vocab_target,
                              length_ratio_mean=self.length_ratio_mean,
                              length_ratio_std=self.length_ratio_std,
                              buckets=self.buckets,
                              num_sents_per_bucket=num_sents_per_bucket,
                              mean_len_target_per_bucket=self.mean_len_target_per_bucket)


def shard_data(source_fnames: List[str],
               target_fname: str,
               source_vocabs: List[vocab.Vocab],
               target_vocab: vocab.Vocab,
               num_shards: int,
               buckets: List[Tuple[int, int]],
               length_ratio_mean: float,
               length_ratio_std: float,
               output_prefix: str) -> Tuple[List[Tuple[List[str], str, 'DataStatistics']], 'DataStatistics']:
    """
    Assign int-coded source/target sentence pairs to shards at random.

    :param source_fnames: The path to the source text (and optional token-parallel factor files).
    :param target_fname: The file name of the target file.
    :param source_vocabs: Source vocabulary (and optional source factor vocabularies).
    :param target_vocab: Target vocabulary.
    :param num_shards: The total number of shards.
    :param buckets: Bucket list.
    :param length_ratio_mean: Mean length ratio.
    :param length_ratio_std: Standard deviation of length ratios.
    :param output_prefix: The prefix under which the shard files will be created.
    :return: Tuple of source (and source factor) file names, target file names and statistics for each shard,
             as well as global statistics.
    """
    os.makedirs(output_prefix, exist_ok=True)
    sources_shard_fnames = [[os.path.join(output_prefix, C.SHARD_SOURCE % i) + ".%d" % f for i in range(num_shards)]
                            for f in range(len(source_fnames))]
    target_shard_fnames = [os.path.join(output_prefix, C.SHARD_TARGET % i)
                           for i in range(num_shards)]  # type: List[str]

    data_stats_accumulator = DataStatisticsAccumulator(buckets, source_vocabs[0], target_vocab,
                                                       length_ratio_mean, length_ratio_std)
    per_shard_stat_accumulators = [DataStatisticsAccumulator(buckets, source_vocabs[0], target_vocab, length_ratio_mean,
                                                             length_ratio_std) for shard_idx in range(num_shards)]

    with ExitStack() as exit_stack:
        sources_shards = [[exit_stack.enter_context(smart_open(f, mode="wt")) for f in sources_shard_fnames[i]] for i in
                          range(len(source_fnames))]
        target_shards = [exit_stack.enter_context(smart_open(f, mode="wt")) for f in target_shard_fnames]

        source_readers, target_reader = create_sequence_readers(source_fnames, target_fname,
                                                                source_vocabs, target_vocab)

        random_shard_iter = iter(lambda: random.randrange(num_shards), None)

        for (sources, target), random_shard_index in zip(parallel_iter(source_readers, target_reader),
                                                         random_shard_iter):
            random_shard_index = cast(int, random_shard_index)
            source_len = len(sources[0])
            target_len = len(target)

            buck_idx, buck = get_parallel_bucket(buckets, source_len, target_len)
            data_stats_accumulator.sequence_pair(sources[0], target, buck_idx)
            per_shard_stat_accumulators[random_shard_index].sequence_pair(sources[0], target, buck_idx)

            if buck is None:
                continue

            for i, line in enumerate(sources):
                sources_shards[i][random_shard_index].write(ids2strids(line) + "\n")
            target_shards[random_shard_index].write(ids2strids(target) + "\n")

    per_shard_stats = [shard_stat_accumulator.statistics for shard_stat_accumulator in per_shard_stat_accumulators]

    sources_shard_fnames_by_shards = zip(*sources_shard_fnames)  # type: List[List[str]]

    return list(
        zip(sources_shard_fnames_by_shards, target_shard_fnames, per_shard_stats)), data_stats_accumulator.statistics


class RawParallelDatasetLoader:
    """
    Loads a data set of variable-length parallel source/target sequences into buckets of NDArrays.

    :param buckets: Bucket list.
    :param eos_id: End-of-sentence id.
    :param pad_id: Padding id.
    :param eos_id: Unknown id.
    :param skip_blanks: Whether to skip blank lines.
    :param dtype: Data type.
    """

    def __init__(self,
                 buckets: List[Tuple[int, int]],
                 eos_id: int,
                 pad_id: int,
                 skip_blanks: bool = True,
                 dtype: str = 'float32') -> None:
        self.buckets = buckets
        self.eos_id = eos_id
        self.pad_id = pad_id
        self.skip_blanks = skip_blanks
        self.dtype = dtype

    def load(self,
             source_iterables: Sequence[Iterable],
             target_iterable: Iterable,
             num_samples_per_bucket: List[int]) -> 'ParallelDataSet':

        assert len(num_samples_per_bucket) == len(self.buckets)
        num_factors = len(source_iterables)

        data_source = [np.full((num_samples, source_len, num_factors), self.pad_id, dtype=self.dtype)
                       for (source_len, target_len), num_samples in zip(self.buckets, num_samples_per_bucket)]
        data_target = [np.full((num_samples, target_len), self.pad_id, dtype=self.dtype)
                       for (source_len, target_len), num_samples in zip(self.buckets, num_samples_per_bucket)]
        data_label = [np.full((num_samples, target_len), self.pad_id, dtype=self.dtype)
                      for (source_len, target_len), num_samples in zip(self.buckets, num_samples_per_bucket)]

        bucket_sample_index = [0 for _ in self.buckets]

        # track amount of padding introduced through bucketing
        num_tokens_source = 0
        num_tokens_target = 0
        num_pad_source = 0
        num_pad_target = 0

        # Bucket sentences as padded np arrays
        for sentno, (sources, target) in enumerate(parallel_iter(source_iterables, target_iterable, skip_blanks=self.skip_blanks), 1):
            sources = [[] if stream is None else stream for stream in sources]
            if target is None:
                target = []
            source_len = len(sources[0])
            target_len = len(target)
            buck_index, buck = get_parallel_bucket(self.buckets, source_len, target_len)
            if buck is None:
                if self.skip_blanks:
                    continue  # skip this sentence pair
                else:
                    buck_index = len(self.buckets)
                    buck = self.buckets[buck_index]

            num_tokens_source += buck[0]
            num_tokens_target += buck[1]
            num_pad_source += buck[0] - source_len
            num_pad_target += buck[1] - target_len

            sample_index = bucket_sample_index[buck_index]
            for i, s in enumerate(sources):
                data_source[buck_index][sample_index, 0:source_len, i] = s
            data_target[buck_index][sample_index, :target_len] = target
            # NOTE(fhieber): while this is wasteful w.r.t memory, we need to explicitly create the label sequence
            # with the EOS symbol here sentence-wise and not per-batch due to variable sequence length within a batch.
            # Once MXNet allows item assignments given a list of indices (probably MXNet 1.0): e.g a[[0,1,5,2]] = x,
            # we can try again to compute the label sequence on the fly in next().
            data_label[buck_index][sample_index, :target_len] = target[1:] + [self.eos_id]

            bucket_sample_index[buck_index] += 1

        for i in range(len(data_source)):
            data_source[i] = mx.nd.array(data_source[i], dtype=self.dtype)
            data_target[i] = mx.nd.array(data_target[i], dtype=self.dtype)
            data_label[i] = mx.nd.array(data_label[i], dtype=self.dtype)

        if num_tokens_source > 0 and num_tokens_target > 0:
            logger.info("Created bucketed parallel data set. Introduced padding: source=%.1f%% target=%.1f%%)",
                        num_pad_source / num_tokens_source * 100,
                        num_pad_target / num_tokens_target * 100)

        return ParallelDataSet(data_source, data_target, data_label)


class RawParallelDatasetLoaderDoc:
    """
    Loads a data set of variable-length parallel source/target sequences into buckets of NDArrays including context data.

    :param window_config: Window config.
    :param buckets: Bucket list.
    :param eos_id: End-of-sentence id.
    :param pad_id: Padding id.
    :param eos_id: Unknown id.
    :param skip_blanks: Whether to skip blank lines.
    :param dtype: Data type.
    """

    def __init__(self,
                 window_config: doc_context.WindowConfig,
                 buckets: List[BucketDoc],
                 eos_id: int,
                 pad_id: int,
                 skip_blanks: bool = True,
                 dtype: str = 'float32') -> None:
        self.window_config = window_config
        self.buckets = buckets
        self.eos_id = eos_id
        self.pad_id = pad_id
        self.skip_blanks = skip_blanks
        self.dtype = dtype

    def load(self,
             source_iterables: Iterable,
             target_iterable: Iterable,
             source_pre_iterable: List['SequenceReaderDoc'],
             source_nxt_iterable: List['SequenceReaderDoc'],
             target_pre_iterable: List['SequenceReaderDoc'],
             target_nxt_iterable: List['SequenceReaderDoc'],
             num_samples_per_bucket: List[int]) -> 'ParallelDataSetDoc':

        assert len(num_samples_per_bucket) == len(self.buckets)

        data_source, data_target, data_label, data_source_pre, data_source_nxt, data_target_pre, data_target_nxt = \
            self.create_data_matrices_doc(num_samples_per_bucket=num_samples_per_bucket, num_factors=1)

        bucket_sample_index = [0 for _ in self.buckets]

        # track amount of padding introduced through bucketing
        num_tokens_source = 0
        num_tokens_target = 0
        num_pad_source = 0
        num_pad_target = 0

        # Bucket sentences as padded np arrays
        for sentno, (source_pre, sources, source_nxt, target_pre, target, target_nxt) in \
                enumerate(parallel_iter_doc_level(source_iterables, target_iterable,
                                                  source_pre_iterable, source_nxt_iterable,
                                                  target_pre_iterable, target_nxt_iterable,
                                                  skip_blanks=self.skip_blanks),
                          1):
            source_pre_lens, source_nxt_lens, target_pre_lens, target_nxt_lens = \
                get_additional_data_lengths(source_pre,
                                            source_nxt,
                                            target_pre,
                                            target_nxt)

            sources = [] if sources is None else sources
            if target is None:
                target = []
            source_len = len(sources)
            target_len = len(target)
            buck_index, buck = get_parallel_bucket_doc(self.buckets, source_len, target_len,
                                                       source_pre_lens, source_nxt_lens,
                                                       target_pre_lens, target_nxt_lens)
            if buck is None:
                if self.skip_blanks:
                    continue  # skip this sentence pair
                else:
                    raise NotImplementedError

            num_tokens_source += buck.src_len
            num_tokens_target += buck.tar_len
            num_pad_source += buck.src_len - source_len
            num_pad_target += buck.tar_len - target_len

            sample_index = bucket_sample_index[buck_index]
            data_source[buck_index][sample_index, 0:source_len, 0] = sources
            data_target[buck_index][sample_index, :target_len] = target
            # NOTE(fhieber): while this is wasteful w.r.t memory, we need to explicitly create the label sequence
            # with the EOS symbol here sentence-wise and not per-batch due to variable sequence length within a batch.
            # Once MXNet allows item assignments given a list of indices (probably MXNet 1.0): e.g a[[0,1,5,2]] = x,
            # we can try again to compute the label sequence on the fly in next().
            data_label[buck_index][sample_index, :target_len] = target[1:] + [self.eos_id]

            # add all additional sentences into data matrices
            for sents_additional, data_additional in zip([source_pre, source_nxt, target_pre, target_nxt],
                                                         [data_source_pre, data_source_nxt,
                                                          data_target_pre, data_target_nxt]):
                for index_additional, sent_additional in enumerate(sents_additional):
                    data_additional[buck_index][index_additional][sample_index, 0:len(sent_additional), 0] = \
                        sent_additional

            bucket_sample_index[buck_index] += 1

        for i in range(len(data_source)):
            data_source[i] = mx.nd.array(data_source[i], dtype=self.dtype)
            data_target[i] = mx.nd.array(data_target[i], dtype=self.dtype)
            data_label[i] = mx.nd.array(data_label[i], dtype=self.dtype)
            data_source_pre[i] = [mx.nd.array(data_source_pre_entry, dtype=self.dtype)
                                  for data_source_pre_entry in data_source_pre[i]]
            data_source_nxt[i] = [mx.nd.array(data_source_nxt_entry, dtype=self.dtype)
                                  for data_source_nxt_entry in data_source_nxt[i]]
            data_target_pre[i] = [mx.nd.array(target_pre_entry, dtype=self.dtype)
                                  for target_pre_entry in data_target_pre[i]]
            data_target_nxt[i] = [mx.nd.array(target_pre_entry, dtype=self.dtype)
                                  for target_pre_entry in data_target_nxt[i]]

        if num_tokens_source > 0 and num_tokens_target > 0:
            logger.info("Created bucketed parallel data set. Introduced padding: source=%.1f%% target=%.1f%%)",
                        num_pad_source / num_tokens_source * 100,
                        num_pad_target / num_tokens_target * 100)

        return ParallelDataSetDoc(data_source, data_target, data_label,
                                  data_source_pre, data_source_nxt,
                                  data_target_pre, data_target_nxt)

    def create_data_matrices_doc(self,
                             num_samples_per_bucket: List[int],
                             num_factors: int) \
            -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray],
                     List[List[np.ndarray]], List[List[np.ndarray]], List[List[np.ndarray]], List[List[np.ndarray]]]:
        """
        Creates data matrices which hold training data as integer sequences including document context information.

        :param num_samples_per_bucket: Number of samples per bucket.
        :param num_factors: Factor size for source sentences.
        :return: Matrices for all possible sentences that can be filled with integer sequences from training data.
        """
        data_source = [np.full((num_samples, bucket.src_len, num_factors), self.pad_id, dtype=self.dtype)
                       for bucket, num_samples in zip(self.buckets, num_samples_per_bucket)]
        data_target = [np.full((num_samples, bucket.tar_len), self.pad_id, dtype=self.dtype)
                       for bucket, num_samples in zip(self.buckets, num_samples_per_bucket)]
        data_label = [np.full((num_samples, bucket.tar_len), self.pad_id, dtype=self.dtype)
                      for bucket, num_samples in zip(self.buckets, num_samples_per_bucket)]

        data_source_pre = [[np.full((num_samples, src_pre_len, num_factors), self.pad_id, dtype=self.dtype)
                            for src_pre_len in bucket.src_pre_lens]
                           for bucket, num_samples in zip(self.buckets, num_samples_per_bucket)]
        data_source_nxt = [[np.full((num_samples, src_nxt_len, num_factors), self.pad_id, dtype=self.dtype)
                            for src_nxt_len in bucket.src_nxt_lens]
                           for bucket, num_samples in zip(self.buckets, num_samples_per_bucket)]

        data_target_pre = [[np.full((num_samples, tar_pre_len, num_factors), self.pad_id, dtype=self.dtype)
                            for tar_pre_len in bucket.tar_pre_lens]
                           for bucket, num_samples in zip(self.buckets, num_samples_per_bucket)]
        data_target_nxt = [[np.full((num_samples, tar_nxt_len, num_factors), self.pad_id, dtype=self.dtype)
                            for tar_nxt_len in bucket.tar_nxt_lens]
                           for bucket, num_samples in zip(self.buckets, num_samples_per_bucket)]

        return data_source, data_target, data_label, data_source_pre, data_source_nxt, data_target_pre, data_target_nxt


def get_num_shards(num_samples: int, samples_per_shard: int, min_num_shards: int) -> int:
    """
    Returns the number of shards.

    :param num_samples: Number of training data samples.
    :param samples_per_shard: Samples per shard.
    :param min_num_shards: Minimum number of shards.
    :return: Number of shards.
    """
    return max(int(math.ceil(num_samples / samples_per_shard)), min_num_shards)


def prepare_data(source_fnames: List[str],
                 target_fname: str,
                 source_vocabs: List[vocab.Vocab],
                 target_vocab: vocab.Vocab,
                 source_vocab_paths: List[Optional[str]],
                 target_vocab_path: Optional[str],
                 shared_vocab: bool,
                 max_seq_len_source: int,
                 max_seq_len_target: int,
                 bucketing: bool,
                 bucket_width: int,
                 samples_per_shard: int,
                 min_num_shards: int,
                 output_prefix: str,
                 keep_tmp_shard_files: bool = False):
    logger.info("Preparing data.")
    # write vocabularies to data folder
    vocab.save_source_vocabs(source_vocabs, output_prefix)
    vocab.save_target_vocab(target_vocab, output_prefix)

    # Pass 1: get target/source length ratios.
    length_statistics = analyze_sequence_lengths(source_fnames, target_fname, source_vocabs, target_vocab,
                                                 max_seq_len_source, max_seq_len_target)

    check_condition(length_statistics.num_sents > 0,
                    "No training sequences found with length smaller or equal than the maximum sequence length."
                    "Consider increasing %s" % C.TRAINING_ARG_MAX_SEQ_LEN)

    # define buckets
    buckets = define_parallel_buckets(max_seq_len_source, max_seq_len_target, bucket_width,
                                      length_statistics.length_ratio_mean) if bucketing else [
        (max_seq_len_source, max_seq_len_target)]
    logger.info("Buckets: %s", buckets)

    # Pass 2: Randomly assign data to data shards
    # no pre-processing yet, just write the sentences to different files
    num_shards = get_num_shards(length_statistics.num_sents, samples_per_shard, min_num_shards)
    logger.info("%d samples will be split into %d shard(s) (requested samples/shard=%d, min_num_shards=%d)."
                % (length_statistics.num_sents, num_shards, samples_per_shard, min_num_shards))
    shards, data_statistics = shard_data(source_fnames=source_fnames,
                                         target_fname=target_fname,
                                         source_vocabs=source_vocabs,
                                         target_vocab=target_vocab,
                                         num_shards=num_shards,
                                         buckets=buckets,
                                         length_ratio_mean=length_statistics.length_ratio_mean,
                                         length_ratio_std=length_statistics.length_ratio_std,
                                         output_prefix=output_prefix)
    data_statistics.log()

    data_loader = RawParallelDatasetLoader(buckets=buckets,
                                           eos_id=target_vocab[C.EOS_SYMBOL],
                                           pad_id=C.PAD_ID)

    # 3. convert each shard to serialized ndarrays
    for shard_idx, (shard_sources, shard_target, shard_stats) in enumerate(shards):
        sources_sentences = [SequenceReader(s) for s in shard_sources]
        target_sentences = SequenceReader(shard_target)
        dataset = data_loader.load(sources_sentences, target_sentences, shard_stats.num_sents_per_bucket)
        shard_fname = os.path.join(output_prefix, C.SHARD_NAME % shard_idx)
        shard_stats.log()
        logger.info("Writing '%s'", shard_fname)
        dataset.save(shard_fname)

        if not keep_tmp_shard_files:
            for f in shard_sources:
                os.remove(f)
            os.remove(shard_target)

    data_info = DataInfo(sources=[os.path.abspath(fname) for fname in source_fnames],
                         target=os.path.abspath(target_fname),
                         source_vocabs=source_vocab_paths,
                         target_vocab=target_vocab_path,
                         shared_vocab=shared_vocab,
                         num_shards=num_shards)
    data_info_fname = os.path.join(output_prefix, C.DATA_INFO)
    logger.info("Writing data info to '%s'", data_info_fname)
    data_info.save(data_info_fname)

    config_data = DataConfig(data_statistics=data_statistics,
                             max_seq_len_source=max_seq_len_source,
                             max_seq_len_target=max_seq_len_target,
                             num_source_factors=len(source_fnames),
                             source_with_eos=True)
    config_data_fname = os.path.join(output_prefix, C.DATA_CONFIG)
    logger.info("Writing data config to '%s'", config_data_fname)
    config_data.save(config_data_fname)

    version_file = os.path.join(output_prefix, C.PREPARED_DATA_VERSION_FILE)

    with open(version_file, "w") as version_out:
        version_out.write(str(C.PREPARED_DATA_VERSION))


def get_data_statistics(source_readers: Optional[Sequence[Iterable]],
                        target_reader: Iterable,
                        buckets: List[Tuple[int, int]],
                        length_ratio_mean: float,
                        length_ratio_std: float,
                        source_vocabs: Optional[List[vocab.Vocab]],
                        target_vocab: vocab.Vocab) -> 'DataStatistics':
    data_stats_accumulator = DataStatisticsAccumulator(buckets,
                                                       source_vocabs[0] if source_vocabs is not None else None,
                                                       target_vocab,
                                                       length_ratio_mean,
                                                       length_ratio_std)

    if source_readers is not None:
        for sources, target in parallel_iter(source_readers, target_reader):
            buck_idx, buck = get_parallel_bucket(buckets, len(sources[0]), len(target))
            data_stats_accumulator.sequence_pair(sources[0], target, buck_idx)
    else:  # Allow stats for target only data
        for target in target_reader:
            buck_idx, buck = get_target_bucket(buckets, len(target))
            data_stats_accumulator.sequence_pair([], target, buck_idx)

    return data_stats_accumulator.statistics


def get_data_statistics_doc(doc_context_config: doc_context.DocumentContextConfig,
                            src_pre_reader: List['SequenceReaderDoc'],
                            src_nxt_reader: List['SequenceReaderDoc'],
                            tar_pre_reader: List['SequenceReaderDoc'],
                            tar_nxt_reader: List['SequenceReaderDoc'],
                            src_readers: Iterable,
                            tar_reader: Iterable,
                            buckets: List[BucketDoc],
                            length_ratio_mean: float,
                            length_ratio_std: float,
                            source_vocabs: Optional[List[vocab.Vocab]],
                            target_vocab: vocab.Vocab) -> 'DataStatisticsDoc':
    data_stats_accumulator = DataStatisticsAccumulatorDoc(buckets,
                                                          doc_context_config.window_config,
                                                          source_vocabs[0] if source_vocabs is not None else None,
                                                          target_vocab,
                                                          length_ratio_mean,
                                                          length_ratio_std)

    if src_readers is not None:
        for q, (source_pre, sources, source_nxt, target_pre, target, target_nxt) in \
                enumerate(parallel_iter_doc_level(src_readers, tar_reader,
                                                  src_pre_reader, src_nxt_reader,
                                                  tar_pre_reader, tar_nxt_reader)):
            source_pre_lens, source_nxt_lens, target_pre_lens, target_nxt_lens = \
                get_additional_data_lengths(source_pre,
                                            source_nxt,
                                            target_pre,
                                            target_nxt)
            buck_idx, buck = get_parallel_bucket_doc(buckets,
                                                     len(sources),
                                                     len(target),
                                                     source_pre_lens,
                                                     source_nxt_lens,
                                                     target_pre_lens,
                                                     target_nxt_lens)
            data_stats_accumulator.sequence_pair(sources, target,
                                                 source_pre_lens, source_nxt_lens,
                                                 target_pre_lens, target_nxt_lens,
                                                 buck_idx)
    else:
        raise NotImplementedError("Not supporting target only data.")

    return data_stats_accumulator.statistics


def get_validation_data_iter(data_loader: RawParallelDatasetLoader,
                             validation_sources: List[str],
                             validation_target: str,
                             buckets: List[Tuple[int, int]],
                             bucket_batch_sizes: List[BucketBatchSize],
                             source_vocabs: List[vocab.Vocab],
                             target_vocab: vocab.Vocab,
                             max_seq_len_source: int,
                             max_seq_len_target: int,
                             batch_size: int) -> 'ParallelSampleIter':
    """
    Returns a ParallelSampleIter for the validation data.
    """
    logger.info("=================================")
    logger.info("Creating validation data iterator")
    logger.info("=================================")
    validation_length_statistics = analyze_sequence_lengths(validation_sources, validation_target,
                                                            source_vocabs, target_vocab,
                                                            max_seq_len_source, max_seq_len_target)

    check_condition(validation_length_statistics.num_sents > 0,
                    "No validation sequences found with length smaller or equal than the maximum sequence length."
                    "Consider increasing %s" % C.TRAINING_ARG_MAX_SEQ_LEN)

    validation_sources_sentences, validation_target_sentences = create_sequence_readers(validation_sources,
                                                                                        validation_target,
                                                                                        source_vocabs, target_vocab)

    validation_data_statistics = get_data_statistics(validation_sources_sentences,
                                                     validation_target_sentences,
                                                     buckets,
                                                     validation_length_statistics.length_ratio_mean,
                                                     validation_length_statistics.length_ratio_std,
                                                     source_vocabs, target_vocab)

    validation_data_statistics.log(bucket_batch_sizes)

    validation_data = data_loader.load(validation_sources_sentences, validation_target_sentences,
                                       validation_data_statistics.num_sents_per_bucket).fill_up(bucket_batch_sizes)

    return ParallelSampleIter(data=validation_data,
                              buckets=buckets,
                              batch_size=batch_size,
                              bucket_batch_sizes=bucket_batch_sizes,
                              num_factors=len(validation_sources))


def get_validation_data_iter_doc(doc_context_config: doc_context.DocumentContextConfig,
                                 data_loader: RawParallelDatasetLoaderDoc,
                                 validation_sources: List[str],
                                 validation_target: str,
                                 buckets: List[BucketDoc],
                                 bucket_batch_sizes: List[BucketBatchSizeDoc],
                                 source_vocabs: List[vocab.Vocab],
                                 target_vocab: vocab.Vocab,
                                 max_seq_len_source: int,
                                 max_seq_len_target: int,
                                 batch_size: int) -> 'ParallelSampleIterDoc':
    """
    Returns a ParallelSampleIter for the validation data, including additional document context information.
    """
    logger.info("=================================")
    logger.info("Creating validation data iterator including document context information.")
    logger.info("=================================")
    validation_length_statistics = analyze_sequence_lengths(validation_sources, validation_target,
                                                            source_vocabs, target_vocab,
                                                            max_seq_len_source, max_seq_len_target)

    check_condition(validation_length_statistics.num_sents > 0,
                    "No validation sequences found with length smaller or equal than the maximum sequence length."
                    "Consider increasing %s" % C.TRAINING_ARG_MAX_SEQ_LEN)

    validation_sources_sentences, validation_target_sentences = create_sequence_readers(validation_sources,
                                                                                        validation_target,
                                                                                        source_vocabs, target_vocab)
    discarded_sentences = get_discarded_sentences_doc(validation_sources_sentences[0],
                                                      validation_target_sentences,
                                                      buckets[-1])

    (source_sentences, target_sentences,
     source_pre_sentences, source_nxt_sentences, target_pre_sentences, target_nxt_sentences) = \
        create_sequence_readers_doc(
                source_original=validation_sources[0],
                target_original=validation_target,
                source_doc=doc_context_config.source_validation,
                target_doc=doc_context_config.target_validation,
                window_config=doc_context_config.window_config,
                vocab_sources=source_vocabs,
                vocab_target=target_vocab,
                discarded_sentences=discarded_sentences)

    validation_data_statistics = get_data_statistics_doc(doc_context_config,
                                                         source_pre_sentences, source_nxt_sentences,
                                                         target_pre_sentences, target_nxt_sentences,
                                                         source_sentences,
                                                         target_sentences,
                                                         buckets,
                                                         validation_length_statistics.length_ratio_mean,
                                                         validation_length_statistics.length_ratio_std,
                                                         source_vocabs, target_vocab)

    validation_data_statistics.log(bucket_batch_sizes)

    validation_data = data_loader.load(source_sentences, target_sentences,
                                       source_pre_sentences, source_nxt_sentences,
                                       target_pre_sentences, target_nxt_sentences,
                                       validation_data_statistics.num_sents_per_bucket).fill_up(bucket_batch_sizes)

    return ParallelSampleIterDoc(window_config=doc_context_config.window_config,
                                 data=validation_data,
                                 buckets=buckets,
                                 batch_size=batch_size,
                                 bucket_batch_sizes=bucket_batch_sizes,
                                 num_factors=len(validation_sources))


def get_prepared_data_iters(prepared_data_dir: str,
                            validation_sources: List[str],
                            validation_target: str,
                            shared_vocab: bool,
                            batch_size: int,
                            batch_by_words: bool,
                            batch_num_devices: int,
                            permute: bool = True) -> Tuple['BaseParallelSampleIter',
                                                           'BaseParallelSampleIter',
                                                           'DataConfig', List[vocab.Vocab], vocab.Vocab]:
    logger.info("===============================")
    logger.info("Creating training data iterator")
    logger.info("===============================")

    version_file = os.path.join(prepared_data_dir, C.PREPARED_DATA_VERSION_FILE)
    with open(version_file) as version_in:
        version = int(version_in.read())
        check_condition(version == C.PREPARED_DATA_VERSION,
                        "The dataset %s was written in an old and incompatible format. Please rerun data "
                        "preparation with a current version of Sockeye." % prepared_data_dir)
    info_file = os.path.join(prepared_data_dir, C.DATA_INFO)
    check_condition(os.path.exists(info_file),
                    "Could not find data info %s. Are you sure %s is a directory created with "
                    "python -m sockeye.prepare_data?" % (info_file, prepared_data_dir))
    data_info = cast(DataInfo, DataInfo.load(info_file))
    config_file = os.path.join(prepared_data_dir, C.DATA_CONFIG)
    check_condition(os.path.exists(config_file),
                    "Could not find data config %s. Are you sure %s is a directory created with "
                    "python -m sockeye.prepare_data?" % (config_file, prepared_data_dir))
    config_data = cast(DataConfig, DataConfig.load(config_file))
    shard_fnames = [os.path.join(prepared_data_dir,
                                 C.SHARD_NAME % shard_idx) for shard_idx in range(data_info.num_shards)]
    for shard_fname in shard_fnames:
        check_condition(os.path.exists(shard_fname), "Shard %s does not exist." % shard_fname)

    check_condition(shared_vocab == data_info.shared_vocab, "Shared vocabulary settings need to match these "
                                                            "of the prepared data (e.g. for weight tying). "
                                                            "Specify or omit %s consistently when training "
                                                            "and preparing the data." % C.VOCAB_ARG_SHARED_VOCAB)

    source_vocabs = vocab.load_source_vocabs(prepared_data_dir)
    target_vocab = vocab.load_target_vocab(prepared_data_dir)

    check_condition(len(source_vocabs) == len(data_info.sources),
                    "Wrong number of source vocabularies. Found %d, need %d." % (len(source_vocabs),
                                                                                 len(data_info.sources)))

    buckets = config_data.data_statistics.buckets
    max_seq_len_source = config_data.max_seq_len_source
    max_seq_len_target = config_data.max_seq_len_target

    bucket_batch_sizes = define_bucket_batch_sizes(buckets,
                                                   batch_size,
                                                   batch_by_words,
                                                   batch_num_devices,
                                                   config_data.data_statistics.average_len_target_per_bucket)

    config_data.data_statistics.log(bucket_batch_sizes)

    train_iter = ShardedParallelSampleIter(shard_fnames,
                                           buckets,
                                           batch_size,
                                           bucket_batch_sizes,
                                           num_factors=len(data_info.sources),
                                           permute=permute)

    data_loader = RawParallelDatasetLoader(buckets=buckets,
                                           eos_id=target_vocab[C.EOS_SYMBOL],
                                           pad_id=C.PAD_ID)

    validation_iter = get_validation_data_iter(data_loader=data_loader,
                                               validation_sources=validation_sources,
                                               validation_target=validation_target,
                                               buckets=buckets,
                                               bucket_batch_sizes=bucket_batch_sizes,
                                               source_vocabs=source_vocabs,
                                               target_vocab=target_vocab,
                                               max_seq_len_source=max_seq_len_source,
                                               max_seq_len_target=max_seq_len_target,
                                               batch_size=batch_size)

    return train_iter, validation_iter, config_data, source_vocabs, target_vocab


def get_training_data_iters(sources: List[str],
                            target: str,
                            validation_sources: List[str],
                            validation_target: str,
                            source_vocabs: List[vocab.Vocab],
                            target_vocab: vocab.Vocab,
                            source_vocab_paths: List[Optional[str]],
                            target_vocab_path: Optional[str],
                            shared_vocab: bool,
                            batch_size: int,
                            batch_by_words: bool,
                            batch_num_devices: int,
                            max_seq_len_source: int,
                            max_seq_len_target: int,
                            bucketing: bool,
                            bucket_width: int,
                            allow_empty: bool = False,
                            doc_context_config: Optional[doc_context.DocumentContextConfig] = None) -> \
        Union[
            Tuple['BaseParallelSampleIter',
                  Optional['BaseParallelSampleIter'],
                  'DataConfig', 'DataInfo'],
            Tuple['BaseParallelSampleIterDoc',
                  Optional['BaseParallelSampleIterDoc'],
                  'DataConfigDoc', 'DataInfoDoc']
        ]:
    """
    Returns data iterators for training and validation data.

    :param sources: Path to source training data (with optional factor data paths).
    :param target: Path to target training data.
    :param validation_sources: Path to source validation data (with optional factor data paths).
    :param validation_target: Path to target validation data.
    :param source_vocabs: Source vocabulary and optional factor vocabularies.
    :param target_vocab: Target vocabulary.
    :param source_vocab_paths: Path to source vocabulary.
    :param target_vocab_path: Path to target vocabulary.
    :param shared_vocab: Whether the vocabularies are shared.
    :param batch_size: Batch size.
    :param batch_by_words: Size batches by words rather than sentences.
    :param batch_num_devices: Number of devices batches will be parallelized across.
    :param max_seq_len_source: Maximum source sequence length.
    :param max_seq_len_target: Maximum target sequence length.
    :param bucketing: Whether to use bucketing.
    :param bucket_width: Size of buckets.
    :param allow_empty: Unless True if no sentences are below or equal to the maximum length an exception is raised.
    :param doc_context_config: Document context configuration.
    :return: Tuple of (training data iterator, validation data iterator, data config).
    """
    logger.info("===============================")
    logger.info("Creating training data iterator")
    logger.info("===============================")
    # Pass 1: get target/source length ratios.
    length_statistics = analyze_sequence_lengths(sources, target, source_vocabs, target_vocab,
                                                 max_seq_len_source, max_seq_len_target)

    if not allow_empty:
        check_condition(length_statistics.num_sents > 0,
                        "No training sequences found with length smaller or equal than the maximum sequence length."
                        "Consider increasing %s" % C.TRAINING_ARG_MAX_SEQ_LEN)

    # define buckets
    buckets = define_parallel_buckets(max_seq_len_source, max_seq_len_target, bucket_width,
                                      length_statistics.length_ratio_mean, doc_context_config) if bucketing else \
        [(max_seq_len_source, max_seq_len_target)]

    sources_sentences, target_sentences = create_sequence_readers(sources, target, source_vocabs, target_vocab)
    if doc_context_config is not None:
        assert isinstance(buckets, List[BucketDoc])
        logger.info("Including additional document level context information.")
        discarded_sentences = get_discarded_sentences_doc(sources_sentences[0],
                                                          target_sentences,
                                                          buckets[-1])

        (sources_sentences,
         target_sentences,
         source_pre_sentences, source_nxt_sentences, target_pre_sentences, target_nxt_sentences) = \
            create_sequence_readers_doc(
                    source_original=sources[0],
                    target_original=target,
                    source_doc=doc_context_config.source_train,
                    target_doc=doc_context_config.target_train,
                    window_config=doc_context_config.window_config,
                    vocab_sources=source_vocabs,
                    vocab_target=target_vocab,
                    discarded_sentences=discarded_sentences)

        data_statistics_doc = get_data_statistics_doc(
                doc_context_config,
                source_pre_sentences, source_nxt_sentences,
                target_pre_sentences, target_nxt_sentences,
                sources_sentences, target_sentences, buckets,
                length_statistics.length_ratio_mean, length_statistics.length_ratio_std,
                source_vocabs, target_vocab
        )

        bucket_batch_sizes_doc = define_bucket_batch_sizes_doc(buckets,
                                                               batch_size,
                                                               batch_by_words,
                                                               batch_num_devices,
                                                               data_statistics_doc.average_len_target_per_bucket,
                                                               data_statistics_doc.average_len_src_pre_per_bucket,
                                                               data_statistics_doc.average_len_src_nxt_per_bucket,
                                                               data_statistics_doc.average_len_tar_pre_per_bucket,
                                                               data_statistics_doc.average_len_tar_nxt_per_bucket)

        data_statistics_doc.log(bucket_batch_sizes_doc)

        data_loader_doc = RawParallelDatasetLoaderDoc(window_config=doc_context_config.window_config,
                                                      buckets=buckets,
                                                      eos_id=target_vocab[C.EOS_SYMBOL],
                                                      pad_id=C.PAD_ID)
        training_data_doc = data_loader_doc.load(sources_sentences, target_sentences,
                                                 source_pre_sentences, source_nxt_sentences,
                                                 target_pre_sentences, target_nxt_sentences,
                                                 data_statistics_doc.num_sents_per_bucket).fill_up(bucket_batch_sizes_doc)

        data_info_doc = DataInfoDoc(sources=sources,
                                    target=target,
                                    source_doc=doc_context_config.source_train,
                                    target_doc=doc_context_config.target_train,
                                    source_vocabs=source_vocab_paths,
                                    target_vocab=target_vocab_path,
                                    shared_vocab=shared_vocab,
                                    num_shards=1)
        config_data_doc = DataConfigDoc(data_statistics=data_statistics_doc,
                                        max_seq_len_source=max_seq_len_source,
                                        max_seq_len_target=max_seq_len_target,
                                        num_source_factors=len(sources),
                                        source_with_eos=True)

        train_iter_doc = ParallelSampleIterDoc(window_config=doc_context_config.window_config,
                                               data=training_data_doc,
                                               buckets=buckets,
                                               batch_size=batch_size,
                                               bucket_batch_sizes=bucket_batch_sizes_doc,
                                               num_factors=len(sources),
                                               permute=True)
        validation_iter_doc = get_validation_data_iter_doc(doc_context_config=doc_context_config,
                                                           data_loader=data_loader_doc,
                                                           validation_sources=validation_sources,
                                                           validation_target=validation_target,
                                                           buckets=buckets,
                                                           bucket_batch_sizes=bucket_batch_sizes_doc,
                                                           source_vocabs=source_vocabs,
                                                           target_vocab=target_vocab,
                                                           max_seq_len_source=max_seq_len_source,
                                                           max_seq_len_target=max_seq_len_target,
                                                           batch_size=batch_size)
        return train_iter_doc, validation_iter_doc, config_data_doc, data_info_doc


    # Pass 2: Get data statistics and determine the number of data points for each bucket.
    data_statistics = get_data_statistics(sources_sentences, target_sentences, buckets,
                                          length_statistics.length_ratio_mean, length_statistics.length_ratio_std,
                                          source_vocabs, target_vocab)

    bucket_batch_sizes = define_bucket_batch_sizes(buckets,
                                                   batch_size,
                                                   batch_by_words,
                                                   batch_num_devices,
                                                   data_statistics.average_len_target_per_bucket)

    data_statistics.log(bucket_batch_sizes)

    # Pass 3: Load the data into memory and return the iterator.
    data_loader = RawParallelDatasetLoader(buckets=buckets,
                                           eos_id=target_vocab[C.EOS_SYMBOL],
                                           pad_id=C.PAD_ID)

    training_data = data_loader.load(sources_sentences, target_sentences,
                                     data_statistics.num_sents_per_bucket).fill_up(bucket_batch_sizes)

    data_info = DataInfo(sources=sources,
                         target=target,
                         source_vocabs=source_vocab_paths,
                         target_vocab=target_vocab_path,
                         shared_vocab=shared_vocab,
                         num_shards=1)

    config_data = DataConfig(data_statistics=data_statistics,
                             max_seq_len_source=max_seq_len_source,
                             max_seq_len_target=max_seq_len_target,
                             num_source_factors=len(sources),
                             source_with_eos=True)

    train_iter = ParallelSampleIter(data=training_data,
                                    buckets=buckets,
                                    batch_size=batch_size,
                                    bucket_batch_sizes=bucket_batch_sizes,
                                    num_factors=len(sources),
                                    permute=True)

    validation_iter = get_validation_data_iter(data_loader=data_loader,
                                               validation_sources=validation_sources,
                                               validation_target=validation_target,
                                               buckets=buckets,
                                               bucket_batch_sizes=bucket_batch_sizes,
                                               source_vocabs=source_vocabs,
                                               target_vocab=target_vocab,
                                               max_seq_len_source=max_seq_len_source,
                                               max_seq_len_target=max_seq_len_target,
                                               batch_size=batch_size)

    return train_iter, validation_iter, config_data, data_info


def get_scoring_data_iters(sources: List[str],
                           target: str,
                           source_vocabs: List[vocab.Vocab],
                           target_vocab: vocab.Vocab,
                           batch_size: int,
                           batch_num_devices: int,
                           max_seq_len_source: int,
                           max_seq_len_target: int) -> 'BaseParallelSampleIter':
    """
    Returns a data iterator for scoring. The iterator loads data on demand,
    batch by batch, and does not skip any lines. Lines that are too long
    are truncated.

    :param sources: Path to source training data (with optional factor data paths).
    :param target: Path to target training data.
    :param source_vocabs: Source vocabulary and optional factor vocabularies.
    :param target_vocab: Target vocabulary.
    :param batch_size: Batch size.
    :param batch_num_devices: Number of devices batches will be parallelized across.
    :param max_seq_len_source: Maximum source sequence length.
    :param max_seq_len_target: Maximum target sequence length.
    :return: The scoring data iterator.
    """
    logger.info("==============================")
    logger.info("Creating scoring data iterator")
    logger.info("==============================")

    # One bucket to hold them all,
    bucket = (max_seq_len_source, max_seq_len_target)

    # ...One loader to raise them,
    data_loader = RawParallelDatasetLoader(buckets=[bucket],
                                           eos_id=target_vocab[C.EOS_SYMBOL],
                                           pad_id=C.PAD_ID,
                                           skip_blanks=False)

    # ...one iterator to traverse them all,
    scoring_iter = BatchedRawParallelSampleIter(data_loader=data_loader,
                                                sources=sources,
                                                target=target,
                                                source_vocabs=source_vocabs,
                                                target_vocab=target_vocab,
                                                bucket=bucket,
                                                batch_size=batch_size,
                                                max_lens=(max_seq_len_source, max_seq_len_target),
                                                num_factors=len(sources))

    # and with the model appraise them.
    return scoring_iter


class LengthStatistics(config.Config):

    def __init__(self,
                 num_sents: int,
                 length_ratio_mean: float,
                 length_ratio_std: float) -> None:
        super().__init__()
        self.num_sents = num_sents
        self.length_ratio_mean = length_ratio_mean
        self.length_ratio_std = length_ratio_std


class DataStatistics(config.Config):

    def __init__(self,
                 num_sents: int,
                 num_discarded,
                 num_tokens_source,
                 num_tokens_target,
                 num_unks_source,
                 num_unks_target,
                 max_observed_len_source,
                 max_observed_len_target,
                 size_vocab_source,
                 size_vocab_target,
                 length_ratio_mean,
                 length_ratio_std,
                 buckets: List[Tuple[int, int]],
                 num_sents_per_bucket: List[int],
                 mean_len_target_per_bucket: List[Optional[float]]) -> None:
        super().__init__()
        self.num_sents = num_sents
        self.num_discarded = num_discarded
        self.num_tokens_source = num_tokens_source
        self.num_tokens_target = num_tokens_target
        self.num_unks_source = num_unks_source
        self.num_unks_target = num_unks_target
        self.max_observed_len_source = max_observed_len_source
        self.max_observed_len_target = max_observed_len_target
        self.size_vocab_source = size_vocab_source
        self.size_vocab_target = size_vocab_target
        self.length_ratio_mean = length_ratio_mean
        self.length_ratio_std = length_ratio_std
        self.buckets = buckets
        self.num_sents_per_bucket = num_sents_per_bucket
        self.average_len_target_per_bucket = mean_len_target_per_bucket

    def log(self, bucket_batch_sizes: Optional[List[BucketBatchSize]] = None):
        logger.info("Tokens: source %d target %d", self.num_tokens_source, self.num_tokens_target)
        if self.num_tokens_source > 0 and self.num_tokens_target > 0:
            logger.info("Vocabulary coverage: source %.0f%% target %.0f%%",
                        (1 - self.num_unks_source / self.num_tokens_source) * 100,
                        (1 - self.num_unks_target / self.num_tokens_target) * 100)
        logger.info("%d sequences across %d buckets", self.num_sents, len(self.num_sents_per_bucket))
        logger.info("%d sequences did not fit into buckets and were discarded", self.num_discarded)
        if bucket_batch_sizes is not None:
            describe_data_and_buckets(self, bucket_batch_sizes)


class DataStatisticsDoc(config.Config):

    def __init__(self,
                 num_sents: int,
                 num_discarded,
                 num_tokens_source,
                 num_tokens_target,
                 num_unks_source,
                 num_unks_target,
                 max_observed_len_source,
                 max_observed_len_target,
                 size_vocab_source,
                 size_vocab_target,
                 length_ratio_mean,
                 length_ratio_std,
                 buckets: List[BucketDoc],
                 num_sents_per_bucket: List[int],
                 mean_len_target_per_bucket: List[Optional[float]],
                 mean_len_src_pre_per_bucket: List[List[Optional[float]]],
                 mean_len_src_nxt_per_bucket: List[List[Optional[float]]],
                 mean_len_tar_pre_per_bucket: List[List[Optional[float]]],
                 mean_len_tar_nxt_per_bucket: List[List[Optional[float]]]) -> None:
        super().__init__()
        self.num_sents = num_sents
        self.num_discarded = num_discarded
        self.num_tokens_source = num_tokens_source
        self.num_tokens_target = num_tokens_target
        self.num_unks_source = num_unks_source
        self.num_unks_target = num_unks_target
        self.max_observed_len_source = max_observed_len_source
        self.max_observed_len_target = max_observed_len_target
        self.size_vocab_source = size_vocab_source
        self.size_vocab_target = size_vocab_target
        self.length_ratio_mean = length_ratio_mean
        self.length_ratio_std = length_ratio_std
        self.buckets = buckets
        self.num_sents_per_bucket = num_sents_per_bucket
        self.average_len_target_per_bucket = mean_len_target_per_bucket
        self.average_len_src_pre_per_bucket = mean_len_src_pre_per_bucket
        self.average_len_src_nxt_per_bucket = mean_len_src_nxt_per_bucket
        self.average_len_tar_pre_per_bucket = mean_len_tar_pre_per_bucket
        self.average_len_tar_nxt_per_bucket = mean_len_tar_nxt_per_bucket

    def log(self, bucket_batch_sizes: Optional[List[BucketBatchSizeDoc]] = None):
        logger.info("Tokens: source %d target %d", self.num_tokens_source, self.num_tokens_target)
        if self.num_tokens_source > 0 and self.num_tokens_target > 0:
            logger.info("Vocabulary coverage: source %.0f%% target %.0f%%",
                        (1 - self.num_unks_source / self.num_tokens_source) * 100,
                        (1 - self.num_unks_target / self.num_tokens_target) * 100)
        logger.info("%d sequences across %d buckets", self.num_sents, len(self.num_sents_per_bucket))
        logger.info("%d sequences did not fit into buckets and were discarded", self.num_discarded)
        if bucket_batch_sizes is not None:
            describe_data_and_buckets_doc(self, bucket_batch_sizes)


class DataStatisticsAccumulatorDoc:
    """
    Statistics accumulator which includes additional context data on document level.
    """
    def __init__(self,
                 buckets: List[BucketDoc],
                 window_config: doc_context.WindowConfig,
                 vocab_source: Optional[Dict[str, int]],
                 vocab_target: Dict[str, int],
                 length_ratio_mean: float,
                 length_ratio_std: float) -> None:
        self.buckets = buckets
        num_buckets = len(buckets)
        self.window_config = window_config
        self.length_ratio_mean = length_ratio_mean
        self.length_ratio_std = length_ratio_std
        if vocab_source is not None:
            self.unk_id_source = vocab_source[C.UNK_SYMBOL]
            self.size_vocab_source = len(vocab_source)
        else:
            self.unk_id_source = None
            self.size_vocab_source = 0
        self.unk_id_target = vocab_target[C.UNK_SYMBOL]
        self.size_vocab_target = len(vocab_target)
        self.num_sents = 0
        self.num_discarded = 0
        self.num_tokens_source = 0
        self.num_tokens_target = 0
        self.num_tokens_doc = [[0] * self.window_config.src_pre,
                               [0] * self.window_config.src_nxt,
                               [0] * self.window_config.tar_pre,
                               [0] * self.window_config.tar_nxt]
        self.num_unks_source = 0
        self.num_unks_target = 0
        self.max_observed_len_source = 0
        self.max_observed_len_target = 0
        self.max_observed_doc = [[0] * self.window_config.src_pre,
                                 [0] * self.window_config.src_nxt,
                                 [0] * self.window_config.tar_pre,
                                 [0] * self.window_config.tar_nxt]

        self._mean_len_target_per_bucket = [OnlineMeanAndVariance() for _ in range(num_buckets)]
        self._mean_len_src_pre_per_bucket = [[OnlineMeanAndVariance() for _ in range(self.window_config.src_pre)]
                                             for _ in range(num_buckets)]
        self._mean_len_src_nxt_per_bucket = [[OnlineMeanAndVariance() for _ in range(self.window_config.src_nxt)]
                                             for _ in range(num_buckets)]
        self._mean_len_tar_pre_per_bucket = [[OnlineMeanAndVariance() for _ in range(self.window_config.tar_pre)]
                                             for _ in range(num_buckets)]
        self._mean_len_tar_nxt_per_bucket = [[OnlineMeanAndVariance() for _ in range(self.window_config.tar_nxt)]
                                             for _ in range(num_buckets)]

        self._mean_len_doc = [self._mean_len_src_pre_per_bucket,
                              self._mean_len_src_nxt_per_bucket,
                              self._mean_len_tar_pre_per_bucket,
                              self._mean_len_tar_nxt_per_bucket]

    def sequence_pair(self,
                      source: List[int],
                      target: List[int],
                      source_pre_lens: Optional[List[int]],
                      source_nxt_lens: Optional[List[int]],
                      target_pre_lens: Optional[List[int]],
                      target_nxt_lens: Optional[List[int]],
                      bucket_idx: Optional[int]):
        if bucket_idx is None:
            self.num_discarded += 1
            return

        source_len = len(source)
        target_len = len(target)

        lengths_additional = [source_pre_lens, source_nxt_lens, target_pre_lens, target_nxt_lens]

        self._mean_len_target_per_bucket[bucket_idx].update(target_len)

        for lengths, mean_len_counter in zip(lengths_additional, self._mean_len_doc):
            if lengths is not None:
                for index, length in enumerate(lengths):
                    mean_len_counter[bucket_idx][index].update(length)

        self.num_sents += 1
        self.num_tokens_source += source_len
        self.num_tokens_target += target_len
        self.max_observed_len_source = max(source_len, self.max_observed_len_source)
        self.max_observed_len_target = max(target_len, self.max_observed_len_target)

        for lengths, num_tokens_counter in zip(lengths_additional, self.num_tokens_doc):
            if lengths is not None:
                for index, length in enumerate(lengths):
                    num_tokens_counter[index] += length

        for lengths, max_observed_counter in zip(lengths_additional, self.max_observed_doc):
            if lengths is not None:
                for index, length in enumerate(lengths):
                    max_observed_counter[index] = max(length, max_observed_counter[index])

        if self.unk_id_source is not None:
            self.num_unks_source += source.count(self.unk_id_source)
        self.num_unks_target += target.count(self.unk_id_target)

    @property
    def mean_len_target_per_bucket(self) -> List[Optional[float]]:
        return [mean_and_variance.mean if mean_and_variance.count > 0 else None
                for mean_and_variance in self._mean_len_target_per_bucket]

    @property
    def mean_len_src_pre_per_bucket(self) -> List[List[Optional[float]]]:
        return [[mean_and_variance.mean if mean_and_variance.count > 0 else None
                 for mean_and_variance in mean_and_variances]
                for mean_and_variances in self._mean_len_src_pre_per_bucket]

    @property
    def mean_len_src_nxt_per_bucket(self) -> List[List[Optional[float]]]:
        return [[mean_and_variance.mean if mean_and_variance.count > 0 else None
                 for mean_and_variance in mean_and_variances]
                for mean_and_variances in self._mean_len_src_nxt_per_bucket]

    @property
    def mean_len_tar_pre_per_bucket(self) -> List[List[Optional[float]]]:
        return [[mean_and_variance.mean if mean_and_variance.count > 0 else None
                 for mean_and_variance in mean_and_variances]
                for mean_and_variances in self._mean_len_tar_pre_per_bucket]

    @property
    def mean_len_tar_nxt_per_bucket(self) -> List[List[Optional[float]]]:
        return [[mean_and_variance.mean if mean_and_variance.count > 0 else None
                 for mean_and_variance in mean_and_variances]
                for mean_and_variances in self._mean_len_tar_nxt_per_bucket]

    @property
    def statistics(self):
        num_sents_per_bucket = [mean_and_variance.count for mean_and_variance in self._mean_len_target_per_bucket]
        return DataStatisticsDoc(num_sents=self.num_sents,
                                 num_discarded=self.num_discarded,
                                 num_tokens_source=self.num_tokens_source,
                                 num_tokens_target=self.num_tokens_target,
                                 num_unks_source=self.num_unks_source,
                                 num_unks_target=self.num_unks_target,
                                 max_observed_len_source=self.max_observed_len_source,
                                 max_observed_len_target=self.max_observed_len_target,
                                 size_vocab_source=self.size_vocab_source,
                                 size_vocab_target=self.size_vocab_target,
                                 length_ratio_mean=self.length_ratio_mean,
                                 length_ratio_std=self.length_ratio_std,
                                 buckets=self.buckets,
                                 num_sents_per_bucket=num_sents_per_bucket,
                                 mean_len_target_per_bucket=self.mean_len_target_per_bucket,
                                 mean_len_src_pre_per_bucket=self.mean_len_src_pre_per_bucket,
                                 mean_len_src_nxt_per_bucket=self.mean_len_src_nxt_per_bucket,
                                 mean_len_tar_pre_per_bucket=self.mean_len_tar_pre_per_bucket,
                                 mean_len_tar_nxt_per_bucket=self.mean_len_tar_nxt_per_bucket)


def describe_data_and_buckets(data_statistics: DataStatistics, bucket_batch_sizes: List[BucketBatchSize]):
    """
    Describes statistics across buckets
    """
    check_condition(len(bucket_batch_sizes) == len(data_statistics.buckets),
                    "Number of bucket batch sizes (%d) does not match number of buckets in statistics (%d)."
                    % (len(bucket_batch_sizes), len(data_statistics.buckets)))
    for bucket_batch_size, num_seq in zip(bucket_batch_sizes, data_statistics.num_sents_per_bucket):
        if num_seq > 0:
            logger.info("Bucket %s: %d samples in %d batches of %d, ~%.1f tokens/batch.",
                        bucket_batch_size.bucket,
                        num_seq,
                        math.ceil(num_seq / bucket_batch_size.batch_size),
                        bucket_batch_size.batch_size,
                        bucket_batch_size.average_words_per_batch)


def describe_data_and_buckets_doc(data_statistics: DataStatisticsDoc,
                                  bucket_batch_sizes: List[BucketBatchSizeDoc]):
    """
    Describes statistics across buckets including additional context information on document level.
    """
    check_condition(len(bucket_batch_sizes) == len(data_statistics.buckets),
                    "Number of bucket batch sizes (%d) does not match number of buckets in statistics (%d)."
                    % (len(bucket_batch_sizes), len(data_statistics.buckets)))
    for bucket_batch_size, num_seq in zip(bucket_batch_sizes, data_statistics.num_sents_per_bucket):
        if num_seq > 0:
            logger.info("Bucket %s: %d samples in %d batches of %d, ~%.1f tokens/batch.",
                        bucket_batch_size.bucket,
                        num_seq,
                        math.ceil(num_seq / bucket_batch_size.batch_size),
                        bucket_batch_size.batch_size,
                        bucket_batch_size.average_words_per_batch)


class DataInfo(config.Config):
    """
    Stores training data information that is not relevant for inference.
    """

    def __init__(self,
                 sources: List[str],
                 target: str,
                 source_vocabs: List[Optional[str]],
                 target_vocab: Optional[str],
                 shared_vocab: bool,
                 num_shards: int) -> None:
        super().__init__()
        self.sources = sources
        self.target = target
        self.source_vocabs = source_vocabs
        self.target_vocab = target_vocab
        self.shared_vocab = shared_vocab
        self.num_shards = num_shards


class DataInfoDoc(DataInfo):
    """
    Stores training data information including document-level ones that are not relevant for inference.
    """

    def __init__(self,
                 sources: List[str],
                 target: str,
                 source_doc: Optional[str],
                 target_doc: Optional[str],
                 source_vocabs: List[Optional[str]],
                 target_vocab: Optional[str],
                 shared_vocab: bool,
                 num_shards: int) -> None:
        super().__init__(sources=sources, target=target, source_vocabs=source_vocabs, target_vocab=target_vocab,
                         shared_vocab=shared_vocab, num_shards=num_shards)
        self.source_doc = source_doc
        self.target_doc = target_doc


class DataConfig(config.Config):
    """
    Stores data statistics relevant for inference.
    """

    def __init__(self,
                 data_statistics: DataStatistics,
                 max_seq_len_source: int,
                 max_seq_len_target: int,
                 num_source_factors: int,
                 source_with_eos: bool = False) -> None:
        super().__init__()
        self.data_statistics = data_statistics
        self.max_seq_len_source = max_seq_len_source
        self.max_seq_len_target = max_seq_len_target
        self.num_source_factors = num_source_factors
        self.source_with_eos = source_with_eos


class DataConfigDoc(config.Config):
    """
    Stores data statistics relevant for inference including few document-level information.
    """

    def __init__(self,
                 data_statistics: DataStatisticsDoc,
                 max_seq_len_source: int,
                 max_seq_len_target: int,
                 num_source_factors: int,
                 source_with_eos: bool = False) -> None:
        super().__init__()
        self.data_statistics = data_statistics
        self.max_seq_len_source = max_seq_len_source
        self.max_seq_len_target = max_seq_len_target
        self.num_source_factors = num_source_factors
        self.source_with_eos = source_with_eos


def read_content(path: str, limit: Optional[int] = None) -> Iterator[List[str]]:
    """
    Returns a list of tokens for each line in path up to a limit.

    :param path: Path to files containing sentences.
    :param limit: How many lines to read from path.
    :return: Iterator over lists of words.
    """
    with smart_open(path) as indata:
        for i, line in enumerate(indata):
            if limit is not None and i == limit:
                break
            yield list(get_tokens(line))


def nxt_shift_iterate(path: str,
                      nxt_shift: int,
                      discarded_sentences: Optional[List[bool]],
                      unk: str):
    with smart_open(path) as indata:
        for _ in range(nxt_shift):
            next(indata)

        for i, line in enumerate(indata, nxt_shift):
            if discarded_sentences is not None and discarded_sentences[i]:
                continue
            yield line

    unk_sequence = unk
    for _ in range(nxt_shift):
        yield unk_sequence


def pre_shift_iterate(path: str,
                      pre_shift: int,
                      discarded_sentences: Optional[List[bool]],
                      unk: str,
                      last_index: int):
    unk_sequence = unk
    for _ in range(pre_shift):
        yield unk_sequence

    with smart_open(path) as indata:
        for i, line in enumerate(indata):
            if last_index is not None and i == last_index:
                break
            if discarded_sentences is not None and discarded_sentences[i]:
                continue
            yield line


def read_content_doc(path: str,
                     unk: str,
                     discarded_sentences: Optional[List[bool]] = None,
                     pre_shift: Optional[int] = None,
                     nxt_shift: Optional[int] = None) -> Iterator[List[str]]:
    """
    Returns a list of tokens for each line in path up to a limit with regards to shiftings for
    document-level context.

    :param path: Path to file containing context sentences.
    :param unk: Unknown token.
    :param discarded_sentences: Denotes which line has been discarded due to sequence length being too long for
                                either current source or current target sentence.
    :param pre_shift: Shift by n previous sentences.
    :param nxt_shift: Shift by n next sentences.
    :return: Iterator over lists of words.
    """
    file_rows = sum(1 for _ in smart_open(path))

    already_computed = False
    if pre_shift is not None:
        already_computed = True
        for index, line in enumerate(pre_shift_iterate(path,
                                                       pre_shift,
                                                       discarded_sentences,
                                                       unk,
                                                       file_rows - pre_shift)):
            yield list(get_tokens(line))

    if nxt_shift is not None:
        if already_computed:
            raise ValueError("Applied pre_shift AND nxt_shift.")
        already_computed = True

        for index, line in enumerate(nxt_shift_iterate(path,
                                                       nxt_shift,
                                                       discarded_sentences,
                                                       unk)):
            yield list(get_tokens(line))

    if not already_computed:
        raise ValueError("Did not specify either pre_shift or nxt_shift.")


def tokens2ids(tokens: Iterable[str], vocab: Dict[str, int]) -> List[int]:
    """
    Returns sequence of integer ids given a sequence of tokens and vocab.

    :param tokens: List of string tokens.
    :param vocab: Vocabulary (containing UNK symbol).
    :return: List of word ids.
    """
    return [vocab.get(w, vocab[C.UNK_SYMBOL]) for w in tokens]


def strids2ids(tokens: Iterable[str]) -> List[int]:
    """
    Returns sequence of integer ids given a sequence of string ids.

    :param tokens: List of integer tokens.
    :return: List of word ids.
    """
    return list(map(int, tokens))


def ids2strids(ids: Iterable[int]) -> str:
    """
    Returns a string representation of a sequence of integers.

    :param ids: Sequence of integers.
    :return: String sequence
    """
    return C.TOKEN_SEPARATOR.join(map(str, ids))


def ids2tokens(token_ids: Iterable[int],
               vocab_inv: Dict[int, str],
               exclude_set: Set[int]) -> Iterator[str]:
    """
    Transforms a list of token IDs into a list of words, excluding any IDs in `exclude_set`.

    :param token_ids: The list of token IDs.
    :param vocab_inv: The inverse vocabulary.
    :param exclude_set: The list of token IDs to exclude.
    :return: The list of words.
    """
    tokens = (vocab_inv[token] for token in token_ids)
    return (tok for token_id, tok in zip(token_ids, tokens) if token_id not in exclude_set)


class SequenceReader(Iterable):
    """
    Reads sequence samples from path and (optionally) creates integer id sequences.
    Streams from disk, instead of loading all samples into memory.
    If vocab is None, the sequences in path are assumed to be integers coded as strings.
    Empty sequences are yielded as None.

    :param path: Path to read data from.
    :param vocabulary: Optional mapping from strings to integer ids.
    :param add_bos: Whether to add Beginning-Of-Sentence (BOS) symbol.
    :param limit: Read limit.
    """

    def __init__(self,
                 path: str,
                 vocabulary: Optional[vocab.Vocab] = None,
                 add_bos: bool = False,
                 add_eos: bool = False,
                 limit: Optional[int] = None) -> None:
        self.path = path
        self.vocab = vocabulary
        self.bos_id = None
        self.eos_id = None
        if vocabulary is not None:
            assert C.UNK_SYMBOL in vocabulary
            assert vocabulary[C.PAD_SYMBOL] == C.PAD_ID
            assert C.BOS_SYMBOL in vocabulary
            assert C.EOS_SYMBOL in vocabulary
            self.bos_id = vocabulary[C.BOS_SYMBOL]
            self.eos_id = vocabulary[C.EOS_SYMBOL]
        else:
            check_condition(not add_bos and not add_eos, "Adding a BOS or EOS symbol requires a vocabulary")
        self.add_bos = add_bos
        self.add_eos = add_eos
        self.limit = limit

    def __iter__(self):
        for tokens in read_content(self.path, self.limit):
            if self.vocab is not None:
                sequence = tokens2ids(tokens, self.vocab)
            else:
                sequence = strids2ids(tokens)
            if len(sequence) == 0:
                yield None
                continue
            if self.add_bos:
                sequence.insert(0, self.bos_id)
            if self.add_eos:
                sequence.append(self.eos_id)
            yield sequence


class SequenceReaderDoc(SequenceReader):
    """
    Reads sequence samples from path and (optionally) creates integer id sequences.
    Streams from disk, instead of loading all samples into memory. Moreover, we allow shifts, that is,
    we allow having sequences of <UNK> samples in order to imitate shifts for document-level readers.

    E.g. if we allow a pre_shift=1 then the first sample be an <UNK>-sequence. Furthermore, we stop one line before the
    end. Note that we have the same amount of yielded sequences as with no shifts.
    pre_shift=1 allows us to retrieve previous 1 sentences from path and so on.

    If vocab is None, the sequences in path are assumed to be integers coded as strings.
    Empty sequences are yielded as None.

    :param path: Path to read data from.
    :param vocabulary: Optional mapping from strings to integer ids.
    :param add_bos: Whether to add Beginning-Of-Sentence (BOS) symbol.
    :param limit: Read limit.
    :param pre_shift: Use n previous
    """

    def __init__(self,
                 path: str,
                 discarded_sentences: Optional[List[bool]] = None,
                 vocabulary: Optional[vocab.Vocab] = None,
                 add_bos: bool = False,
                 add_eos: bool = False,
                 limit: Optional[int] = None,
                 pre_shift: Optional[int] = None,
                 nxt_shift: Optional[int] = None) -> None:
        super().__init__(path=path, vocabulary=vocabulary, add_bos=add_bos, add_eos=add_eos, limit=limit)
        self.discarded_sentences = discarded_sentences
        self.pre_shift = pre_shift
        self.nxt_shift = nxt_shift

    def __iter__(self) -> List[int]:
        for tokens in read_content_doc(self.path,
                                       discarded_sentences=self.discarded_sentences,
                                       pre_shift=self.pre_shift,
                                       nxt_shift=self.nxt_shift,
                                       unk=C.UNK_SYMBOL):
            if self.vocab is not None:
                sequence = tokens2ids(tokens, self.vocab)
            else:
                sequence = strids2ids(tokens)
            if len(sequence) == 0:
                yield None
                continue
            if self.add_bos:
                sequence.insert(0, self.bos_id)
            if self.add_eos:
                sequence.append(self.eos_id)
            yield sequence


def create_sequence_readers(sources: List[str], target: str,
                            vocab_sources: List[vocab.Vocab],
                            vocab_target: vocab.Vocab) -> Tuple[List[SequenceReader], SequenceReader]:
    """
    Create source readers with EOS and target readers with BOS.

    :param sources: The file names of source data and factors.
    :param target: The file name of the target data.
    :param vocab_sources: The source vocabularies.
    :param vocab_target: The target vocabularies.
    :return: The source sequence readers and the target reader.
    """
    source_sequence_readers = [SequenceReader(source, vocab, add_eos=True) for source, vocab in
                               zip(sources, vocab_sources)]
    target_sequence_reader = SequenceReader(target, vocab_target, add_bos=True)
    return source_sequence_readers, target_sequence_reader


def get_discarded_sentences_doc(source_sentences: 'SequenceReader',
                                target_sentences: 'SequenceReader',
                                max_bucket: BucketDoc) -> List[bool]:
    """
    Obtain a list of discarded sentences from training data.

    :param source_sentences: Source sentence reader.
    :param target_sentences: Target sentence reader.
    :param max_bucket: Maximum bucket for longest sequences.
    :return:
    """
    max_bucket_size_src = max_bucket.src_len
    max_bucket_size_tar = max_bucket.tar_len
    discarded_sentences = []  # type: List[bool]
    num_discarded = 0
    for src, tar in zip(source_sentences, target_sentences):
        if len(src) > max_bucket_size_src or len(tar) > max_bucket_size_tar:
            discarded_sentences.append(True)
            num_discarded += 1
        else:
            discarded_sentences.append(False)

    logger.info("Discarded {} sentences from parallel data due "
                "to longer lengths than largest bucket size.".format(num_discarded))
    return discarded_sentences


def create_sequence_readers_doc(source_original: str,
                                target_original: str,
                                source_doc: Optional[str],
                                target_doc: Optional[str],
                                window_config: doc_context.WindowConfig,
                                vocab_sources: List[vocab.Vocab],
                                vocab_target: vocab.Vocab,
                                discarded_sentences: Optional[List[bool]] = None) \
        -> Tuple[SequenceReaderDoc,
                 SequenceReaderDoc,
                 List[SequenceReaderDoc],
                 List[SequenceReaderDoc],
                 List[SequenceReaderDoc],
                 List[SequenceReaderDoc]]:
    """
    Construct source and target readers for each additional context data, that is, each context sentence has its own
    sequence reader including shifts.

    :param source_original: Parallel source data file.
    :param target_original: Parallel target data file.
    :param source_doc: Document context source data file.
    :param target_doc: Document context target data file.
    :param window_config: Context window configuration.
    :param vocab_sources: Source vocabularies.
    :param vocab_target: Target vocabulary.
    :param discarded_sentences: Discarded sentences with regards to parallel source and target data.
    :return: The source sequence readers and the target reader for document context data.
    """
    source_reader = SequenceReaderDoc(source_original,
                                      discarded_sentences,
                                      vocab_sources[0],
                                      add_eos=True,
                                      nxt_shift=0)

    target_reader = SequenceReaderDoc(target_original,
                                      discarded_sentences,
                                      vocab_target,
                                      add_bos=True,
                                      nxt_shift=0)

    source_pre_seq_readers = [SequenceReaderDoc(
            source_doc,
            discarded_sentences,
            vocab_sources[0],
            add_eos=True,
            pre_shift=i + 1) for i in range(window_config.src_pre) if source_doc is not None]

    source_nxt_seq_readers = [SequenceReaderDoc(
            source_doc,
            discarded_sentences,
            vocab_sources[0],
            add_eos=True,
            nxt_shift=j + 1) for j in range(window_config.src_nxt) if source_doc is not None]

    target_pre_seq_readers = [SequenceReaderDoc(
            target_doc,
            discarded_sentences,
            vocab_target,
            add_bos=True,
            pre_shift=k + 1) for k in range(window_config.tar_pre) if target_doc is not None]

    target_nxt_seq_readers = [SequenceReaderDoc(
            target_doc,
            discarded_sentences,
            vocab_target,
            add_bos=True,
            nxt_shift=l + 1) for l in range(window_config.tar_nxt) if target_doc is not None]

    return (source_reader, target_reader,
            source_pre_seq_readers, source_nxt_seq_readers, target_pre_seq_readers, target_nxt_seq_readers)


def parallel_iter(source_iterables: Sequence[Iterable[Optional[Any]]],
                  target_iterable: Iterable[Optional[Any]],
                  skip_blanks: bool = True):
    """
    Creates iterators over parallel iteratables by calling iter() on the iterables
    and chaining to parallel_iterate(). The purpose of the separation is to allow
    the caller to save iterator state between calls, if desired.

    :param source_iterables: A list of source iterables.
    :param target_iterable: A target iterable.
    :param skip_blanks: Whether to skip empty target lines.
    :return: Iterators over sources and target.
    """
    source_iterators = [iter(s) for s in source_iterables]
    target_iterator = iter(target_iterable)
    return parallel_iterate(source_iterators, target_iterator, skip_blanks)


def parallel_iter_doc_level(source_iterable: Iterable[Optional[Any]],
                            target_iterable: Iterable[Optional[Any]],
                            source_pre_iterables: List[Iterable],
                            source_nxt_iterables: List[Iterable],
                            target_pre_iterables: List[Iterable],
                            target_nxt_iterables: List[Iterable],
                            skip_blanks: bool = True):
    """
    Creates iterators over parallel iteratables by calling iter() on the iterables
    and chaining to parallel_iterate_doc_level(). The purpose of the separation is to allow
    the caller to save iterator state between calls, if desired.

    :param source_iterable: A source iterable.
    :param target_iterable: A target iterable.
    :param source_pre_iterables: Iterables over previous source sentences.
    :param source_nxt_iterables: Iterables over next source sentences.
    :param target_pre_iterables: Iterables over previous target sentences.
    :param target_nxt_iterables: Iterables over next target sentences.
    :param skip_blanks: Whether to skip empty target lines.
    :return: Iterators over sources and target.
    """
    source_iterators = iter(source_iterable)
    target_iterator = iter(target_iterable)
    source_pre_iterators = [iter(source_pre_iterable) for source_pre_iterable in source_pre_iterables]
    source_nxt_iterators = [iter(source_nxt_iterable) for source_nxt_iterable in source_nxt_iterables]
    target_pre_iterators = [iter(target_pre_iterable) for target_pre_iterable in target_pre_iterables]
    target_nxt_iterators = [iter(target_nxt_iterable) for target_nxt_iterable in target_nxt_iterables]
    return parallel_iterate_doc_level(source_iterators, target_iterator,
                                      source_pre_iterators, source_nxt_iterators,
                                      target_pre_iterators, target_nxt_iterators,
                                      skip_blanks)


def parallel_iterate(source_iterators: Sequence[Iterator[Optional[Any]]],
                     target_iterator: Iterator[Optional[Any]],
                     skip_blanks: bool = True):
    """
    Yields parallel source(s), target sequences from iterables.
    Checks for token parallelism in source sequences.
    Skips pairs where element in at least one iterable is None.
    Checks that all iterables have the same number of elements.
    Can optionally continue from an already-begun iterator.

    :param source_iterators: A list of source iterators.
    :param target_iterator: A target iterator.
    :param skip_blanks: Whether to skip empty target lines.
    :return: Iterators over sources and target.
    """
    num_skipped = 0
    while True:
        try:
            sources = [next(source_iter) for source_iter in source_iterators]
            target = next(target_iterator)
        except StopIteration:
            break
        if skip_blanks and (any((s is None for s in sources)) or target is None):
            num_skipped += 1
            continue
        check_condition(are_none(sources) or are_token_parallel(sources), "Source sequences are not token-parallel: %s" % (str(sources)))
        yield sources, target

    if num_skipped > 0:
        logger.warning("Parallel reading of sequences skipped %d elements", num_skipped)

    check_condition(
        all(next(cast(Iterator, s), None) is None for s in source_iterators) and next(cast(Iterator, target_iterator),
                                                                                      None) is None,
        "Different number of lines in source(s) and target iterables.")


def parallel_iterate_doc_level(source_iterator: Iterator[Optional[Any]],
                               target_iterator: Iterator[Optional[Any]],
                               source_pre_iterators: Sequence[Iterator],
                               source_nxt_iterators: Sequence[Iterator],
                               target_pre_iterators: Sequence[Iterator],
                               target_nxt_iterators: Sequence[Iterator],
                               skip_blanks: bool = True):
    """
    Yields parallel source, target and additional surrounding sequences from iterables.
    Skips pairs where element in at least one iterable of current source/target sentence is None.
    Checks that all iterables have the same number of elements.
    Can optionally continue from an already-begun iterator.

    :param source_iterator: A list of source iterators.
    :param target_iterator: A target iterator.
    :param source_pre_iterators: A list of previous source sentence iterators.
    :param source_nxt_iterators: A list of next source sentence iterators.
    :param target_pre_iterators: A list of previous target sentence iterators.
    :param target_nxt_iterators: A list of next target sentence iterators.
    :param skip_blanks: Whether to skip empty target lines.
    :return: Iterators over sources and target.
    """
    num_skipped = 0
    while True:
        try:
            sources = next(source_iterator)
            target = next(target_iterator)
            source_pre = [next(source_pre_iter) for source_pre_iter in source_pre_iterators]
            source_nxt = [next(source_nxt_iter) for source_nxt_iter in source_nxt_iterators]
            target_pre = [next(target_pre_iter) for target_pre_iter in target_pre_iterators]
            target_nxt = [next(target_nxt_iter) for target_nxt_iter in target_nxt_iterators]
        except StopIteration:
            break
        if skip_blanks and (any((s is None for s in sources)) or target is None):
            num_skipped += 1
            continue
        yield source_pre, sources, source_nxt, target_pre, target, target_nxt

    if num_skipped > 0:
        logger.warning("Parallel reading of sequences skipped %d elements", num_skipped)

    check_condition(
            all(next(cast(Iterator, s), None) is None for s in source_iterator) and next(
                cast(Iterator, target_iterator),
                None) is None,
            "Different number of lines in source(s) and target iterables.")

    check_condition(
            all(next(cast(Iterator, src_pre), None) is None for src_pre in source_pre_iterators)
            and all(next(cast(Iterator, src_nxt), None) is None for src_nxt in source_nxt_iterators)
            and all(next(cast(Iterator, tar_pre), None) is None for tar_pre in target_pre_iterators)
            and all(next(cast(Iterator, tar_nxt), None) is None for tar_nxt in target_nxt_iterators),
            "Different number of lines in the additional data")


class FileListReader(Iterator):
    """
    Reads sequence samples from path provided in a file.

    :param fname: File name containing a list of relative paths.
    :param path: Path to read data from, which is prefixed to the relative paths of fname.
    """

    def __init__(self,
                 fname: str,
                 path: str) -> None:
        self.fname = fname
        self.path = path
        self.fd = smart_open(fname)
        self.count = 0

    def __next__(self):
        fname = self.fd.readline().strip("\n")

        if fname is None:
            self.fd.close()
            raise StopIteration

        self.count += 1
        return os.path.join(self.path, fname)


def get_default_bucket_key(buckets: Union[List[Tuple[int, int]], List[BucketDoc]]) -> Union[Tuple[int, int],
                                                                                            BucketDoc]:
    """
    Returns the default bucket from a list of buckets, i.e. the largest bucket.

    :param buckets: List of buckets.
    :return: The largest bucket in the list.
    """
    return max(buckets)


def get_parallel_bucket(buckets: List[Tuple[int, int]],
                        length_source: int,
                        length_target: int) -> Tuple[Optional[int], Optional[Tuple[int, int]]]:
    """
    Returns bucket index and bucket from a list of buckets, given source and target length.
    Algorithm assumes buckets are sorted from shortest to longest.
    Returns (None, None) if no bucket fits.

    :param buckets: List of buckets, in sorted order, shortest to longest.
    :param length_source: Length of source sequence.
    :param length_target: Length of target sequence.
    :return: Tuple of (bucket index, bucket), or (None, None) if not fitting.
    """
    for j, (source_bkt, target_bkt) in enumerate(buckets):
        if source_bkt >= length_source and target_bkt >= length_target:
            return j, (source_bkt, target_bkt)
    return None, None


def get_parallel_bucket_doc(buckets: List[BucketDoc],
                            length_source: int,
                            length_target: int,
                            lengths_src_pre: Optional[List[int]],
                            lengths_src_nxt: Optional[List[int]],
                            lengths_tar_pre: Optional[List[int]],
                            lengths_tar_nxt: Optional[List[int]]) -> Tuple[Optional[int],
                                                                           Optional[BucketDoc]]:
    """
    Returns bucket index and bucket from a list of buckets, given source and target length and the additional
    sentence lengths. The algorithm assumes buckets are sorted from shortest to longest.
    Returns (None, None) if no bucket fits.

    :param buckets: List of buckets, in sorted order, shortest to longest.
    :param length_source: Length of source sequence.
    :param length_target: Length of target sequence.
    :param lengths_src_pre: Lengths of previous source sequences.
    :param lengths_src_nxt: Lengths of next source sequences.
    :param lengths_tar_pre: Lengths of previous target sequences.
    :param lengths_tar_nxt: Lengths of next target sequences.
    :return: Tuple of (bucket index, bucket), or (None, None) if not fitting.
    """
    for index, bucket in enumerate(buckets):
        if bucket.fits(length_source,
                       length_target,
                       lengths_src_pre,
                       lengths_src_nxt,
                       lengths_tar_pre,
                       lengths_tar_nxt):
            return index, bucket
    return None, None


def get_target_bucket(buckets: List[Tuple[int, int]],
                      length_target: int) -> Optional[Tuple[int, Tuple[int, int]]]:
    """
    Returns bucket index and bucket from a list of buckets, given source and target length.
    Returns (None, None) if no bucket fits.

    :param buckets: List of buckets.
    :param length_target: Length of target sequence.
    :return: Tuple of (bucket index, bucket), or (None, None) if not fitting.
    """
    bucket = None, None  # type: Tuple[int, Tuple[int, int]]
    for j, (source_bkt, target_bkt) in enumerate(buckets):
        if target_bkt >= length_target:
            bucket = j, (source_bkt, target_bkt)
            break
    return bucket


def get_additional_data_lengths(source_pre: List[List[str]],
                                source_nxt: List[List[str]],
                                target_pre: List[List[str]],
                                target_nxt: List[List[str]]) -> Tuple[Optional[List[int]],
                                                                      Optional[List[int]],
                                                                      Optional[List[int]],
                                                                      Optional[List[int]]]:
    """
    Computes lengths of additional data.

    :param source_pre: List of previous source sentences as strings.
    :param source_nxt: List of next source sentences as strings.
    :param target_pre: List of previous target sentences as strings.
    :param target_nxt: List of next target sentences as strings.
    :return: Respective lengths of additional input strings.
    """
    source_pre_lens = [len(src_pre) for src_pre in source_pre] if source_pre else None
    source_nxt_lens = [len(src_nxt) for src_nxt in source_nxt] if source_nxt else None
    target_pre_lens = [len(tar_pre) for tar_pre in target_pre] if target_pre else None
    target_nxt_lens = [len(tar_nxt) for tar_nxt in target_nxt] if target_nxt else None
    return source_pre_lens, source_nxt_lens, target_pre_lens, target_nxt_lens


class ParallelDataSet(Sized):
    """
    Bucketed parallel data set with labels
    """

    def __init__(self,
                 source: List[mx.nd.array],
                 target: List[mx.nd.array],
                 label: List[mx.nd.array]) -> None:
        check_condition(len(source) == len(target) == len(label),
                        "Number of buckets for source/target/label do not match: %d/%d/%d." % (len(source),
                                                                                               len(target),
                                                                                               len(label)))
        self.source = source
        self.target = target
        self.label = label

    def __len__(self) -> int:
        return len(self.source)

    def get_bucket_counts(self):
        return [len(self.source[buck_idx]) for buck_idx in range(len(self))]

    def save(self, fname: str):
        """
        Saves the dataset to a binary .npy file.
        """
        mx.nd.save(fname, self.source + self.target + self.label)

    @staticmethod
    def load(fname: str) -> 'ParallelDataSet':
        """
        Loads a dataset from a binary .npy file.
        """
        data = mx.nd.load(fname)
        n = len(data) // 3
        source = data[:n]
        target = data[n:2 * n]
        label = data[2 * n:]
        assert len(source) == len(target) == len(label)
        return ParallelDataSet(source, target, label)

    def fill_up(self,
                bucket_batch_sizes: List[BucketBatchSize],
                seed: int = 42) -> 'ParallelDataSet':
        """
        Returns a new dataset with buckets filled up.

        :param bucket_batch_sizes: Bucket batch sizes.
        :param seed: The random seed used for sampling sentences to fill up.
        :return: New dataset with buckets filled up to the next multiple of batch size
        """
        source = list(self.source)
        target = list(self.target)
        label = list(self.label)

        rs = np.random.RandomState(seed)

        for bucket_idx in range(len(self)):
            bucket = bucket_batch_sizes[bucket_idx].bucket
            bucket_batch_size = bucket_batch_sizes[bucket_idx].batch_size
            bucket_source = self.source[bucket_idx]
            bucket_target = self.target[bucket_idx]
            bucket_label = self.label[bucket_idx]
            num_samples = bucket_source.shape[0]

            # Fill up the last batch by randomly sampling from the extant items.
            if num_samples % bucket_batch_size != 0:
                rest = bucket_batch_size - num_samples % bucket_batch_size
                desired_indices_np = rs.randint(num_samples, size=rest)
                desired_indices = mx.nd.array(desired_indices_np)

                if isinstance(source[bucket_idx], np.ndarray):
                    source[bucket_idx] = np.concatenate((bucket_source, bucket_source.take(desired_indices_np)), axis=0)
                else:
                    source[bucket_idx] = mx.nd.concat(bucket_source, bucket_source.take(desired_indices), dim=0)
                target[bucket_idx] = mx.nd.concat(bucket_target, bucket_target.take(desired_indices), dim=0)
                label[bucket_idx] = mx.nd.concat(bucket_label, bucket_label.take(desired_indices), dim=0)

        return ParallelDataSet(source, target, label)

    def permute(self, permutations: List[mx.nd.NDArray]) -> 'ParallelDataSet':
        """
        Permutes the data within each bucket. The permutation is received as an argument,
        allowing the data to be unpermuted (i.e., restored) later on.

        :param permutations: For each bucket, a permutation of the data within that bucket.
        :return: A new, permuted ParallelDataSet.
        """
        assert len(self) == len(permutations)
        source = []
        target = []
        label = []
        for buck_idx in range(len(self)):
            num_samples = self.source[buck_idx].shape[0]
            if num_samples:  # not empty bucket
                permutation = permutations[buck_idx]
                if isinstance(self.source[buck_idx], np.ndarray):
                    source.append(self.source[buck_idx].take(np.int64(permutation.asnumpy())))
                else:
                    source.append(self.source[buck_idx].take(permutation))
                target.append(self.target[buck_idx].take(permutation))
                label.append(self.label[buck_idx].take(permutation))
            else:
                source.append(self.source[buck_idx])
                target.append(self.target[buck_idx])
                label.append(self.label[buck_idx])

        return ParallelDataSet(source, target, label)


class ParallelDataSetDoc(ParallelDataSet):
    """
    Bucketed parallel data set with labels for document-level context models.
    """

    def __init__(self,
                 source: List[mx.nd.array],
                 target: List[mx.nd.array],
                 label: List[mx.nd.array],
                 src_pre: List[mx.nd.array],
                 src_nxt: List[mx.nd.array],
                 tar_pre: List[mx.nd.array],
                 tar_nxt: List[mx.nd.array]) -> None:
        super().__init__(source=source, target=target, label=label)
        self.src_pre = src_pre
        self.src_nxt = src_nxt
        self.tar_pre = tar_pre
        self.tar_nxt = tar_nxt

    def __len__(self) -> int:
        return len(self.source)

    def get_bucket_counts(self):
        return [len(self.source[buck_idx]) for buck_idx in range(len(self))]

    def save(self, fname: str):
        """
        Saves the dataset to a binary .npy file.
        """
        mx.nd.save(fname, self.source + self.target + self.label + self.src_pre + self.src_nxt + self.tar_pre + self.tar_nxt)

    @staticmethod
    def load(fname: str) -> 'ParallelDataSetDoc':
        """
        Loads a dataset from a binary .npy file.
        """
        data = mx.nd.load(fname)
        n = len(data) // 3
        source = data[:n]
        target = data[n:2 * n]
        label = data[2 * n:3 * n]
        src_pre = data[3 * n:4 * n]
        src_nxt = data[4 * n:5 * n]
        tar_pre = data[5 * n:6 * n]
        tar_nxt = data[6 * n:]
        assert len(source) == len(target) == len(label)
        return ParallelDataSetDoc(source, target, label, src_pre, src_nxt, tar_pre, tar_nxt)

    def fill_up(self,
                bucket_batch_sizes: List[BucketBatchSizeDoc],
                seed: int = 42) -> 'ParallelDataSetDoc':
        """
        Returns a new dataset with buckets filled up including context data.

        :param bucket_batch_sizes: Bucket batch sizes.
        :param seed: The random seed used for sampling sentences to fill up.
        :return: New dataset with buckets filled up to the next multiple of batch size
        """
        source = list(self.source)
        target = list(self.target)
        label = list(self.label)
        source_pre = list(self.src_pre)
        source_nxt = list(self.src_nxt)
        target_pre = list(self.tar_pre)
        target_nxt = list(self.tar_nxt)

        rs = np.random.RandomState(seed)

        for bucket_idx in range(len(self)):
            bucket = bucket_batch_sizes[bucket_idx].bucket
            bucket_batch_size = bucket_batch_sizes[bucket_idx].batch_size
            bucket_source = self.source[bucket_idx]
            bucket_target = self.target[bucket_idx]
            bucket_label = self.label[bucket_idx]
            bucket_source_pre = [bucket_src_pre for bucket_src_pre in self.src_pre[bucket_idx]]
            bucket_source_nxt = [bucket_src_nxt for bucket_src_nxt in self.src_nxt[bucket_idx]]
            bucket_target_pre = [bucket_tar_pre for bucket_tar_pre in self.tar_pre[bucket_idx]]
            bucket_target_nxt = [bucket_tar_nxt for bucket_tar_nxt in self.tar_nxt[bucket_idx]]
            num_samples = bucket_source.shape[0]

            # Fill up the last batch by randomly sampling from the extant items.
            if num_samples % bucket_batch_size != 0:
                rest = bucket_batch_size - num_samples % bucket_batch_size
                desired_indices_np = rs.randint(num_samples, size=rest)
                desired_indices = mx.nd.array(desired_indices_np)

                if isinstance(source[bucket_idx], np.ndarray):
                    source[bucket_idx] = np.concatenate((bucket_source, bucket_source.take(desired_indices_np)), axis=0)
                else:
                    source[bucket_idx] = mx.nd.concat(bucket_source, bucket_source.take(desired_indices), dim=0)
                target[bucket_idx] = mx.nd.concat(bucket_target, bucket_target.take(desired_indices), dim=0)
                label[bucket_idx] = mx.nd.concat(bucket_label, bucket_label.take(desired_indices), dim=0)
                source_pre[bucket_idx] = [mx.nd.concat(bucket_src_pre_entry,
                                                       bucket_src_pre_entry.take(desired_indices), dim=0)
                                          for bucket_src_pre_entry in bucket_source_pre]
                source_nxt[bucket_idx] = [mx.nd.concat(bucket_src_nxt_entry,
                                                       bucket_src_nxt_entry.take(desired_indices), dim=0)
                                          for bucket_src_nxt_entry in bucket_source_nxt]
                target_pre[bucket_idx] = [mx.nd.concat(bucket_tar_pre_entry,
                                                       bucket_tar_pre_entry.take(desired_indices), dim=0)
                                          for bucket_tar_pre_entry in bucket_target_pre]
                target_nxt[bucket_idx] = [mx.nd.concat(bucket_tar_nxt_entry,
                                                       bucket_tar_nxt_entry.take(desired_indices), dim=0)
                                          for bucket_tar_nxt_entry in bucket_target_nxt]

        return ParallelDataSetDoc(source, target, label,
                                  source_pre, source_nxt, target_pre, target_nxt)

    def permute(self, permutations: List[mx.nd.NDArray]) -> 'ParallelDataSetDoc':
        """
        Permutes the data within each bucket. The permutation is received as an argument,
        allowing the data to be unpermuted (i.e., restored) later on.

        :param permutations: For each bucket, a permutation of the data within that bucket.
        :return: A new, permuted ParallelDataSet.
        """
        assert len(self) == len(permutations)
        source = []  # type: List[mx.nd.NDArray]
        target = []  # type: List[mx.nd.NDArray]
        label = []  # type: List[mx.nd.NDArray]
        source_pre = []  # type: List[List[mx.nd.NDArray]]
        source_nxt = []  # type: List[List[mx.nd.NDArray]]
        target_pre = []  # type: List[List[mx.nd.NDArray]]
        target_nxt = []  # type: List[List[mx.nd.NDArray]]
        for buck_idx in range(len(self)):
            num_samples = self.source[buck_idx].shape[0]
            if num_samples:  # not empty bucket
                permutation = permutations[buck_idx]
                if isinstance(self.source[buck_idx], np.ndarray):
                    source.append(self.source[buck_idx].take(np.int64(permutation.asnumpy())))
                else:
                    source.append(self.source[buck_idx].take(permutation))
                target.append(self.target[buck_idx].take(permutation))
                label.append(self.label[buck_idx].take(permutation))
                source_pre.append([src_pre_entry.take(permutation) for src_pre_entry in self.src_pre[buck_idx]])
                source_nxt.append([src_nxt_entry.take(permutation) for src_nxt_entry in self.src_nxt[buck_idx]])
                target_pre.append([tar_pre_entry.take(permutation) for tar_pre_entry in self.tar_pre[buck_idx]])
                target_nxt.append([tar_nxt_entry.take(permutation) for tar_nxt_entry in self.tar_nxt[buck_idx]])
            else:
                source.append(self.source[buck_idx])
                target.append(self.target[buck_idx])
                label.append(self.label[buck_idx])
                source_pre.append([src_pre_entry for src_pre_entry in self.src_pre[buck_idx]])
                source_nxt.append([src_nxt_entry for src_nxt_entry in self.src_nxt[buck_idx]])
                target_pre.append([tar_pre_entry for tar_pre_entry in self.tar_pre[buck_idx]])
                target_nxt.append([tar_nxt_entry for tar_nxt_entry in self.tar_nxt[buck_idx]])

        return ParallelDataSetDoc(source, target, label,
                                  source_pre, source_nxt, target_pre, target_nxt)


def get_permutations(bucket_counts: List[int]) -> Tuple[List[mx.nd.NDArray], List[mx.nd.NDArray]]:
    """
    Returns the indices of a random permutation for each bucket and the corresponding inverse permutations that can
    restore the original order of the data if applied to the permuted data.

    :param bucket_counts: The number of elements per bucket.
    :return: For each bucket a permutation and inverse permutation is returned.
    """
    data_permutations = []  # type: List[mx.nd.NDArray]
    inverse_data_permutations = []  # type: List[mx.nd.NDArray]
    for num_samples in bucket_counts:
        if num_samples == 0:
            num_samples = 1
        # new random order:
        data_permutation = np.random.permutation(num_samples)
        inverse_data_permutation = np.empty(num_samples, np.int32)
        inverse_data_permutation[data_permutation] = np.arange(num_samples)
        inverse_data_permutation = mx.nd.array(inverse_data_permutation)
        data_permutation = mx.nd.array(data_permutation)

        data_permutations.append(data_permutation)
        inverse_data_permutations.append(inverse_data_permutation)
    return data_permutations, inverse_data_permutations


def get_batch_indices(data: ParallelDataSet,
                      bucket_batch_sizes: List[BucketBatchSize]) -> List[Tuple[int, int]]:
    """
    Returns a list of index tuples that index into the bucket and the start index inside a bucket given
    the batch size for a bucket. These indices are valid for the given dataset.

    Put another way, this returns the starting points for all batches within the dataset, across all buckets.

    :param data: Data to create indices for.
    :param bucket_batch_sizes: Bucket batch sizes.
    :return: List of 2d indices.
    """
    # create index tuples (i,j) into buckets: i := bucket index ; j := row index of bucket array
    idxs = []  # type: List[Tuple[int, int]]
    for buck_idx, buck in enumerate(data.source):
        bucket = bucket_batch_sizes[buck_idx].bucket
        batch_size = bucket_batch_sizes[buck_idx].batch_size
        num_samples = data.source[buck_idx].shape[0]
        rest = num_samples % batch_size
        if rest > 0:
            logger.info("Ignoring %d samples from bucket %s with %d samples due to incomplete batch",
                        rest, bucket, num_samples)
        idxs.extend([(buck_idx, j) for j in range(0, num_samples - batch_size + 1, batch_size)])
    return idxs


def get_batch_indices_doc(data: ParallelDataSetDoc,
                          bucket_batch_sizes: List[BucketBatchSizeDoc]) -> List[Tuple[int, int]]:
    """
    Returns a list of index tuples that index into the bucket and the start index inside a bucket given
    the batch size for a bucket. These indices are valid for the given dataset.

    Put another way, this returns the starting points for all batches within the dataset, across all buckets.

    :param data: Data to create indices for.
    :param bucket_batch_sizes: Bucket batch sizes.
    :return: List of 2d indices.
    """
    # create index tuples (i,j) into buckets: i := bucket index ; j := row index of bucket array
    idxs = []  # type: List[Tuple[int, int]]
    for buck_idx, buck in enumerate(data.source):
        bucket = bucket_batch_sizes[buck_idx].bucket
        batch_size = bucket_batch_sizes[buck_idx].batch_size
        num_samples = data.source[buck_idx].shape[0]
        rest = num_samples % batch_size
        if rest > 0:
            logger.info("Ignoring %d samples from bucket %s with %d samples due to incomplete batch",
                        rest, bucket, num_samples)
        idxs.extend([(buck_idx, j) for j in range(0, num_samples - batch_size + 1, batch_size)])
    return idxs


class MetaBaseParallelSampleIter(ABC):
    pass


class BaseParallelSampleIter(mx.io.DataIter):
    """
    Base parallel sample iterator.

    :param buckets: The list of buckets.
    :param bucket_batch_sizes: A list, parallel to `buckets`, containing the number of samples in each bucket.
    :param source_data_name: The source data name.
    :param target_data_name: The target data name.
    :param label_name: The label name.
    :param num_factors: The number of source factors.
    :param permute: Randomly shuffle the parallel data.
    :param dtype: The MXNet data type.
    """
    __metaclass__ = MetaBaseParallelSampleIter

    def __init__(self,
                 buckets: List[Tuple[int, int]],
                 batch_size: int,
                 bucket_batch_sizes: List[BucketBatchSize],
                 source_data_name: str,
                 target_data_name: str,
                 label_name: str,
                 num_factors: int = 1,
                 permute: bool = True,
                 dtype='float32') -> None:
        super().__init__(batch_size=batch_size)

        self.buckets = list(buckets)
        self.default_bucket_key = get_default_bucket_key(self.buckets)
        self.bucket_batch_sizes = bucket_batch_sizes
        self.source_data_name = source_data_name
        self.target_data_name = target_data_name
        self.label_name = label_name
        self.num_factors = num_factors
        self.permute = permute
        self.dtype = dtype

        # "Staging area" that needs to fit any size batch we're using by total number of elements.
        # When computing per-bucket batch sizes, we guarantee that the default bucket will have the
        # largest total batch size.
        # Note: this guarantees memory sharing for input data and is generally a good heuristic for
        # other parts of the model, but it is possible that some architectures will have intermediate
        # operations that produce shapes larger than the default bucket size.  In these cases, MXNet
        # will silently allocate additional memory.
        self.provide_data = [
            mx.io.DataDesc(name=self.source_data_name,
                           shape=(self.bucket_batch_sizes[-1].batch_size, self.default_bucket_key[0],
                                  self.num_factors),
                           layout=C.BATCH_MAJOR),
            mx.io.DataDesc(name=self.target_data_name,
                           shape=(self.bucket_batch_sizes[-1].batch_size, self.default_bucket_key[1]),
                           layout=C.BATCH_MAJOR)]
        self.provide_label = [
            mx.io.DataDesc(name=self.label_name,
                           shape=(self.bucket_batch_sizes[-1].batch_size, self.default_bucket_key[1]),
                           layout=C.BATCH_MAJOR)]

        self.data_names = [self.source_data_name, self.target_data_name]
        self.label_names = [self.label_name]

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def iter_next(self) -> bool:
        pass

    @abstractmethod
    def next(self) -> mx.io.DataBatch:
        pass

    @abstractmethod
    def save_state(self, fname: str):
        pass

    @abstractmethod
    def load_state(self, fname: str):
        pass


class BaseParallelSampleIterDoc(mx.io.DataIter):
    """
    Base parallel sample iterator for document level exploitation.

    :param buckets: The list of buckets.
    :param bucket_batch_sizes: A list, parallel to `buckets`, containing the number of samples in each bucket.
    :param source_data_name: The source data name.
    :param target_data_name: The target data name.
    :param label_name: The label name.
    :param num_factors: The number of source factors.
    :param permute: Randomly shuffle the parallel data.
    :param dtype: The MXNet data type.
    """
    __metaclass__ = MetaBaseParallelSampleIter

    def __init__(self,
                 window_config: doc_context.WindowConfig,
                 buckets: List[BucketDoc],
                 batch_size: int,
                 bucket_batch_sizes: List[BucketBatchSizeDoc],
                 source_data_name: str,
                 target_data_name: str,
                 label_name: str,
                 source_pre_data_name: str,
                 source_nxt_data_name: str,
                 target_pre_data_name: str,
                 target_nxt_data_name: str,
                 num_factors: int = 1,
                 permute: bool = True,
                 dtype='float32') -> None:
        super().__init__(batch_size=batch_size)

        self.window_config = window_config
        self.buckets = list(buckets)
        self.default_bucket_key = get_default_bucket_key(self.buckets)
        self.bucket_batch_sizes = bucket_batch_sizes
        self.source_data_name = source_data_name
        self.target_data_name = target_data_name
        self.label_name = label_name
        self.source_pre_data_name = source_pre_data_name
        self.source_nxt_data_name = source_nxt_data_name
        self.target_pre_data_name = target_pre_data_name
        self.target_nxt_data_name = target_nxt_data_name
        self.context_name_formats = [self.source_pre_data_name, self.source_nxt_data_name,
                                     self.target_pre_data_name, self.target_nxt_data_name]
        self.num_factors = num_factors
        self.permute = permute
        self.dtype = dtype

        # "Staging area" that needs to fit any size batch we're using by total number of elements.
        # When computing per-bucket batch sizes, we guarantee that the default bucket will have the
        # largest total batch size.
        # Note: this guarantees memory sharing for input data and is generally a good heuristic for
        # other parts of the model, but it is possible that some architectures will have intermediate
        # operations that produce shapes larger than the default bucket size.  In these cases, MXNet
        # will silently allocate additional memory.
        last_batch_size = self.bucket_batch_sizes[-1].batch_size
        self.provide_data = [
            mx.io.DataDesc(name=self.source_data_name,
                           shape=(last_batch_size, self.default_bucket_key.src_len,
                                  self.num_factors),
                           layout=C.BATCH_MAJOR),
            mx.io.DataDesc(name=self.target_data_name,
                           shape=(last_batch_size, self.default_bucket_key.tar_len),
                           layout=C.BATCH_MAJOR)]
        self.provide_label = [
            mx.io.DataDesc(name=self.label_name,
                           shape=(last_batch_size, self.default_bucket_key.tar_len),
                           layout=C.BATCH_MAJOR)]

        self.data_names = [self.source_data_name, self.target_data_name]
        self.label_names = [self.label_name]

        # Add context sentence data descriptions
        for window_size, context_name_format, bucket in zip(self.window_config.sizes,
                                                            self.context_name_formats,
                                                            [self.default_bucket_key.src_pre_lens,
                                                             self.default_bucket_key.src_nxt_lens,
                                                             self.default_bucket_key.tar_pre_lens,
                                                             self.default_bucket_key.tar_nxt_lens]):
            for i in range(window_size):
                self.provide_data.append(
                        mx.io.DataDesc(name=context_name_format % i,
                                       shape=(last_batch_size, bucket[i], 1),
                                       layout=C.BATCH_MAJOR)
                )
                self.data_names.append(context_name_format % i)

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def iter_next(self) -> bool:
        pass

    @abstractmethod
    def next(self) -> mx.io.DataBatch:
        pass

    @abstractmethod
    def save_state(self, fname: str):
        pass

    @abstractmethod
    def load_state(self, fname: str):
        pass


class BatchedRawParallelSampleIter(BaseParallelSampleIter):
    """
    Goes through the raw data, loading only one batch at a time into memory.
    Used by the scorer. Iterates through the data in order, and therefore does
    not support bucketing.
    """

    def __init__(self,
                 data_loader: RawParallelDatasetLoader,
                 sources: List[str],
                 target: str,
                 source_vocabs: List[vocab.Vocab],
                 target_vocab: vocab.Vocab,
                 bucket: Tuple[int, int],
                 batch_size: int,
                 max_lens: Tuple[int, int],
                 num_factors: int = 1,
                 source_data_name=C.SOURCE_NAME,
                 target_data_name=C.TARGET_NAME,
                 label_name=C.TARGET_LABEL_NAME,
                 dtype='float32') -> None:
        super().__init__(buckets=[bucket], batch_size=batch_size, bucket_batch_sizes=[BucketBatchSize(bucket, batch_size, None)],
                         source_data_name=source_data_name, target_data_name=target_data_name,
                         label_name=label_name, num_factors=num_factors, permute=False, dtype=dtype)
        self.data_loader = data_loader
        self.sources_sentences, self.target_sentences = create_sequence_readers(sources, target, source_vocabs, target_vocab)
        self.sources_iters = [iter(s) for s in self.sources_sentences]
        self.target_iter = iter(self.target_sentences)
        self.max_len_source, self.max_len_target = max_lens
        self.next_batch = None
        self.sentno = 1

    def reset(self):
        raise Exception('Not supported!')

    def iter_next(self) -> bool:
        """
        True if the iterator can return another batch.
        """

        # Read batch_size lines from the source stream
        sources_sentences = [[] for x in self.sources_sentences]  # type: List[List[str]]
        target_sentences = []  # type: List[str]
        num_read = 0
        for num_read, (sources, target) in enumerate(parallel_iterate(self.sources_iters, self.target_iter, skip_blanks=False), 1):
            source_len = 0 if sources[0] is None else len(sources[0])
            target_len = 0 if target is None else len(target)
            if source_len > self.max_len_source:
                logger.info("Trimming source sentence {} ({} -> {})".format(self.sentno + num_read, source_len, self.max_len_source))
                sources = [source[0:self.max_len_source] for source in sources]
            if target_len > self.max_len_target:
                logger.info("Trimming target sentence {} ({} -> {})".format(self.sentno + num_read, target_len, self.max_len_target))
                target = target[0:self.max_len_target]

            for i, source in enumerate(sources):
                sources_sentences[i].append(source)
            target_sentences.append(target)
            if num_read == self.batch_size:
                break

        self.sentno += num_read

        if num_read == 0:
            self.next_batch = None
            return False

        # The final batch may be underfilled, so mark it
        num_pad = self.batch_size - num_read

        dataset = self.data_loader.load(sources_sentences,
                                        target_sentences,
                                        [num_read]).fill_up(self.bucket_batch_sizes)

        data = [dataset.source[0], dataset.target[0]]
        label = dataset.label

        provide_data = [mx.io.DataDesc(name=n, shape=x.shape, layout=C.BATCH_MAJOR) for n, x in
                        zip(self.data_names, data)]
        provide_label = [mx.io.DataDesc(name=n, shape=x.shape, layout=C.BATCH_MAJOR) for n, x in
                         zip(self.label_names, label)]

        self.next_batch = mx.io.DataBatch(data, label,
                                          pad=num_pad, index=None, bucket_key=self.buckets[0],
                                          provide_data=provide_data, provide_label=provide_label)

        return True

    def next(self) -> mx.io.DataBatch:
        """
        Returns the next batch.
        """
        if self.iter_next():
            return self.next_batch
        raise StopIteration

    def save_state(self, fname: str):
        raise Exception('Not supported!')

    def load_state(self, fname: str):
        raise Exception('Not supported!')


class ShardedParallelSampleIter(BaseParallelSampleIter):
    """
    Goes through the data one shard at a time. The memory consumption is limited by the memory consumption of the
    largest shard. The order in which shards are traversed is changed with each reset.
    """

    def __init__(self,
                 shards_fnames: List[str],
                 buckets,
                 batch_size,
                 bucket_batch_sizes,
                 source_data_name=C.SOURCE_NAME,
                 target_data_name=C.TARGET_NAME,
                 label_name=C.TARGET_LABEL_NAME,
                 num_factors: int = 1,
                 permute: bool = True,
                 dtype='float32') -> None:
        super().__init__(buckets=buckets, batch_size=batch_size, bucket_batch_sizes=bucket_batch_sizes,
                         source_data_name=source_data_name, target_data_name=target_data_name,
                         label_name=label_name, num_factors=num_factors, permute=permute, dtype=dtype)
        assert len(shards_fnames) > 0
        self.shards_fnames = list(shards_fnames)
        self.shard_index = -1

        self.reset()

    def _load_shard(self):
        shard_fname = self.shards_fnames[self.shard_index]
        logger.info("Loading shard %s.", shard_fname)
        dataset = ParallelDataSet.load(self.shards_fnames[self.shard_index]).fill_up(self.bucket_batch_sizes,
                                                                                     seed=self.shard_index)
        self.shard_iter = ParallelSampleIter(data=dataset,
                                             buckets=self.buckets,
                                             batch_size=self.batch_size,
                                             bucket_batch_sizes=self.bucket_batch_sizes,
                                             source_data_name=self.source_data_name,
                                             target_data_name=self.target_data_name,
                                             num_factors=self.num_factors,
                                             permute=self.permute)

    def reset(self):
        if len(self.shards_fnames) > 1:
            logger.info("Shuffling the shards.")
            # Making sure to not repeat a shard:
            if self.shard_index < 0:
                current_shard_fname = ""
            else:
                current_shard_fname = self.shards_fnames[self.shard_index]
            remaining_shards = [shard for shard in self.shards_fnames if shard != current_shard_fname]
            next_shard_fname = random.choice(remaining_shards)
            remaining_shards = [shard for shard in self.shards_fnames if shard != next_shard_fname]
            random.shuffle(remaining_shards)

            self.shards_fnames = [next_shard_fname] + remaining_shards

            self.shard_index = 0
            self._load_shard()
        else:
            if self.shard_index < 0:
                self.shard_index = 0
                self._load_shard()
            # We can just reset the shard_iter as we only have a single shard
            self.shard_iter.reset()

    def iter_next(self) -> bool:
        next_shard_index = self.shard_index + 1
        return self.shard_iter.iter_next() or next_shard_index < len(self.shards_fnames)

    def next(self) -> mx.io.DataBatch:
        if not self.shard_iter.iter_next():
            if self.shard_index < len(self.shards_fnames) - 1:
                self.shard_index += 1
                self._load_shard()
            else:
                raise StopIteration
        return self.shard_iter.next()

    def save_state(self, fname: str):
        with open(fname, "wb") as fp:
            pickle.dump(self.shards_fnames, fp)
            pickle.dump(self.shard_index, fp)
        self.shard_iter.save_state(fname + ".sharditer")

    def load_state(self, fname: str):
        with open(fname, "rb") as fp:
            self.shards_fnames = pickle.load(fp)
            self.shard_index = pickle.load(fp)
        self._load_shard()
        self.shard_iter.load_state(fname + ".sharditer")


class ParallelSampleIter(BaseParallelSampleIter):
    """
    Data iterator on a bucketed ParallelDataSet. Shuffles data at every reset and supports saving and loading the
    iterator state.
    """

    def __init__(self,
                 data: ParallelDataSet,
                 buckets,
                 batch_size,
                 bucket_batch_sizes,
                 source_data_name=C.SOURCE_NAME,
                 target_data_name=C.TARGET_NAME,
                 label_name=C.TARGET_LABEL_NAME,
                 num_factors: int = 1,
                 permute: bool = True,
                 dtype='float32') -> None:
        super().__init__(buckets=buckets, batch_size=batch_size, bucket_batch_sizes=bucket_batch_sizes,
                         source_data_name=source_data_name, target_data_name=target_data_name,
                         label_name=label_name, num_factors=num_factors, permute=permute, dtype=dtype)

        # create independent lists to be shuffled
        self.data = ParallelDataSet(list(data.source), list(data.target), list(data.label))

        # create index tuples (buck_idx, batch_start_pos) into buckets.
        # This is the list of all batches across all buckets in the dataset. These will be shuffled.
        self.batch_indices = get_batch_indices(self.data, bucket_batch_sizes)
        self.curr_batch_index = 0

        # Produces a permutation of the batches within each bucket, along with the permutation that inverts it.
        self.inverse_data_permutations = [mx.nd.arange(0, max(1, self.data.source[i].shape[0]))
                                          for i in range(len(self.data))]
        self.data_permutations = [mx.nd.arange(0, max(1, self.data.source[i].shape[0]))
                                  for i in range(len(self.data))]

        self.reset()

    def reset(self):
        """
        Resets and reshuffles the data.
        """
        self.curr_batch_index = 0
        if self.permute:
            # shuffle batch start indices
            random.shuffle(self.batch_indices)

            # restore the data permutation
            self.data = self.data.permute(self.inverse_data_permutations)

            # permute the data within each batch
            self.data_permutations, self.inverse_data_permutations = get_permutations(self.data.get_bucket_counts())
            self.data = self.data.permute(self.data_permutations)

    def iter_next(self) -> bool:
        """
        True if iterator can return another batch
        """
        return self.curr_batch_index != len(self.batch_indices)

    def next(self) -> mx.io.DataBatch:
        """
        Returns the next batch from the data iterator.
        """
        if not self.iter_next():
            raise StopIteration

        i, j = self.batch_indices[self.curr_batch_index]
        self.curr_batch_index += 1

        batch_size = self.bucket_batch_sizes[i].batch_size
        source = self.data.source[i][j:j + batch_size]
        target = self.data.target[i][j:j + batch_size]
        data = [source, target]
        label = [self.data.label[i][j:j + batch_size]]

        provide_data = [mx.io.DataDesc(name=n, shape=x.shape, layout=C.BATCH_MAJOR) for n, x in
                        zip(self.data_names, data)]
        provide_label = [mx.io.DataDesc(name=n, shape=x.shape, layout=C.BATCH_MAJOR) for n, x in
                         zip(self.label_names, label)]

        # TODO: num pad examples is not set here if fillup policy would be padding
        return mx.io.DataBatch(data, label,
                               pad=0, index=None, bucket_key=self.buckets[i],
                               provide_data=provide_data, provide_label=provide_label)

    def save_state(self, fname: str):
        """
        Saves the current state of iterator to a file, so that iteration can be
        continued. Note that the data is not saved, i.e. the iterator must be
        initialized with the same parameters as in the first call.

        :param fname: File name to save the information to.
        """
        with open(fname, "wb") as fp:
            pickle.dump(self.batch_indices, fp)
            pickle.dump(self.curr_batch_index, fp)
            np.save(fp, [a.asnumpy() for a in self.inverse_data_permutations])
            np.save(fp, [a.asnumpy() for a in self.data_permutations])

    def load_state(self, fname: str):
        """
        Loads the state of the iterator from a file.

        :param fname: File name to load the information from.
        """

        # restore order
        self.data = self.data.permute(self.inverse_data_permutations)

        with open(fname, "rb") as fp:
            self.batch_indices = pickle.load(fp)
            self.curr_batch_index = pickle.load(fp)
            inverse_data_permutations = np.load(fp)
            data_permutations = np.load(fp)

        # Right after loading the iterator state, next() should be called
        self.curr_batch_index -= 1

        # load previous permutations
        self.inverse_data_permutations = []
        self.data_permutations = []

        for bucket in range(len(self.data)):
            inverse_permutation = mx.nd.array(inverse_data_permutations[bucket])
            self.inverse_data_permutations.append(inverse_permutation)

            permutation = mx.nd.array(data_permutations[bucket])
            self.data_permutations.append(permutation)

        self.data = self.data.permute(self.data_permutations)


class ParallelSampleIterDoc(BaseParallelSampleIterDoc):
    """
    Data iterator on a bucketed ParallelDataSetDoc. Shuffles data at every reset and supports
    saving and loading the iterator state, including context information data.
    """

    def __init__(self,
                 window_config: doc_context.WindowConfig,
                 data: ParallelDataSetDoc,
                 buckets: List[BucketDoc],
                 batch_size: int,
                 bucket_batch_sizes: List[BucketBatchSizeDoc],
                 source_data_name: str = C.SOURCE_NAME,
                 target_data_name: str = C.TARGET_NAME,
                 label_name: str = C.TARGET_LABEL_NAME,
                 source_pre_data_name: str = doc_context.SOURCE_PRE_NAME,
                 source_nxt_data_name: str = doc_context.SOURCE_NXT_NAME,
                 target_pre_data_name: str = doc_context.TARGET_PRE_NAME,
                 target_nxt_data_name: str = doc_context.TARGET_NXT_NAME,
                 num_factors: int = 1,
                 permute: bool = True,
                 dtype: str = 'float32') -> None:
        super().__init__(window_config=window_config,
                         buckets=buckets, batch_size=batch_size, bucket_batch_sizes=bucket_batch_sizes,
                         source_data_name=source_data_name, target_data_name=target_data_name,
                         source_pre_data_name=source_pre_data_name, source_nxt_data_name=source_nxt_data_name,
                         target_pre_data_name=target_pre_data_name, target_nxt_data_name=target_nxt_data_name,
                         label_name=label_name, num_factors=num_factors, permute=permute, dtype=dtype)

        # create independent lists to be shuffled
        self.data = ParallelDataSetDoc(list(data.source), list(data.target), list(data.label),
                                       list(data.src_pre), list(data.src_nxt),
                                       list(data.tar_pre), list(data.tar_nxt))

        # create index tuples (buck_idx, batch_start_pos) into buckets.
        # This is the list of all batches across all buckets in the dataset. These will be shuffled.
        self.batch_indices = get_batch_indices_doc(self.data, bucket_batch_sizes)
        self.curr_batch_index = 0

        # Produces a permutation of the batches within each bucket, along with the permutation that inverts it.
        self.inverse_data_permutations = [mx.nd.arange(0, max(1, self.data.source[i].shape[0]))
                                          for i in range(len(self.data))]
        self.data_permutations = [mx.nd.arange(0, max(1, self.data.source[i].shape[0]))
                                  for i in range(len(self.data))]

        self.reset()

    def reset(self):
        """
        Resets and reshuffles the data.
        """
        self.curr_batch_index = 0
        if self.permute:
            # shuffle batch start indices
            random.shuffle(self.batch_indices)

            # restore the data permutation
            self.data = self.data.permute(self.inverse_data_permutations)

            # permute the data within each batch
            self.data_permutations, self.inverse_data_permutations = get_permutations(self.data.get_bucket_counts())
            self.data = self.data.permute(self.data_permutations)

    def iter_next(self) -> bool:
        """
        True if iterator can return another batch
        """
        return self.curr_batch_index != len(self.batch_indices)

    def next(self) -> mx.io.DataBatch:
        """
        Returns the next batch from the data iterator.
        """
        if not self.iter_next():
            raise StopIteration

        i, j = self.batch_indices[self.curr_batch_index]
        self.curr_batch_index += 1

        batch_size = self.bucket_batch_sizes[i].batch_size
        source = self.data.source[i][j:j + batch_size]
        target = self.data.target[i][j:j + batch_size]
        data = [source, target]
        label = [self.data.label[i][j:j + batch_size]]

        for size, additional_data in zip(self.window_config.sizes,
                                         [self.data.src_pre, self.data.src_nxt,
                                          self.data.tar_pre, self.data.tar_nxt]):
            for k in range(size):
                data.append(additional_data[i][k][j:j + batch_size])

        provide_data = [mx.io.DataDesc(name=n, shape=x.shape, layout=C.BATCH_MAJOR) for n, x in
                        zip(self.data_names, data)]
        provide_label = [mx.io.DataDesc(name=n, shape=x.shape, layout=C.BATCH_MAJOR) for n, x in
                         zip(self.label_names, label)]
        return mx.io.DataBatch(data, label,
                               pad=0, index=None, bucket_key=self.buckets[i],
                               provide_data=provide_data, provide_label=provide_label)

    def save_state(self, fname: str):
        """
        Saves the current state of iterator to a file, so that iteration can be
        continued. Note that the data is not saved, i.e. the iterator must be
        initialized with the same parameters as in the first call.

        :param fname: File name to save the information to.
        """
        with open(fname, "wb") as fp:
            pickle.dump(self.batch_indices, fp)
            pickle.dump(self.curr_batch_index, fp)
            np.save(fp, [a.asnumpy() for a in self.inverse_data_permutations])
            np.save(fp, [a.asnumpy() for a in self.data_permutations])

    def load_state(self, fname: str):
        """
        Loads the state of the iterator from a file.

        :param fname: File name to load the information from.
        """

        # restore order
        self.data = self.data.permute(self.inverse_data_permutations)

        with open(fname, "rb") as fp:
            self.batch_indices = pickle.load(fp)
            self.curr_batch_index = pickle.load(fp)
            inverse_data_permutations = np.load(fp)
            data_permutations = np.load(fp)

        # Right after loading the iterator state, next() should be called
        self.curr_batch_index -= 1

        # load previous permutations
        self.inverse_data_permutations = []
        self.data_permutations = []

        for bucket in range(len(self.data)):
            inverse_permutation = mx.nd.array(inverse_data_permutations[bucket])
            self.inverse_data_permutations.append(inverse_permutation)

            permutation = mx.nd.array(data_permutations[bucket])
            self.data_permutations.append(permutation)

        self.data = self.data.permute(self.data_permutations)
