# Copyright 2017--2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
Translation CLI.
"""
import argparse
import sys
import time
import logging
from contextlib import ExitStack
from typing import cast, Generator, Iterable, Iterator, Optional, List, Sequence, Tuple, Union

from sockeye.lexicon import TopKLexicon
from sockeye.log import setup_main_logger
from sockeye.output_handler import get_output_handler, OutputHandler
from sockeye.utils import determine_context, log_basic_info, check_condition, grouper
from . import arguments
from . import constants as C
from . import data_io
from . import doc_context
from . import inference
from . import utils

logger = logging.getLogger(__name__)


def main():
    params = arguments.ConfigArgumentParser(description='Translate CLI')
    arguments.add_translate_cli_args(params)
    arguments.add_translate_cli_args_doc(params)
    args = params.parse_args()
    run_translate(args)


def run_translate(args: argparse.Namespace):

    # Seed randomly unless a seed has been passed
    utils.seed_rngs(args.seed if args.seed is not None else int(time.time()))

    if args.output is not None:
        setup_main_logger(console=not args.quiet,
                          file_logging=True,
                          path="%s.%s" % (args.output, C.LOG_NAME),
                          level=args.loglevel)
    else:
        setup_main_logger(file_logging=False, level=args.loglevel)

    log_basic_info(args)

    if args.nbest_size > 1:
        if args.output_type != C.OUTPUT_HANDLER_JSON:
            logger.warning("For nbest translation, you must specify `--output-type '%s'; overriding your setting of '%s'.",
                           C.OUTPUT_HANDLER_JSON, args.output_type)
            args.output_type = C.OUTPUT_HANDLER_JSON
    output_handler = get_output_handler(args.output_type,
                                        args.output,
                                        args.sure_align_threshold)

    with ExitStack() as exit_stack:
        check_condition(len(args.device_ids) == 1, "translate only supports single device for now")
        context = determine_context(device_ids=args.device_ids,
                                    use_cpu=args.use_cpu,
                                    disable_device_locking=args.disable_device_locking,
                                    lock_dir=args.lock_dir,
                                    exit_stack=exit_stack)[0]
        logger.info("Translate Device: %s", context)

        models, source_vocabs, target_vocab = inference.load_models(
            context=context,
            max_input_len=args.max_input_len,
            beam_size=args.beam_size,
            batch_size=args.batch_size,
            model_folders=args.models,
            checkpoints=args.checkpoints,
            softmax_temperature=args.softmax_temperature,
            max_output_length_num_stds=args.max_output_length_num_stds,
            decoder_return_logit_inputs=args.restrict_lexicon is not None,
            cache_output_layer_w_b=args.restrict_lexicon is not None,
            override_dtype=args.override_dtype,
            output_scores=output_handler.reports_score(),
            sampling=args.sample,
            doc_context_method=args.method)
        restrict_lexicon = None  # type: Optional[TopKLexicon]
        if args.restrict_lexicon:
            restrict_lexicon = TopKLexicon(source_vocabs[0], target_vocab)
            restrict_lexicon.load(args.restrict_lexicon, k=args.restrict_lexicon_topk)
        store_beam = args.output_type == C.OUTPUT_HANDLER_BEAM_STORE

        translator_selection = {
            doc_context.OUTSIDE_DECODER: inference.TranslatorOutsideDecoder
        }
        selected_translator = translator_selection.get(args.method, inference.Translator)
        translator = selected_translator(context=context,
                                         ensemble_mode=args.ensemble_mode,
                                         bucket_source_width=args.bucket_width,
                                         bucket_target_width=args.bucket_width,
                                         length_penalty=inference.LengthPenalty(args.length_penalty_alpha,
                                                                                args.length_penalty_beta),
                                         beam_prune=args.beam_prune,
                                         beam_search_stop=args.beam_search_stop,
                                         nbest_size=args.nbest_size,
                                         models=models,
                                         source_vocabs=source_vocabs,
                                         target_vocab=target_vocab,
                                         restrict_lexicon=restrict_lexicon,
                                         avoid_list=args.avoid_list,
                                         store_beam=store_beam,
                                         strip_unknown_words=args.strip_unknown_words,
                                         skip_topk=args.skip_topk,
                                         sample=args.sample)
        read_and_translate(translator=translator,
                           output_handler=output_handler,
                           chunk_size=args.chunk_size,
                           input_file=args.input,
                           input_factors=args.input_factors,
                           input_is_json=args.json_input,
                           input_source_doc=args.input_source_doc,
                           input_target_doc=args.input_target_doc)


def make_inputs(input_file: Optional[str],
                translator: inference.Translator,
                input_is_json: bool,
                input_factors: Optional[List[str]] = None) -> Generator[inference.TranslatorInput, None, None]:
    """
    Generates TranslatorInput instances from input. If input is None, reads from stdin. If num_input_factors > 1,
    the function will look for factors attached to each token, separated by '|'.
    If source is not None, reads from the source file. If num_source_factors > 1, num_source_factors source factor
    filenames are required.

    :param input_file: The source file (possibly None).
    :param translator: Translator that will translate each line of input.
    :param input_is_json: Whether the input is in json format.
    :param input_factors: Source factor files.
    :return: TranslatorInput objects.
    """
    if input_file is None:
        check_condition(input_factors is None, "Translating from STDIN, not expecting any factor files.")
        for sentence_id, line in enumerate(sys.stdin, 1):
            if input_is_json:
                yield inference.make_input_from_json_string(sentence_id=sentence_id, json_string=line)
            else:
                yield inference.make_input_from_factored_string(sentence_id=sentence_id,
                                                                factored_string=line,
                                                                translator=translator)
    else:
        input_factors = [] if input_factors is None else input_factors
        inputs = [input_file] + input_factors
        if not input_is_json:
            check_condition(translator.num_source_factors == len(inputs),
                            "Model(s) require %d factors, but %d given (through --input and --input-factors)." % (
                                translator.num_source_factors, len(inputs)))
        with ExitStack() as exit_stack:
            streams = [exit_stack.enter_context(data_io.smart_open(i)) for i in inputs]
            for sentence_id, inputs in enumerate(zip(*streams), 1):
                if input_is_json:
                    yield inference.make_input_from_json_string(sentence_id=sentence_id, json_string=inputs[0])
                else:
                    yield inference.make_input_from_multiple_strings(sentence_id=sentence_id, strings=list(inputs))


def make_inputs_doc(input_file: Optional[str],
                    translator: Union[inference.TranslatorOutsideDecoder],
                    input_is_json: bool,
                    input_factors: Optional[List[str]] = None,
                    input_source_doc: Optional[str] = None,
                    input_target_doc: Optional[str] = None) -> Generator[inference.TranslatorInputDoc, None, None]:
    """
    Generates TranslatorInputDoc instances from input and the additional context sentence inputs.
    If input is None, reads from stdin. If num_input_factors > 1,
    the function will look for factors attached to each token, separated by '|'.
    If source is not None, reads from the source file. If num_source_factors > 1, num_source_factors source factor
    filenames are required.

    :param input_file: The source file (possibly None).
    :param translator: Translator that will translate each line of input.
    :param input_is_json: Whether the input is in json format.
    :param input_factors: Source factor files.
    :param input_source_doc: Source context file.
    :param input_target_doc: Target context file.
    :return: TranslatorInputDoc objects.
    """
    check_condition(input_source_doc is not None or input_target_doc is not None,
                    "Missing additional context data.")

    doc_context_config = translator.models[0].config.doc_context_config
    source_pre_sentences, source_nxt_sentences, target_pre_sentences, target_nxt_sentences = \
        read_doc_sequences(input_source_doc,
                           input_target_doc,
                           doc_context_config.window_config)
    if input_file is None:
        check_condition(input_factors is None, "Translating from STDIN, not expecting any factor files.")
        for sentence_id, (src_pre, line, src_nxt, tar_pre, tar_nxt) in \
                enumerate(iter_doc_level(sys.stdin, source_pre_sentences, source_nxt_sentences,
                                         target_pre_sentences, target_nxt_sentences), 1):
            if input_is_json:
                raise NotImplementedError("Do not use JSON! - not supported for document-level context reading")
            else:
                yield inference.make_input_from_factored_string_doc(sentence_id=sentence_id,
                                                                    factored_string=line,
                                                                    src_pre=src_pre,
                                                                    src_nxt=src_nxt,
                                                                    tar_pre=tar_pre,
                                                                    tar_nxt=tar_nxt,
                                                                    translator=translator)
    else:
        check_condition(input_factors is None, "We do not support input factors.")
        with ExitStack() as exit_stack:
            stream = exit_stack.enter_context(data_io.smart_open(input_file))
            for sentence_id, (src_pre, line, src_nxt, tar_pre, tar_nxt) in \
                    enumerate(iter_doc_level(stream, source_pre_sentences, source_nxt_sentences,
                                             target_pre_sentences, target_nxt_sentences), 1):
                if input_is_json:
                    raise NotImplementedError("Do not use JSON! - not supported for document-level context reading")
                else:
                    yield inference.make_input_from_multiple_strings_doc(sentence_id=sentence_id,
                                                                         strings=line,
                                                                         src_pre=src_pre,
                                                                         src_nxt=src_nxt,
                                                                         tar_pre=tar_pre,
                                                                         tar_nxt=tar_nxt)


def iter_doc_level(source_iterable: Iterable,
                   source_pre_iterables: List[Iterable],
                   source_nxt_iterables: List[Iterable],
                   target_pre_iterables: List[Iterable],
                   target_nxt_iterables: List[Iterable]):
    """
    Iterate through current source and additional context data.

    :param source_iterable:
    :param source_pre_iterables:
    :param source_nxt_iterables:
    :param target_pre_iterables:
    :param target_nxt_iterables:
    :return:
    """
    source_iterator = iter(source_iterable)
    source_pre_iterators = [iter(source_pre_iterable) for source_pre_iterable in source_pre_iterables]
    source_nxt_iterators = [iter(source_nxt_iterable) for source_nxt_iterable in source_nxt_iterables]
    target_pre_iterators = [iter(target_pre_iterable) for target_pre_iterable in target_pre_iterables]
    target_nxt_iterators = [iter(target_nxt_iterable) for target_nxt_iterable in target_nxt_iterables]
    return iterate_doc_level(source_iterator,
                             source_pre_iterators, source_nxt_iterators,
                             target_pre_iterators, target_nxt_iterators)


def iterate_doc_level(source_iterators: Iterator,
                      source_pre_iterators: Sequence[Iterator],
                      source_nxt_iterators: Sequence[Iterator],
                      target_pre_iterators: Sequence[Iterator],
                      target_nxt_iterators: Sequence[Iterator]):
    """
    Generator that yields current source sentences and context sentences.

    :param source_iterators: Current source sentences.
    :param source_pre_iterators: Previous source sentences.
    :param source_nxt_iterators: Next source sentences.
    :param target_pre_iterators: Previous target sentences.
    :param target_nxt_iterators: Next target sentences.
    :return: Current source string sequence including context ones.
    """
    while True:
        try:
            sources = next(source_iterators)
            source_pre = [next(source_pre_iter) for source_pre_iter in source_pre_iterators]
            source_nxt = [next(source_nxt_iter) for source_nxt_iter in source_nxt_iterators]
            target_pre = [next(target_pre_iter) for target_pre_iter in target_pre_iterators]
            target_nxt = [next(target_nxt_iter) for target_nxt_iter in target_nxt_iterators]
        except StopIteration:
            break
        yield source_pre, sources, source_nxt, target_pre, target_nxt

    check_condition(next(cast(Iterator, source_iterators), None) is None,
                    "Different number of lines in the source original data")

    check_condition(
            all(next(cast(Iterator, src_pre), None) is None for src_pre in source_pre_iterators)
            and all(next(cast(Iterator, src_nxt), None) is None for src_nxt in source_nxt_iterators)
            and all(next(cast(Iterator, tar_pre), None) is None for tar_pre in target_pre_iterators)
            and all(next(cast(Iterator, tar_nxt), None) is None for tar_nxt in target_nxt_iterators),
            "Different number of lines in the additional data")


class SequencesDocLevel(Iterable):
    """
    Reads context sequence samples from path with respective shifts.

    :param path: Path to read data from.
    :param pre_shift: Shift for previous sentences.
    :param nxt_shift: Shift for next sentences.
    """

    def __init__(self,
                 path: str,
                 pre_shift: Optional[int] = None,
                 nxt_shift: Optional[int] = None) -> None:
        self.path = path
        self.pre_shift = pre_shift
        self.nxt_shift = nxt_shift

    def __iter__(self):
        for tokens in data_io.read_content_doc(self.path,
                                               unk=C.UNK_SYMBOL,
                                               pre_shift=self.pre_shift,
                                               nxt_shift=self.nxt_shift):
            if len(tokens) == 0:
                yield None
                continue
            yield tokens


def read_doc_sequences(input_source_doc: Optional[str],
                       input_target_doc: Optional[str],
                       window_config: doc_context.WindowConfig) \
        -> Tuple[List[SequencesDocLevel],
                 List[SequencesDocLevel],
                 List[SequencesDocLevel],
                 List[SequencesDocLevel]]:
    source_pre_seq = [SequencesDocLevel(input_source_doc,
                                        pre_shift=i + 1)
                      for i in range(window_config.src_pre) if input_source_doc is not None]

    source_nxt_seq = [SequencesDocLevel(input_source_doc,
                                        nxt_shift=j + 1)
                      for j in range(window_config.src_nxt) if input_source_doc is not None]

    target_pre_seq = [SequencesDocLevel(input_target_doc,
                                        pre_shift=k + 1)
                      for k in range(window_config.tar_pre) if input_target_doc is not None]

    target_nxt_seq = [SequencesDocLevel(input_target_doc,
                                        nxt_shift=l + 1)
                      for l in range(window_config.tar_nxt) if input_target_doc is not None]
    return source_pre_seq, source_nxt_seq, target_pre_seq, target_nxt_seq


def read_and_translate(translator: inference.Translator,
                       output_handler: OutputHandler,
                       chunk_size: Optional[int],
                       input_file: Optional[str] = None,
                       input_factors: Optional[List[str]] = None,
                       input_is_json: bool = False,
                       input_source_doc: Optional[str] = None,
                       input_target_doc: Optional[str] = None) -> None:
    """
    Reads from either a file or stdin and translates each line, calling the output_handler with the result.

    :param output_handler: Handler that will write output to a stream.
    :param translator: Translator that will translate each line of input.
    :param chunk_size: The size of the portion to read at a time from the input.
    :param input_file: Optional path to file which will be translated line-by-line if included, if none use stdin.
    :param input_factors: Optional list of paths to files that contain source factors.
    :param input_is_json: Whether the input is in json format.
    :param input_source_doc: Optional source input for context sentences
    :param input_target_doc: Optional target input for context sentences
    """
    batch_size = translator.max_batch_size
    if chunk_size is None:
        if translator.max_batch_size == 1:
            # No batching, therefore there is not need to read segments in chunks.
            chunk_size = C.CHUNK_SIZE_NO_BATCHING
        else:
            # Get a constant number of batches per call to Translator.translate.
            chunk_size = C.CHUNK_SIZE_PER_BATCH_SEGMENT * translator.max_batch_size
    else:
        if chunk_size < translator.max_batch_size:
            logger.warning("You specified a chunk size (%d) smaller than the max batch size (%d). This will lead to "
                           "a reduction in translation speed. Consider choosing a larger chunk size." % (chunk_size,
                                                                                                         batch_size))
    logger.info("Translating...")

    total_time, total_lines = 0.0, 0

    if isinstance(translator, inference.Translator):
        for chunk in grouper(make_inputs(input_file, translator, input_is_json, input_factors), size=chunk_size):
            chunk_time = translate(output_handler, chunk, translator)
            total_lines += len(chunk)
            total_time += chunk_time
    elif isinstance(translator, inference.TranslatorOutsideDecoder):
        for chunk in grouper(make_inputs_doc(input_file, translator, input_is_json, input_factors,
                                             input_source_doc, input_target_doc), size=chunk_size):
            chunk_time = translate(output_handler, chunk, translator)
            total_lines += len(chunk)
            total_time += chunk_time

    if total_lines != 0:
        logger.info("Processed %d lines. Total time: %.4f, sec/sent: %.4f, sent/sec: %.4f",
                    total_lines, total_time, total_time / total_lines, total_lines / total_time)
    else:
        logger.info("Processed 0 lines.")


def translate(output_handler: OutputHandler,
              trans_inputs: Union[List[inference.TranslatorInput],
                                  List[inference.TranslatorInputDoc]],
              translator: Union[inference.Translator, inference.TranslatorOutsideDecoder]) -> float:
    """
    Translates each line from source_data, calling output handler after translating a batch.

    :param output_handler: A handler that will be called once with the output of each translation.
    :param trans_inputs: A enumerable list of translator inputs.
    :param translator: The translator that will be used for each line of input.
    :return: Total time taken.
    """
    tic = time.time()
    trans_outputs = translator.translate(trans_inputs)
    total_time = time.time() - tic
    batch_time = total_time / len(trans_inputs)
    for trans_input, trans_output in zip(trans_inputs, trans_outputs):
        output_handler.handle(trans_input, trans_output, batch_time)
    return total_time


if __name__ == '__main__':
    main()
