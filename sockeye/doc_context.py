from typing import List, Optional, Tuple

from . import config
from . import layers
from .transformer import TransformerProcessBlock, get_variable_length_bias_extended

import mxnet as mx


# Architecture choices
OUTSIDE_DECODER = "outside-decoder"
INSIDE_DECODER_PARALLEL = "inside-decoder-parallel-attention"
INSIDE_DECODER_SEQUENTIAL = "inside-decoder-sequential-attention"
ARCHITECTURE_CHOICES = [OUTSIDE_DECODER, INSIDE_DECODER_PARALLEL, INSIDE_DECODER_SEQUENTIAL]


# I/O variable names; for internal computation graph
SOURCE_PRE_NAME = "src_pre_%d"
SOURCE_PRE_LENGTH_NAME = "src_pre_length"
SOURCE_NXT_NAME = "src_nxt_%d"
SOURCE_NXT_LENGTH_NAME = "src_nxt_length"
TARGET_PRE_NAME = "tar_pre_%d"
TARGET_PRE_LENGTH_NAME = "tar_pre_length"
TARGET_NXT_NAME = "tar_nxt_%d"
TARGET_NXT_LENGTH_NAME = "tar_nxt_length"


# Prefixes for tensors; for internal computation graph
EMBEDDING_PREFIX = "embed_"
SOURCE_DOC_EMBEDDING_PREFIX = "source_doc" + EMBEDDING_PREFIX
TARGET_DOC_EMBEDDING_PREFIX = "target_doc" + EMBEDDING_PREFIX
SOURCE_PRE_ENC_PREFIX = "source_doc_enc_pre_"
SOURCE_NXT_ENC_PREFIX = "source_doc_enc_nxt_"
TARGET_PRE_ENC_PREFIX = "target_doc_enc_pre_"
TARGET_NXT_ENC_PREFIX = "target_doc_enc_nxt_"


# Prefixes for neural modules
ENCODER_PREFIX = "doc_context_enc_"


# Names for gating mechanism
GATE_LINEAR = "gate_linear"
GATE_ACT = "gate_act"
GATE_MAPPED_INPUT_1 = "gate_mapped_input_1"
GATE_MAPPED_INPUT_2 = "gate_mapped_input_2"
GATE_MAPPED_INPUT_PREFIX = "gate_mapped_input_{}_{}"


# Names integration mechanisms
DOC_LENGTH_BIAS = "doc_length_bias"
CROSS_ATTENTION_PREFIX = "cross_attention_"


class WindowConfig(config.Config):
    def __init__(self,
                 src_pre: int,
                 src_nxt: int,
                 tar_pre: int,
                 tar_nxt: int):
        super().__init__()
        self.src_pre = src_pre
        self.src_nxt = src_nxt
        self.tar_pre = tar_pre
        self.tar_nxt = tar_nxt

    @property
    def sizes(self) -> Tuple[int, int, int, int]:
        return self.src_pre, self.src_nxt, self.tar_pre, self.tar_nxt

    @property
    def number_source_side(self):
        return self.src_pre + self.src_nxt

    @property
    def number_target_side(self):
        return self.tar_pre + self.tar_nxt

    @property
    def use_source_side(self):
        return self.number_source_side > 0

    @property
    def use_target_side(self):
        return self.number_target_side > 0

    @property
    def source_side_only(self):
        return self.use_source_side and not self.use_target_side

    @property
    def target_side_only(self):
        return not self.use_source_side and self.use_target_side

    @property
    def use_both_sides(self):
        return self.use_source_side and self.use_target_side

    @property
    def use_exactly_one_side(self):
        return self.is_used and not self.use_both_sides

    @property
    def is_used(self):
        return self.use_source_side or self.use_target_side


class DocumentContextConfig(config.Config):
    def __init__(self,
                 method: str,
                 window_config: WindowConfig,
                 source_train: Optional[str],
                 source_validation: Optional[str],
                 target_train: Optional[str],
                 target_validation: Optional[str],
                 bucket_width: int):
        super().__init__()
        self.method = method
        self.window_config = window_config
        self.source_train = source_train
        self.target_train = target_train
        self.source_validation = source_validation
        self.target_validation = target_validation
        self.bucket_width = bucket_width


class GatingMechanism:
    """
    Interpolates two values a,b with a computed gate-weight value g: g*a + (1-g)*b
    """

    def __init__(self, prefix: str, output_dim: int):
        self.prefix = prefix
        self.output_dim = output_dim

        self.gate_w = mx.sym.Variable(self.prefix + "gate_weight")
        self.gate_b = mx.sym.Variable(self.prefix + "gate_bias")

        self.mapped_input_1_w = mx.sym.Variable(self.prefix + GATE_MAPPED_INPUT_PREFIX.format(1, "weight"))
        self.mapped_input_1_b = mx.sym.Variable(self.prefix + GATE_MAPPED_INPUT_PREFIX.format(1, "bias"))

        self.mapped_input_2_w = mx.sym.Variable(self.prefix + GATE_MAPPED_INPUT_PREFIX.format(2, "weight"))
        self.mapped_input_2_b = mx.sym.Variable(self.prefix + GATE_MAPPED_INPUT_PREFIX.format(2, "bias"))

    def __call__(self,
                 input_one: mx.sym.Symbol,
                 input_two: mx.sym.Symbol) -> mx.sym.Symbol:
        """

        :param input_one: Shape=(batch_size, length, feature_dimensions1)
        :param input_two: Shape=(batch_size, length, feature_dimensions2)
        :return: Shape=(batch_size, length, output_dimensions)
        """
        gate_input = mx.sym.concat(input_one, input_two, dim=2)

        gate = mx.sym.FullyConnected(data=gate_input,
                                     weight=self.gate_w,
                                     bias=self.gate_b,
                                     num_hidden=self.output_dim,
                                     flatten=False,
                                     name="%s%s" % (self.prefix, GATE_LINEAR))
        gate = mx.sym.Activation(data=gate, act_type="sigmoid",
                                 name="%s%s" % (self.prefix, GATE_ACT))
        mapped_input_one = mx.sym.FullyConnected(data=input_one,
                                                 weight=self.mapped_input_1_w,
                                                 bias=self.mapped_input_1_b,
                                                 num_hidden=self.output_dim,
                                                 flatten=False,
                                                 name="%s%s" % (self.prefix, GATE_MAPPED_INPUT_1))
        mapped_input_two = mx.sym.FullyConnected(data=input_two,
                                                 weight=self.mapped_input_2_w,
                                                 bias=self.mapped_input_2_b,
                                                 num_hidden=self.output_dim,
                                                 flatten=False,
                                                 name="%s%s" % (self.prefix, GATE_MAPPED_INPUT_2))
        return gate * mapped_input_one + (1 - gate) * mapped_input_two


class OutsideDecoderCombination:

    def __init__(self, prefix: str, attention_depth: int, attention_heads: int, output_dim: int):
        self.prefix = prefix
        self.attention_heads = attention_heads
        self.gating_mechanism = GatingMechanism(prefix=self.prefix, output_dim=output_dim)
        self.cross_attention = layers.MultiHeadAttention(prefix=self.prefix + CROSS_ATTENTION_PREFIX,
                                                         depth_att=attention_depth,
                                                         heads=attention_heads,
                                                         depth_out=output_dim)

    def __call__(self,
                 source_encoded: mx.sym.Symbol,
                 doc_context_encoded: mx.sym.Symbol,
                 doc_context_encoded_length: mx.sym.Symbol,
                 doc_context_encoded_seq_len: List[int]):
        # (batch_size * heads, max_length)
        doc_bias = get_variable_length_bias_extended(lengths=doc_context_encoded_length,
                                                     max_length=doc_context_encoded_seq_len,
                                                     num_heads=self.attention_heads,
                                                     fold_heads=True,
                                                     name=DOC_LENGTH_BIAS)
        # (batch_size * heads, 1, max_length)
        doc_bias = mx.sym.expand_dims(doc_bias, axis=1)
        attended_to_context = self.cross_attention(queries=source_encoded,
                                                   memory=doc_context_encoded,
                                                   bias=doc_bias)
        return self.gating_mechanism(input_one=source_encoded,
                                     input_two=attended_to_context)
