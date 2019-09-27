# Copyright 2017 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
import ast
from typing import Dict, List, Optional, TYPE_CHECKING, Union

import mxnet as mx
import numpy as np

from . import config
from . import constants as C
from . import doc_context
from . import layers

if TYPE_CHECKING:
    from . import encoder


class GatingMechanism:
    """
    Interpolates two values a,b with a computed gate-weight value g: g*a + (1-g)*b
    """

    def __init__(self, prefix: str, output_dim: int):
        self.prefix = prefix
        self.output_dim = output_dim

        self.gate_w = mx.sym.Variable(self.prefix + "gate_weight")
        self.gate_b = mx.sym.Variable(self.prefix + "gate_bias")

        self.mapped_input_1_w = mx.sym.Variable(self.prefix + doc_context.GATE_MAPPED_INPUT_PREFIX.format(1, "weight"))
        self.mapped_input_1_b = mx.sym.Variable(self.prefix + doc_context.GATE_MAPPED_INPUT_PREFIX.format(1, "bias"))

        self.mapped_input_2_w = mx.sym.Variable(self.prefix + doc_context.GATE_MAPPED_INPUT_PREFIX.format(2, "weight"))
        self.mapped_input_2_b = mx.sym.Variable(self.prefix + doc_context.GATE_MAPPED_INPUT_PREFIX.format(2, "bias"))

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
                                     name="%s%s" % (self.prefix, doc_context.GATE_LINEAR))
        gate = mx.sym.Activation(data=gate, act_type="sigmoid",
                                 name="%s%s" % (self.prefix, doc_context.GATE_ACT))
        mapped_input_one = mx.sym.FullyConnected(data=input_one,
                                                 weight=self.mapped_input_1_w,
                                                 bias=self.mapped_input_1_b,
                                                 num_hidden=self.output_dim,
                                                 flatten=False,
                                                 name="%s%s" % (self.prefix, doc_context.GATE_MAPPED_INPUT_1))
        mapped_input_two = mx.sym.FullyConnected(data=input_two,
                                                 weight=self.mapped_input_2_w,
                                                 bias=self.mapped_input_2_b,
                                                 num_hidden=self.output_dim,
                                                 flatten=False,
                                                 name="%s%s" % (self.prefix, doc_context.GATE_MAPPED_INPUT_2))
        return gate * mapped_input_one + (1 - gate) * mapped_input_two


class OutsideDecoderCombination:
    # TODO: pre and post

    def __init__(self, prefix: str, attention_depth: int, attention_heads: int, output_dim: int):
        self.prefix = prefix
        self.attention_heads = attention_heads
        self.gating_mechanism = GatingMechanism(prefix=self.prefix, output_dim=output_dim)
        self.cross_attention = layers.MultiHeadAttention(prefix=self.prefix + doc_context.CROSS_ATTENTION_PREFIX,
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
                                                     name=doc_context.DOC_LENGTH_BIAS)
        # (batch_size * heads, 1, max_length)
        doc_bias = mx.sym.expand_dims(doc_bias, axis=1)
        attended_to_context = self.cross_attention(queries=source_encoded,
                                                   memory=doc_context_encoded,
                                                   bias=doc_bias)
        return self.gating_mechanism(input_one=source_encoded,
                                     input_two=attended_to_context)


class TransformerConfig(config.Config):

    def __init__(self,
                 model_size: int,
                 attention_heads: int,
                 feed_forward_num_hidden: int,
                 act_type: str,
                 num_layers: int,
                 dropout_attention: float,
                 dropout_act: float,
                 dropout_prepost: float,
                 positional_embedding_type: str,
                 preprocess_sequence: str,
                 postprocess_sequence: str,
                 max_seq_len_source: int,
                 max_seq_len_target: int,
                 conv_config: Optional['encoder.ConvolutionalEmbeddingConfig'] = None,
                 lhuc: bool = False,
                 dtype: str = C.DTYPE_FP32) -> None:  # type: ignore
        super().__init__()
        self.model_size = model_size
        self.attention_heads = attention_heads
        self.feed_forward_num_hidden = feed_forward_num_hidden
        self.act_type = act_type
        self.num_layers = num_layers
        self.dropout_attention = dropout_attention
        self.dropout_act = dropout_act
        self.dropout_prepost = dropout_prepost
        self.positional_embedding_type = positional_embedding_type
        self.preprocess_sequence = preprocess_sequence
        self.postprocess_sequence = postprocess_sequence
        self.max_seq_len_source = max_seq_len_source
        self.max_seq_len_target = max_seq_len_target
        self.conv_config = conv_config
        self.use_lhuc = lhuc
        self.dtype = dtype


class TransformerConfigInsideDecoder(TransformerConfig):

    def __init__(self,
                 use_parallel_attention: bool,
                 model_size: int,
                 attention_heads: int,
                 attention_heads_doc: int,
                 feed_forward_num_hidden: int,
                 act_type: str,
                 num_layers: int,
                 dropout_attention: float,
                 dropout_attention_doc: float,
                 dropout_act: float,
                 dropout_act_doc: float,
                 dropout_prepost: float,
                 dropout_prepost_doc: float,
                 positional_embedding_type: str,
                 preprocess_sequence: str,
                 preprocess_sequence_doc: str,
                 postprocess_sequence: str,
                 postprocess_sequence_doc: str,
                 max_seq_len_source: int,
                 max_seq_len_target: int,
                 conv_config: Optional['encoder.ConvolutionalEmbeddingConfig'] = None,
                 lhuc: bool = False,
                 dtype: str = C.DTYPE_FP32) -> None:  # type: ignore
        super().__init__(model_size=model_size, attention_heads=attention_heads,
                         feed_forward_num_hidden=feed_forward_num_hidden,
                         act_type=act_type, num_layers=num_layers, dropout_attention=dropout_attention,
                         dropout_act=dropout_act, dropout_prepost=dropout_prepost,
                         positional_embedding_type=positional_embedding_type, preprocess_sequence=preprocess_sequence,
                         postprocess_sequence=postprocess_sequence, max_seq_len_source=max_seq_len_source,
                         max_seq_len_target=max_seq_len_target, conv_config=conv_config, lhuc=lhuc, dtype=dtype)
        self.use_parallel_attention = use_parallel_attention
        self.attention_heads_doc = attention_heads_doc
        self.dropout_attention_doc = dropout_attention_doc
        self.dropout_act_doc = dropout_act_doc
        self.dropout_prepost_doc = dropout_prepost_doc
        self.preprocess_sequence_doc = preprocess_sequence_doc
        self.postprocess_sequence_doc = postprocess_sequence_doc


class TransformerConfigWithDocEncoders(config.Config):

    def __init__(self,
                 transformer_config: TransformerConfig,
                 doc_config: TransformerConfig,
                 number_doc_sentences: int) -> None:
        super().__init__()
        self.transformer_config = transformer_config
        self.doc_config = doc_config
        self.number_doc_sentences = number_doc_sentences


class TransformerEncoderBlock:
    """
    A transformer encoder block consists self-attention and a feed-forward layer with pre/post process blocks
    in between.
    """

    def __init__(self,
                 config: TransformerConfig,
                 prefix: str) -> None:
        self.pre_self_attention = TransformerProcessBlock(sequence=config.preprocess_sequence,
                                                          dropout=config.dropout_prepost,
                                                          prefix="%satt_self_pre_" % prefix)
        self.self_attention = layers.MultiHeadSelfAttention(depth_att=config.model_size,
                                                            heads=config.attention_heads,
                                                            depth_out=config.model_size,
                                                            dropout=config.dropout_attention,
                                                            prefix="%satt_self_" % prefix)
        self.post_self_attention = TransformerProcessBlock(sequence=config.postprocess_sequence,
                                                           dropout=config.dropout_prepost,
                                                           prefix="%satt_self_post_" % prefix)

        self.pre_ff = TransformerProcessBlock(sequence=config.preprocess_sequence,
                                              dropout=config.dropout_prepost,
                                              prefix="%sff_pre_" % prefix)
        self.ff = TransformerFeedForward(num_hidden=config.feed_forward_num_hidden,
                                         num_model=config.model_size,
                                         act_type=config.act_type,
                                         dropout=config.dropout_act,
                                         prefix="%sff_" % prefix)
        self.post_ff = TransformerProcessBlock(sequence=config.postprocess_sequence,
                                               dropout=config.dropout_prepost,
                                               prefix="%sff_post_" % prefix)
        self.lhuc = None
        if config.use_lhuc:
            self.lhuc = layers.LHUC(config.model_size, prefix=prefix)

    def __call__(self, data: mx.sym.Symbol, bias: mx.sym.Symbol) -> mx.sym.Symbol:
        # self-attention
        data_self_att = self.self_attention(inputs=self.pre_self_attention(data, None),
                                            bias=bias,
                                            cache=None)
        data = self.post_self_attention(data_self_att, data)

        # feed-forward
        data_ff = self.ff(self.pre_ff(data, None))
        data = self.post_ff(data_ff, data)

        if self.lhuc:
            data = self.lhuc(data)

        return data


class TransformerDecoderBlock:
    """
    A transformer encoder block consists self-attention, encoder attention, and a feed-forward layer
    with pre/post process blocks in between.
    """

    def __init__(self,
                 config: TransformerConfig,
                 prefix: str) -> None:
        self.prefix = prefix
        self.pre_self_attention = TransformerProcessBlock(sequence=config.preprocess_sequence,
                                                          dropout=config.dropout_prepost,
                                                          prefix="%satt_self_pre_" % prefix)
        self.self_attention = layers.MultiHeadSelfAttention(depth_att=config.model_size,
                                                            heads=config.attention_heads,
                                                            depth_out=config.model_size,
                                                            dropout=config.dropout_attention,
                                                            prefix="%satt_self_" % prefix)
        self.post_self_attention = TransformerProcessBlock(sequence=config.postprocess_sequence,
                                                           dropout=config.dropout_prepost,
                                                           prefix="%satt_self_post_" % prefix)

        self.pre_enc_attention = TransformerProcessBlock(sequence=config.preprocess_sequence,
                                                         dropout=config.dropout_prepost,
                                                         prefix="%satt_enc_pre_" % prefix)
        self.enc_attention = layers.MultiHeadAttention(depth_att=config.model_size,
                                                       heads=config.attention_heads,
                                                       depth_out=config.model_size,
                                                       dropout=config.dropout_attention,
                                                       prefix="%satt_enc_" % prefix)
        self.post_enc_attention = TransformerProcessBlock(sequence=config.postprocess_sequence,
                                                          dropout=config.dropout_prepost,
                                                          prefix="%satt_enc_post_" % prefix)

        self.pre_ff = TransformerProcessBlock(sequence=config.preprocess_sequence,
                                              dropout=config.dropout_prepost,
                                              prefix="%sff_pre_" % prefix)
        self.ff = TransformerFeedForward(num_hidden=config.feed_forward_num_hidden,
                                         num_model=config.model_size,
                                         act_type=config.act_type,
                                         dropout=config.dropout_act,
                                         prefix="%sff_" % prefix)
        self.post_ff = TransformerProcessBlock(sequence=config.postprocess_sequence,
                                               dropout=config.dropout_prepost,
                                               prefix="%sff_post_" % prefix)

        self.lhuc = None
        if config.use_lhuc:
            self.lhuc = layers.LHUC(config.model_size, prefix=prefix)

    def __call__(self,
                 target: mx.sym.Symbol,
                 target_bias: mx.sym.Symbol,
                 source: mx.sym.Symbol,
                 source_bias: mx.sym.Symbol,
                 cache: Optional[Dict[str, Optional[mx.sym.Symbol]]] = None) -> mx.sym.Symbol:
        # self-attention
        target_self_att = self.self_attention(inputs=self.pre_self_attention(target, None),
                                              bias=target_bias,
                                              cache=cache)
        target = self.post_self_attention(target_self_att, target)

        # encoder attention
        target_enc_att = self.enc_attention(queries=self.pre_enc_attention(target, None),
                                            memory=source,
                                            bias=source_bias)
        target = self.post_enc_attention(target_enc_att, target)

        # feed-forward
        target_ff = self.ff(self.pre_ff(target, None))
        target = self.post_ff(target_ff, target)

        if self.lhuc:
            target = self.lhuc(target)

        return target


class TransformerDecoderBlockInsideContext:
    """
    In addition to self-attention, encoder attention, and a feed-forward sublayer we use a seperate
    attention layer for context information. Both attention outputs are then interpolated linearly before
    forwarding to the feed-forward sublayer.

    For this matter, there are two options of signal construction. One is using the second attention sequentially:
        1. self-attention -> encoder-decoder-attention a1 -> context-attention a2
        2. compute linear interpolation: L := g * a2 + (1-g) * a1
        3. forward L to feed-forward sublayer

    or in parallel:
        1. self-attention -> encoder-decoder-attention a1
                          -> context-attention a2
        2. compute linear interpolation: L := g * a2 + (1-g) * a1
        3. forward L to feed-forward sublayer
    """

    def __init__(self,
                 config: TransformerConfigInsideDecoder,
                 prefix: str) -> None:
        self.prefix = prefix
        self.use_parallel_attention = config.use_parallel_attention
        self.pre_self_attention = TransformerProcessBlock(sequence=config.preprocess_sequence,
                                                          dropout=config.dropout_prepost,
                                                          prefix="%satt_self_pre_" % prefix)
        self.self_attention = layers.MultiHeadSelfAttention(depth_att=config.model_size,
                                                            heads=config.attention_heads,
                                                            depth_out=config.model_size,
                                                            dropout=config.dropout_attention,
                                                            prefix="%satt_self_" % prefix)
        self.post_self_attention = TransformerProcessBlock(sequence=config.postprocess_sequence,
                                                           dropout=config.dropout_prepost,
                                                           prefix="%satt_self_post_" % prefix)

        self.pre_enc_attention = TransformerProcessBlock(sequence=config.preprocess_sequence,
                                                         dropout=config.dropout_prepost,
                                                         prefix="%satt_enc_pre_" % prefix)
        self.enc_attention = layers.MultiHeadAttention(depth_att=config.model_size,
                                                       heads=config.attention_heads,
                                                       depth_out=config.model_size,
                                                       dropout=config.dropout_attention,
                                                       prefix="%satt_enc_" % prefix)
        self.post_enc_attention = TransformerProcessBlock(sequence=config.postprocess_sequence,
                                                          dropout=config.dropout_prepost,
                                                          prefix="%satt_enc_post_" % prefix)


        self.pre_attention_doc = TransformerProcessBlock(sequence=config.preprocess_sequence_doc,
                                                         dropout=config.dropout_prepost_doc,
                                                         prefix="%satt_pre_doc_" % prefix)
        self.attention_doc = layers.MultiHeadAttention(depth_att=config.model_size,
                                                       heads=config.attention_heads_doc,
                                                       depth_out=config.model_size,
                                                       dropout=config.dropout_attention_doc,
                                                       prefix="%satt_doc_" % prefix)
        self.post_attention_doc = TransformerProcessBlock(sequence=config.postprocess_sequence_doc,
                                                          dropout=config.dropout_prepost_doc,
                                                          prefix="%satt_post_doc_" % prefix)
        self.gating = GatingMechanism(prefix="%sgating_" % prefix,
                                      output_dim=config.model_size)

        self.pre_ff = TransformerProcessBlock(sequence=config.preprocess_sequence,
                                              dropout=config.dropout_prepost,
                                              prefix="%sff_pre_" % prefix)
        self.ff = TransformerFeedForward(num_hidden=config.feed_forward_num_hidden,
                                         num_model=config.model_size,
                                         act_type=config.act_type,
                                         dropout=config.dropout_act,
                                         prefix="%sff_" % prefix)
        self.post_ff = TransformerProcessBlock(sequence=config.postprocess_sequence,
                                               dropout=config.dropout_prepost,
                                               prefix="%sff_post_" % prefix)

        self.lhuc = None
        if config.use_lhuc:
            self.lhuc = layers.LHUC(config.model_size, prefix=prefix)

    def __call__(self,
                 target: mx.sym.Symbol,
                 target_bias: mx.sym.Symbol,
                 source: mx.sym.Symbol,
                 source_bias: mx.sym.Symbol,
                 doc: mx.sym.Symbol,
                 doc_bias: mx.sym.Symbol,
                 cache: Optional[Dict[str, Optional[mx.sym.Symbol]]] = None) -> mx.sym.Symbol:
        # self-attention
        target_self_att = self.self_attention(inputs=self.pre_self_attention(target, None),
                                              bias=target_bias,
                                              cache=cache)
        target = self.post_self_attention(target_self_att, target)

        # encoder attention
        target_enc_att = self.enc_attention(queries=self.pre_enc_attention(target, None),
                                            memory=source,
                                            bias=source_bias)
        encoder_decoder_attention_out = self.post_enc_attention(target_enc_att, target)

        # context attention
        doc_context_input = target if self.use_parallel_attention else encoder_decoder_attention_out

        doc_context_attention = self.attention_doc(
                queries=self.pre_attention_doc(doc_context_input, None),
                memory=doc,
                bias=doc_bias
        )

        doc_context_attention_out = self.post_attention_doc(doc_context_attention, None)

        target = self.gating(doc_context_attention_out, encoder_decoder_attention_out)

        # feed-forward
        target_ff = self.ff(self.pre_ff(target, None))
        target = self.post_ff(target_ff, target)

        if self.lhuc:
            target = self.lhuc(target)

        return target


class TransformerProcessBlock:
    """
    Block to perform pre/post processing on layer inputs.
    The processing steps are determined by the sequence argument, which can contain one of the three operations:
    n: layer normalization
    r: residual connection
    d: dropout
    """

    def __init__(self,
                 sequence: str,
                 dropout: float,
                 prefix: str) -> None:
        self.sequence = sequence
        self.dropout = dropout
        self.prefix = prefix
        self.layer_norm = None
        if "n" in sequence:
            self.layer_norm = layers.LayerNormalization(prefix="%snorm" % self.prefix)

    def __call__(self,
                 data: mx.sym.Symbol,
                 prev: Optional[mx.sym.Symbol]) -> mx.sym.Symbol:
        """
        Apply processing sequence to data with optional previous input.

        :param data: Input data. Shape: (batch, length, num_hidden).
        :param prev: Previous data. Shape: (batch, length, num_hidden).
        :return: Processed data. Shape: (batch, length, num_hidden).
        """
        if not self.sequence:
            return data

        if prev is None:
            assert 'r' not in self.sequence, "Residual connection not allowed if no previous value given."

        for step in self.sequence:

            if step == "r":
                data = mx.sym._internal._plus(data, prev, name="%sresidual" % self.prefix)

            elif step == "n":
                data = self.layer_norm(data=data)

            elif step == "d":
                if self.dropout > 0.0:
                    data = mx.sym.Dropout(data, p=self.dropout, name="%sdropout" % self.prefix)
            else:
                raise ValueError("Unknown step in sequence: %s" % step)

        return data


class TransformerFeedForward:
    """
    Position-wise feed-forward network with activation.
    """

    def __init__(self,
                 num_hidden: int,
                 num_model: int,
                 act_type: str,
                 dropout: float,
                 prefix: str) -> None:
        self.num_hidden = num_hidden
        self.num_model = num_model
        self.dropout = dropout
        self.prefix = prefix
        self.act_type = act_type
        self.w_i2h = mx.sym.Variable('%si2h_weight' % prefix)
        self.b_i2h = mx.sym.Variable('%si2h_bias' % prefix)
        self.w_h2o = mx.sym.Variable('%sh2o_weight' % prefix)
        self.b_h2o = mx.sym.Variable('%sh2o_bias' % prefix)

    def __call__(self, x) -> mx.sym.Symbol:
        """
        Position-wise feed-forward network with activation.

        :param x: Symbol of shape (batch_size, seq_len, num_hidden)
        :return: Symbol of shape (batch_size, seq_len, num_hidden)
        """
        h = mx.sym.FullyConnected(data=x, num_hidden=self.num_hidden, weight=self.w_i2h, bias=self.b_i2h, flatten=False)
        h = layers.activation(h, act_type=self.act_type)
        if self.dropout > 0.0:
            h = mx.sym.Dropout(h, p=self.dropout)
        y = mx.sym.FullyConnected(data=h, num_hidden=self.num_model, weight=self.w_h2o, bias=self.b_h2o, flatten=False)
        return y


class VariableLengthBias(mx.operator.CustomOp):
    """
    Returns bias/mask given a vector of sequence lengths.
    """

    def __init__(self, max_length: int) -> None:
        super().__init__()
        self.max_length = max_length

    def forward(self, is_train, req, in_data, out_data, aux):
        # lengths: (batch_size,)
        lengths = in_data[0]
        dtype = lengths.dtype
        dtype_str = np.dtype(dtype).name

        # (batch_size, max_length)
        data = mx.nd.zeros((lengths.shape[0], self.max_length), dtype=dtype, ctx=lengths.context)
        data = mx.nd.SequenceMask(data=data,
                                  use_sequence_length=True,
                                  sequence_length=lengths,
                                  axis=1,
                                  value=-C.LARGE_VALUES[dtype_str])
        self.assign(out_data[0], req[0], data)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        pass


@mx.operator.register("variable_length_bias")
class VariableLengthBiasProp(mx.operator.CustomOpProp):

    def __init__(self, max_length: str) -> None:
        super().__init__()
        self.max_length = int(max_length)

    def list_arguments(self):
        return ['data']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        batch_size = in_shape[0][0]
        return in_shape, [(batch_size, self.max_length)], []

    def infer_type(self, in_type):
        return in_type, in_type, []

    def create_operator(self, ctx, shapes, dtypes):
        return VariableLengthBias(max_length=self.max_length)


def get_variable_length_bias(lengths: mx.sym.Symbol,
                             max_length: int,
                             num_heads: Optional[int] = None,
                             fold_heads: bool = True,
                             name: str = '') -> mx.sym.Symbol:
    """
    Returns bias/mask for variable sequence lengths.

    :param lengths: Sequence lengths. Shape: (batch,).
    :param max_length: Maximum sequence length.
    :param num_heads: Number of attention heads.
    :param fold_heads: Whether to fold heads dimension into batch dimension.
    :param name: Name of symbol.
    :return: Bias symbol.
    """
    # (batch_size, max_length)
    x = mx.symbol.Custom(data=lengths, max_length=max_length, op_type='variable_length_bias')
    if num_heads is not None:
        # (batch_size, heads, max_length) if fold_heads == False else (batch_size * heads, max_length)
        x = layers.broadcast_to_heads(x, num_heads, ndim=2, fold_heads=fold_heads)
    return mx.sym.BlockGrad(x, name='%sbias' % name)


class VariableLengthBiasExtended(mx.operator.CustomOp):
    """
    Returns bias/mask given a matrix of sequence lengths in each row.
    """

    def __init__(self, max_length: List[int]) -> None:
        super().__init__()
        self.max_length = max_length

    def forward(self, is_train, req, in_data, out_data, aux):
        # lengths: (batch_size, num_additional_input)
        lengths = in_data[0]
        dtype = lengths.dtype
        dtype_str = np.dtype(dtype).name

        # (batch_size, max_length)
        data = mx.nd.zeros((lengths.shape[0], sum(self.max_length)), dtype=dtype, ctx=lengths.context)
        start = 0
        for i, max_length in enumerate(self.max_length):
            data[:, start:start+max_length] = mx.nd.SequenceMask(data=data[:, start:start+max_length],
                                                                 use_sequence_length=True,
                                                                 sequence_length=lengths[:, i],
                                                                 axis=1,
                                                                 value=-C.LARGE_VALUES[dtype_str])
            start += max_length
        self.assign(out_data[0], req[0], data)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        pass


@mx.operator.register("variable_length_bias_extended")
class VariableLengthBiasPropExtended(mx.operator.CustomOpProp):

    def __init__(self, max_length: str) -> None:
        super().__init__()
        self.max_length = ast.literal_eval(max_length)

    def list_arguments(self):
        return ['data']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        batch_size = in_shape[0][0]
        num_additional_input = in_shape[0][1]
        return [(batch_size, num_additional_input)], [(batch_size, sum(self.max_length))], []

    def infer_type(self, in_type):
        return in_type, in_type, []

    def create_operator(self, ctx, shapes, dtypes):
        return VariableLengthBiasExtended(max_length=self.max_length)


def get_variable_length_bias_extended(lengths: mx.sym.Symbol,
                                      max_length: List[int],
                                      num_heads: Optional[int] = None,
                                      fold_heads: bool = True,
                                      name: str = '') -> mx.sym.Symbol:
    """
    Returns bias/mask for a variable sequence length matrix, extended to multiple sentences.

    :param lengths: Sequence lengths. Shape: (batch,).
    :param max_length: Maximum sequence lengths - one for each context sequence.
    :param num_heads: Number of attention heads.
    :param fold_heads: Whether to fold heads dimension into batch dimension.
    :param name: Name of symbol.
    :return: Bias symbol.
    """
    # (batch_size, max_length)
    x = mx.symbol.Custom(data=lengths, max_length=max_length, op_type='variable_length_bias_extended')
    if num_heads is not None:
        # (batch_size, heads, max_length) if fold_heads == False else (batch_size * heads, max_length)
        x = layers.broadcast_to_heads(x, num_heads, ndim=2, fold_heads=fold_heads)
    return mx.sym.BlockGrad(x, name='%sbias' % name)


def get_autoregressive_bias(max_length: int, name: str) -> mx.sym.Symbol:
    """
    Returns bias/mask to ensure position i can only attend to positions <i.

    :param max_length: Sequence length.
    :param name: Name of symbol.
    :return: Bias symbol of shape (1, max_length, max_length).
    """
    return mx.sym.BlockGrad(mx.symbol.Custom(length=max_length,
                                             name=name,
                                             op_type='auto_regressive_bias'))


class AutoRegressiveBias(mx.operator.CustomOp):
    """
    Returns a symbol of shape (1, length, length) with cells above the main diagonal
    set to a large negative value, e.g.
    length=4

    0 1 1 1
    0 0 1 1   * LARGE_NEGATIVE_VALUE
    0 0 0 1
    0 0 0 0
    """

    def __init__(self, length: int, dtype: str, ctx: mx.Context) -> None:
        super().__init__()
        self.bias = self.get_bias(length, dtype, ctx)

    @staticmethod
    def get_bias(length: int, dtype: str, ctx: mx.Context):
        # matrix with lower triangle and main diagonal set to 0, upper triangle set to 1
        upper_triangle = np.triu(np.ones((length, length), dtype=dtype), k=1)
        # (1, length, length)
        bias = -C.LARGE_VALUES[dtype] * np.reshape(upper_triangle, (1, length, length))
        return mx.nd.array(bias, ctx=ctx)

    def forward(self, is_train, req, in_data, out_data, aux):
        self.assign(out_data[0], req[0], self.bias)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        pass


@mx.operator.register("auto_regressive_bias")
class AutoRegressiveBiasProp(mx.operator.CustomOpProp):

    def __init__(self, length: str, dtype: str = C.DTYPE_FP32) -> None:
        super().__init__()
        self.length = int(length)
        self.dtype = dtype

    def list_arguments(self):
        return []

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        return [], [(1, self.length, self.length)], []

    def infer_type(self, in_type):
        return [], [np.dtype(self.dtype).type], []

    def create_operator(self, ctx, shapes, dtypes):
        return AutoRegressiveBias(length=self.length, dtype=self.dtype, ctx=ctx)
