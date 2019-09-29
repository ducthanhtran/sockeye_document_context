# Sockeye with Document Context Information
This sockeye-based neural machine translation (NMT) toolkit is capable of using surrounding sentences as additional context information. For further information we refer to our [publication](https://github.com/ducthanhtran/sockeye_document_context#citation).

We have used [sockeye version 1.18.85](https://github.com/awslabs/sockeye/releases/tag/1.18.85) as a starting point of our implementation and worked on the Transformer architecture of [(Vaswani et al., 2017 NIPS)](https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf) due to its efficient training computation and better translation performance in comparison to recurrent neural networks.


* [Installation](https://github.com/ducthanhtran/sockeye_document_context#Installation)
* [Training](https://github.com/ducthanhtran/sockeye_document_context#Training)
    * [Context-aware Architectures using multiple Encoders](https://github.com/ducthanhtran/sockeye_document_context#Context-aware%20Architectures%20using%20multiple%20Encoders)
* [Inference](https://github.com/ducthanhtran/sockeye_document_context#Inference)
* [Citation](https://github.com/ducthanhtran/sockeye_document_context#Citation)
* [Parameters](https://github.com/ducthanhtran/sockeye_document_context#Parameters)
 


## Installation
1. Clone this repository with ```git clone git@github.com:ducthanhtran/sockeye_document_context.git```
2. Install required packages via ```pip install -r requirements/requirements.txt```
3. [Optional-GPU] In order to run sockeye on GPUs one has to further install packages via ```pip install -r requirements/requirements.gpu-cu${CUDA_VERSION}.txt``` where ```${CUDA_VERSION}``` can be 80 (8.0), 90 (9.0) or 100 (10.0)
 
 
## Training
Here is an example of using the **outside-decoder** implementation with the base transformer model parameters from [(Vaswani et al., 2017 NIPS)](https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf), that is, using 6 encoder and decoder layers, 512 dimensions for embedding and model size, 2048 feature dimensions for the feed-forward-sublayers, among others.

Here is a small training example file which trains a model on CPU. 
```bash
python -m sockeye.train -s ${TRAINING_SOURCE} \
                        -t ${TRAINING_TARGET} \
                        -vs ${VALIDATION_SOURCE} \
                        -vt ${VALIDATION_TARGET} \
                        --batch-type word \
                        --batch-size 3000 \
                        --embed-dropout 0:0 \
                        --checkpoint-frequency 2000 \
                        --encoder transformer \
                        --num-layers 6:6 \
                        --num-layers-doc 6 \
                        --transformer-model-size 512 \
                        --transformer-model-size-doc 512 \
                        --transformer-attention-heads 8 \
                        --transformer-attention-heads-doc 8 \
                        --transformer-feed-forward-num-hidden 2048 \
                        --transformer-feed-forward-num-hidden-doc 2048 \
                        --transformer-preprocess n \
                        --transformer-preprocess-doc n \
                        --transformer-postprocess dr \
                        --transformer-postprocess-doc dr \
                        --transformer-dropout-prepost 0.1 \
                        --transformer-dropout-prepost-doc 0.1 \
                        --transformer-dropout-act-doc 0.1 \
                        --transformer-dropout-attention-doc 0.1 \
                        --transformer-positional-embedding-type fixed \
                        --label-smoothing 0.1 \
                        --num-embed 512:512 \
                        --learning-rate-reduce-num-not-improved 3 \
                        --max-num-checkpoint-not-improved 4 \
                        --seed 100 \
                        --max-seq-len 99:99 \
                        --decode-and-evaluate 0 \
                        --use-cpu \
                        --method outside-decoder \
                        --src-pre 1 \
                        --src-nxt 0 \
                        --tar-pre 0 \
                        --tar-nxt 0 \
                        --source-train-doc ${SOURCE_TRAIN_CONTEXT} \
                        --source-valid-doc ${VALIDATION_SOURCE_CONTEXT} \
                        --output ${MODEL_DIR_OUTPUT}
```
Most context-related parameters have the suffix *-doc*. For example, the parameter *--transformer-dropout-attention-doc* denotes how much dropout is used in all additional attention components throughout the network.

Moreover, the parameters *--src-pre*, *--src-nxt*, *--tar-pre* and *--tar-nxt* are used to specify the context window size. At the moment we are not able to use both source and target context information together. Note that in order to use additional source context information, we also need to employ data to the parameters *--source-train-doc* and *--source-valid-doc*. The same is done correspondingly when using target context data, which uses *--target-train-doc* and *--target-valid-doc*, respectively.


### Context-aware Architectures using multiple Encoders
The parameter *--method* is used to select the desired architecture. One can choose the following:
* **outside-decoder**: Combine encoder/embedding representations together with the representation of the current sentence. Afterwards, we perform a linear interpolation which results in a final encoder representation of all inputs. Consequently, this representation is used in all decoder layers of the transformer in the encoder-decoder attention component.
* **inside-decoder-sequential-attention**: The encoder representations of context data are integrated inside each decoder layer by using an attention component. The query is the output of the encoder-decoder attention component, thus a sequential attention computation is performed: first, attending to the current source sentence and then attending to context sentences. Afterwards, both attention outputs are interpolated linearly with a gating mechanism. The result is then forwarded to the feed-forward-sublayer.
* **inside-decoder-parallel-attention**: Instead of performing attention sequentially, we perform them in parallel this time. Hence, the input/query for the context-attention is the self-attended representation in the decoder layer. Afterwards, we follow the gated interpolation as stated before.  


## Inference
The following example depicts the translation process where a beam-size of 5 is utilized. For this matter, the model directory and a regular source file is used. As context sentences, we use the original test sentences (see *--input-source-doc*). Right now, the implementation requires the *--method* parameter also in the inference call and should be identical to the trained model in `${MODEL_OUTPUT_DIR}`. Here, the trained model is using the *outside-decoder* context-ware architecture, thus we need to specify this in the inference call as well. 
```bash
zcat ${TEST_SOURCE} | python -m sockeye.translate \
                                --beam-size 5 \
                                --model ${MODEL_OUTPUT_DIR} \
                                --use-cpu \
                                --method outside-decoder \
                                --input-source-doc ${TEST_SOURCE} \
                                --output ${HYP_OUTPUT}
```

 
## Citation
If you use this software, please cite the following publication
> Yunsu Kim, Duc Thanh Tran, Hermann Ney: **When and Why is Document-level Context Useful in Neural Machine Translation?** at EMNLP 2019 4th Workshop on Discourse in Machine Translation (DiscoMT 2019), Hong Kong, China, November 2019.

and

> Felix Hieber, Tobias Domhan, Michael Denkowski, David Vilar, Artem Sokolov, Ann Clifton, Matt Post:
**The Sockeye Neural Machine Translation Toolkit** at AMTA 2018. AMTA (1) 2018: 200-207

## Parameters

### Training
#### Context Window 
* --src-pre: Number of previous source sentences taken as context information
* --src-nxt: Number of next source sentences taken as context information
* --tar-pre: Number of previous target sentences taken as context information
* --tar-nxt: Number of next source sentences taken as context information

#### Context Inputs
* --source-train-doc: Training context data for source side
* --target-train-doc: Training context data for target side
* --source-valid-doc: Validation context data for source side
* -target-valid-doc: Validation context data for target side
* --bucket-width-doc: Bucket width for context sentences

#### Model Parameters
* --method: Selection of context-aware model architecture
* --encoder-doc: Context encoder architecture. At the moment we only support the Transformer
* --num-layers-doc: Number of Transformer encoder layers for context sentences
* --transformer-model-size-doc: Model dimensionality of context encoders. All model sizes should have the same value to enable gating mechanism
* --transformer-attention-heads-doc: Number of attention heads in all context-relevant multihead attention layers
* --transformer-feed-forward-num-hidden-doc: Specifies dimensionality of feed-forward sublayers, if context encoder layers are used
* --transformer-activation-type-doc: Activation function for feed-forward sublayers
* --transformer-positional-embedding-type-doc: Positional embedding for context sentences. Only used when encoder layers are utilized
* --transformer-preprocess-doc: Pre-process sequence before each sublayer in the context encoders
* --transformer-postprocess-doc: Post-process sequence after each sublayer in the context encoders

#### Training Parameters
* --embed-dropout-doc: Apply dropout onto context embeddings
* --transformer-dropout-attention-doc: Apply dropout within attention layers in context encoders
* --transformer-dropout-act-doc: Apply dropout within feed-forward sublayers in context encoders
* --transformer-dropout-prepost-doc: Apply dropout in the pre-/post-sequences in context encoders



### Inference
* --method: Should be identical to the trained model
* --input-source-doc: Context sentences for source side in inference mode 
* --input-target-doc: Context sentences for target side in inference modefor 