import torch
import torch.nn as nn
from transformers import (
    VisionEncoderDecoderModel,
    ViTImageProcessor,
    AutoConfig,
    AutoModelForCausalLM,
    AutoModel,
)


def build_model(
    encoder_name: str = "microsoft/trocr-base-handwritten",
    decoder_name: str = "microsoft/trocr-base-handwritten",
    tokenizer=None,
    max_length: int = 256,
    dropout: float = 0.1,
    freeze_encoder: bool = False,
) -> VisionEncoderDecoderModel:
    if encoder_name == decoder_name:
        model = VisionEncoderDecoderModel.from_pretrained(encoder_name)
        if tokenizer is not None:
            model.decoder.resize_token_embeddings(tokenizer.vocab_size)
            model.config.vocab_size = tokenizer.vocab_size
            model.config.decoder.vocab_size = tokenizer.vocab_size
    else:
        encoder_config = AutoConfig.from_pretrained(encoder_name)
        encoder = AutoModel.from_pretrained(encoder_name, config=encoder_config)
    
        decoder_config = AutoConfig.from_pretrained(decoder_name)
        decoder_config.is_decoder = True
        decoder_config.add_cross_attention = True
        decoder_config.max_position_embeddings = max_length
        decoder_config.dropout = dropout
        decoder_config.attention_dropout = dropout
        decoder_config.resid_pdrop = dropout
    
        if tokenizer is not None:
            decoder_config.vocab_size = tokenizer.vocab_size
    
        decoder = AutoModelForCausalLM.from_config(decoder_config)
        model = VisionEncoderDecoderModel(encoder=encoder, decoder=decoder)

    if freeze_encoder:
        for param in model.encoder.parameters():
            param.requires_grad = False

    bos = tokenizer.bos_token_id if tokenizer else 0
    pad = tokenizer.pad_token_id if tokenizer else 0
    eos = tokenizer.eos_token_id if tokenizer else 0

    model.config.decoder_start_token_id = bos
    model.config.pad_token_id = pad
    model.config.eos_token_id = eos
    if tokenizer is not None:
        model.config.vocab_size = tokenizer.vocab_size

    model.config.decoder.pad_token_id = pad
    model.config.decoder.eos_token_id = eos

    model.generation_config.decoder_start_token_id = bos
    model.generation_config.pad_token_id = pad
    model.generation_config.eos_token_id = eos
    model.generation_config.max_length = max_length

    return model


def get_processor(encoder_name: str = "google/vit-base-patch16-384"):
    return ViTImageProcessor.from_pretrained(encoder_name)


def count_parameters(model: VisionEncoderDecoderModel) -> dict:
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    decoder_params = sum(p.numel() for p in model.decoder.parameters())
    total_params = encoder_params + decoder_params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "encoder": encoder_params,
        "decoder": decoder_params,
        "total": total_params,
        "trainable": trainable,
    }
