
import torchtext

def model_init(vocab_size: int,
                 embedding_dim: int,
                 ffn_dimension: int,
                 num_attention_heads: int,
                 num_encoder_layers: int,
                 max_seq_len: int,
                 padding_idx: int = 1,
                 dropout: float = 0.1,
                 scaling = None,
                 normalize_before: bool = False):
    '''Initialize a distilled RoBERTa encoder and returns the
       model and the transformations it was trained.'''

    base = torchtext.models.ROBERTA_DISTILLED_ENCODER
    config = base.encoderConf

    config.vocab_size=vocab_size
    config.embedding_dim=embedding_dim
    config.ffn_dimension=ffn_dimension
    config.padding_idx=padding_idx
    config.max_seq_len=max_seq_len
    config.num_attention_heads=num_attention_heads
    config.num_encoder_layers=num_encoder_layers
    config.dropout=dropout
    config.scaling=scaling
    config.normalize_before=normalize_before

    model = base.build_model(encoder_conf=config)
    transforms = base.transform()

    transformer_encoder = model.encoder.transformer.layers.layers

    for i in range(num_encoder_layers):
        transformer_encoder.get_submodule(f'{i}').linear1 = nn.Linear(in_features=embedding_dim, out_features=ffn_dimension, bias=True)
        transformer_encoder.get_submodule(f'{i}').linear2 = nn.Linear(in_features=ffn_dimension, out_features=embedding_dim, bias=True)
    
    print(config)
    return model, transforms
