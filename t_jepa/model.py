import tensorflow as tf
from tensorflow.keras import Model
from t_jepa.encoder import JEPAEncoderCore
from t_jepa.decoder import DecoderHead, create_look_ahead_mask, create_padding_mask

class TJEPA(Model):
    """
    The complete Text-Joint Embedding Predictive Architecture (T-JEPA) model.
    """
    def __init__(self, num_layers_enc, embed_dim, num_heads_enc, ff_dim_enc, vocab_size, max_len,
                 predictor_hidden_dim, tau,
                 num_layers_dec, num_heads_dec, ff_dim_dec):
        super(TJEPA, self).__init__()

        self.encoder = JEPAEncoderCore(
            num_layers=num_layers_enc,
            embed_dim=embed_dim,
            num_heads=num_heads_enc,
            ff_dim=ff_dim_enc,
            vocab_size=vocab_size,
            max_len=max_len,
            predictor_hidden_dim=predictor_hidden_dim,
            tau=tau
        )

        self.decoder = DecoderHead(
            num_layers=num_layers_dec,
            d_model=embed_dim, # Decoder's d_model must match encoder's embed_dim
            num_heads=num_heads_dec,
            dff=ff_dim_dec,
            target_vocab_size=vocab_size, # For simplicity, shared vocab
            max_len=max_len
        )

    def freeze_encoder(self):
        """
        Freezes the encoder for Phase 2 training.
        """
        self.encoder.online_encoder.trainable = False
        self.encoder.predictor.trainable = False
        print("Encoder frozen.")

    def unfreeze_encoder(self):
        """
        Unfreezes the encoder for Phase 1 training.
        """
        self.encoder.online_encoder.trainable = True
        self.encoder.predictor.trainable = True
        print("Encoder unfrozen.")

    def call(self, inputs, training=False):
        # This call is for Phase 2: Supervised fine-tuning
        encoder_input, decoder_input = inputs

        # Create masks for the decoder
        look_ahead_mask = create_look_ahead_mask(tf.shape(decoder_input)[1])
        # The padding mask for the decoder's cross-attention is not strictly
        # necessary if the encoder input is not padded, but it's good practice.
        # For this toy example, we assume no padding on the encoder side.

        # 1. Get the "thought vector" from the frozen online encoder
        enc_output = self.encoder.online_encoder(encoder_input, training=False)

        # 2. The decoder generates the output sequence
        dec_output, _ = self.decoder(
            x=decoder_input,
            enc_output=enc_output,
            training=training,
            look_ahead_mask=look_ahead_mask
        )

        return dec_output
