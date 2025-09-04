import tensorflow as tf
from tensorflow.keras import layers, Model
from t_jepa.encoder import positional_encoding

class DecoderLayer(layers.Layer):
    """
    A single layer of the Transformer decoder.
    """
    def __init__(self, *, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()

        self.mha1 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.mha2 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)

        self.ffn = tf.keras.Sequential([
            layers.Dense(dff, activation='relu'),
            layers.Dense(d_model)
        ])

        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)
        self.dropout3 = layers.Dropout(rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask=None):
        # Self-attention block (with look-ahead mask)
        attn1, attn_weights_block1 = self.mha1(
            query=x, value=x, key=x,
            attention_mask=look_ahead_mask,
            return_attention_scores=True
        )
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        # Cross-attention block (with padding mask)
        attn2, attn_weights_block2 = self.mha2(
            query=out1,
            value=enc_output,
            key=enc_output,
            attention_mask=padding_mask,
            return_attention_scores=True)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)

        # Feed-forward block
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)

        return out3, attn_weights_block1, attn_weights_block2


class DecoderHead(Model):
    """
    The full Decoder Head model.
    """
    def __init__(self, *, num_layers, d_model, num_heads, dff, target_vocab_size, max_len, rate=0.1):
        super(DecoderHead, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = positional_encoding(max_len, d_model)

        self.dec_layers = [
            DecoderLayer(d_model=d_model, num_heads=num_heads, dff=dff, rate=rate)
            for _ in range(num_layers)
        ]
        self.dropout = layers.Dropout(rate)
        self.final_layer = layers.Dense(target_vocab_size)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask=None):
        seq_len = tf.shape(x)[1]
        attention_weights = {}

        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:seq_len, :]

        x = self.dropout(x, training=training)

        for i, dec_layer in enumerate(self.dec_layers):
            x, block1, block2 = dec_layer(
                x, enc_output, training=training, look_ahead_mask=look_ahead_mask, padding_mask=padding_mask
            )
            attention_weights[f'decoder_layer{i+1}_block1'] = block1
            attention_weights[f'decoder_layer{i+1}_block2'] = block2

        final_output = self.final_layer(x)
        return final_output, attention_weights

def create_look_ahead_mask(size):
    """
    Creates a look-ahead mask for self-attention in the decoder.
    """
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask

def create_padding_mask(seq):
    """
    Creates a padding mask for the input sequence.
    """
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    return seq[:, tf.newaxis, tf.newaxis, :] # (batch_size, 1, 1, seq_len)
