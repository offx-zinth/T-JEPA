import tensorflow as tf
from tensorflow.keras import layers, Model

def positional_encoding(length, depth):
    """
    Generates positional encodings for a transformer model.
    """
    depth = depth / 2
    positions = tf.range(length, dtype=tf.float32)[:, tf.newaxis]
    depths = tf.range(depth, dtype=tf.float32)[tf.newaxis, :] / depth
    angle_rates = 1 / (10000**depths)
    angle_rads = positions * angle_rates
    pos_encoding = tf.concat([tf.sin(angle_rads), tf.cos(angle_rads)], axis=-1)
    return tf.cast(pos_encoding, dtype=tf.float32)

class TransformerEncoderBlock(layers.Layer):
    """
    A single block of the Transformer encoder.
    """
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerEncoderBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim)]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class JEPAEncoderModel(Model):
    """
    The base JEPA encoder model.
    """
    def __init__(self, num_layers, embed_dim, num_heads, ff_dim, vocab_size, max_len, rate=0.1):
        super(JEPAEncoderModel, self).__init__()
        self.embed_dim = embed_dim
        self.token_emb = layers.Embedding(vocab_size, embed_dim)
        self.pos_emb = positional_encoding(max_len, embed_dim)
        self.encoder_blocks = [
            TransformerEncoderBlock(embed_dim, num_heads, ff_dim, rate)
            for _ in range(num_layers)
        ]
        self.dropout = layers.Dropout(rate)

    def call(self, x, training=False):
        seq_len = tf.shape(x)[1]
        x = self.token_emb(x)
        x *= tf.math.sqrt(tf.cast(self.embed_dim, tf.float32))
        x += self.pos_emb[:seq_len, :]
        x = self.dropout(x, training=training)

        for block in self.encoder_blocks:
            x = block(x, training=training)
        return x

class Predictor(Model):
    """
    The predictor network that predicts target representations from online representations.
    """
    def __init__(self, embed_dim, hidden_dim, rate=0.1):
        super(Predictor, self).__init__()
        self.model = tf.keras.Sequential([
            layers.Dense(hidden_dim, activation='relu'),
            layers.Dropout(rate),
            layers.Dense(embed_dim)
        ])

    def call(self, x, training=False):
        return self.model(x, training=training)

class JEPAEncoderCore(Model):
    """
    Manages the online encoder, target encoder, and predictor.
    """
    def __init__(self, num_layers, embed_dim, num_heads, ff_dim, vocab_size, max_len, predictor_hidden_dim, tau=0.996):
        super(JEPAEncoderCore, self).__init__()
        self.tau = tau

        # Online Network
        self.online_encoder = JEPAEncoderModel(num_layers, embed_dim, num_heads, ff_dim, vocab_size, max_len)
        self.predictor = Predictor(embed_dim, predictor_hidden_dim)

        # Target Network
        self.target_encoder = JEPAEncoderModel(num_layers, embed_dim, num_heads, ff_dim, vocab_size, max_len)
        self.target_encoder.set_weights(self.online_encoder.get_weights())
        self.target_encoder.trainable = False

    def ema_update(self):
        """
        Update the target encoder's weights using exponential moving average.
        """
        for online_weight, target_weight in zip(self.online_encoder.weights, self.target_encoder.weights):
            target_weight.assign(self.tau * target_weight + (1 - self.tau) * online_weight)

    def call(self, x_online, x_target=None, training=False):
        online_reps = self.online_encoder(x_online, training=training)

        if training:
            predicted_reps = self.predictor(online_reps, training=training)
            target_reps = self.target_encoder(x_target, training=False)
            return predicted_reps, tf.stop_gradient(target_reps)

        return online_reps

    def train_step(self, data):
        (x_online, x_target), masked_indices = data

        with tf.GradientTape() as tape:
            predicted_reps, target_reps = self(x_online, x_target, training=True)

            # Create indices for gather_nd
            batch_size = tf.shape(x_online)[0]
            num_masked = tf.shape(masked_indices)[1]

            batch_idx = tf.range(batch_size)
            batch_idx = tf.reshape(batch_idx, (batch_size, 1))
            batch_idx = tf.tile(batch_idx, (1, num_masked))
            batch_idx = tf.reshape(batch_idx, (-1, 1))

            masked_indices_flat = tf.reshape(masked_indices, (-1, 1))

            full_indices = tf.concat([tf.cast(batch_idx, masked_indices_flat.dtype), masked_indices_flat], axis=1)

            # Gather the representations of the masked tokens
            masked_predictions = tf.gather_nd(predicted_reps, full_indices)
            masked_targets = tf.gather_nd(target_reps, full_indices)

            loss = self.compiled_loss(masked_targets, masked_predictions)

        # Compute gradients
        trainable_vars = self.online_encoder.trainable_variables + self.predictor.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update EMA for the target encoder
        self.ema_update()

        # Update metrics
        self.compiled_metrics.update_state(masked_targets, masked_predictions)

        return {m.name: m.result() for m in self.metrics}
