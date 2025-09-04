import tensorflow as tf
import numpy as np
from t_jepa.model import TJEPA
from data import get_tokenizer, generate_pretraining_data, generate_finetuning_data

# --- 1. Hyperparameters ---
# Model Dims
EMBED_DIM = 64
FF_DIM = 256
NUM_HEADS = 4
NUM_LAYERS_ENC = 3
NUM_LAYERS_DEC = 3
MAX_LEN = 48  # Reduced to better fit the small corpus
PREDICTOR_HIDDEN_DIM = 128

# Training Params
BATCH_SIZE = 8
EPOCHS_PHASE1 = 15
EPOCHS_PHASE2 = 30
LEARNING_RATE_P1 = 1e-3
LEARNING_RATE_P2 = 1e-3
TAU = 0.996

# --- 2. Toy Data ---
pretrain_corpus = (
    "The quick brown fox jumps over the lazy dog. "
    "Artificial intelligence is a field of computer science. "
    "JEPA stands for Joint Embedding Predictive Architecture. "
    "This architecture separates understanding from expression. "
    "The encoder builds a world model, and the decoder generates language. "
    "TensorFlow is a popular deep learning framework used for training models."
)
qa_pairs = [
    {'q': 'what is ai?', 'a': 'a field of computer science'},
    {'q': 'what is jepa?', 'a': 'joint embedding predictive architecture'},
    {'q': 'how does t-jepa work?', 'a': 'it separates understanding and expression'},
    {'q': 'what is tensorflow?', 'a': 'a deep learning framework'}
]

# --- 3. Tokenizer and Datasets ---
full_corpus = pretrain_corpus + " ".join([p['q'] + ' ' + p['a'] for p in qa_pairs])
tokenizer = get_tokenizer(full_corpus)
VOCAB_SIZE = tokenizer.vocabulary_size()
PADDING_TOKEN_ID = tokenizer.vocabulary_size() - tokenizer.vocabulary_size() # This is 0 if mask token is at the start

# Create datasets
pretrain_dataset = generate_pretraining_data(pretrain_corpus, tokenizer, MAX_LEN, BATCH_SIZE)
finetune_dataset = generate_finetuning_data(qa_pairs, tokenizer, MAX_LEN, BATCH_SIZE)

# --- 4. Instantiate Model ---
t_jepa_model = TJEPA(
    num_layers_enc=NUM_LAYERS_ENC, embed_dim=EMBED_DIM, num_heads_enc=NUM_HEADS,
    ff_dim_enc=FF_DIM, vocab_size=VOCAB_SIZE, max_len=MAX_LEN,
    predictor_hidden_dim=PREDICTOR_HIDDEN_DIM, tau=TAU,
    num_layers_dec=NUM_LAYERS_DEC, num_heads_dec=NUM_HEADS, ff_dim_dec=FF_DIM,
)

# --- 5. Phase 1: Unsupervised Pre-training ---
print("--- Starting Phase 1: Unsupervised Pre-training ---")
t_jepa_model.unfreeze_encoder()
encoder_core = t_jepa_model.encoder
encoder_core.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE_P1),
    loss=tf.keras.losses.MeanSquaredError()
)
encoder_core.fit(pretrain_dataset, epochs=EPOCHS_PHASE1, verbose=1)
print("\n--- Phase 1 Complete ---")

# --- 6. Phase 2: Supervised Fine-tuning ---
print("\n--- Starting Phase 2: Supervised Fine-tuning ---")
t_jepa_model.freeze_encoder()

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
def masked_loss(label, pred):
    mask = tf.math.not_equal(label, PADDING_TOKEN_ID)
    loss_ = loss_object(label, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_sum(loss_) / tf.reduce_sum(mask)

def masked_accuracy(label, pred):
    pred = tf.argmax(pred, axis=2, output_type=label.dtype)
    mask = tf.math.not_equal(label, PADDING_TOKEN_ID)
    match = (label == pred) & mask
    return tf.reduce_sum(tf.cast(match, tf.float32)) / tf.reduce_sum(tf.cast(mask, tf.float32))

t_jepa_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE_P2),
    loss=masked_loss,
    metrics=[masked_accuracy]
)
t_jepa_model.fit(finetune_dataset, epochs=EPOCHS_PHASE2, verbose=1)
print("\n--- Phase 2 Complete ---")

# --- 7. Inference Demonstration ---
def generate_response(question, model, tokenizer, max_len=MAX_LEN):
    print(f"\n> Generating response for: '{question}'")

    tokenized_q = tokenizer(tf.strings.bytes_split([question], 'UTF-8'))
    enc_input = tf.keras.preprocessing.sequence.pad_sequences(
        tokenized_q, maxlen=max_len, padding='post', value=PADDING_TOKEN_ID
    )

    start_token_id = tokenizer(['[START]'])[0]
    end_token_id = tokenizer(['[END]'])[0]

    output = tf.constant([start_token_id], dtype=tf.int64)
    output = tf.expand_dims(output, 0)

    for _ in range(max_len):
        predictions = model((enc_input, output), training=False)
        predictions = predictions[:, -1:, :]
        predicted_id = tf.argmax(predictions, axis=-1, output_type=tf.int64)

        if predicted_id[0, 0] == end_token_id:
            break

        output = tf.concat([output, predicted_id], axis=-1)

    result_ids = output.numpy().flatten()
    vocab = tokenizer.get_vocabulary()
    response_tokens = [vocab[i] for i in result_ids if i < len(vocab)]

    response = "".join([t for t in response_tokens if t not in ['[START]', '[END]', '[MASK]', '[UNK]']])
    print(f"< Response: {response}")

# Demonstrate
generate_response("what is jepa?", t_jepa_model, tokenizer)
generate_response("what is ai?", t_jepa_model, tokenizer)
