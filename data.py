import tensorflow as tf
import numpy as np

# --- Tokenizer and Vocabulary ---
def get_tokenizer(corpus):
    """
    Creates a simple character-level tokenizer from a corpus.
    Adds special tokens for masking and sequence start/end.
    """
    # Create a vocabulary from the corpus
    vocab = sorted(set(corpus))

    # Define special tokens
    special_tokens = ['[MASK]', '[UNK]', '[START]', '[END]']

    # Combine vocab and special tokens, ensuring no duplicates
    full_vocab = special_tokens + [v for v in vocab if v not in special_tokens]

    # Create the StringLookup layer
    tokenizer = tf.keras.layers.StringLookup(
        vocabulary=full_vocab,
        mask_token='[MASK]',  # This is the token to use for masking
        oov_token='[UNK]'     # This is the token for out-of-vocabulary words
    )
    return tokenizer

# --- Phase 1: Unsupervised Pre-training Data ---
def generate_pretraining_data(corpus, tokenizer, max_len, batch_size, mask_fraction=0.25):
    """
    Generates a tf.data.Dataset for Phase 1 pre-training.
    """
    # 1. Tokenize the entire corpus
    tokenized_corpus = tokenizer(tf.strings.bytes_split(corpus, 'UTF-8'))

    # 2. Create overlapping sequences of max_len
    sequences = []
    for i in range(len(tokenized_corpus) - max_len + 1):
        sequences.append(tokenized_corpus[i:i+max_len])

    if not sequences:
        raise ValueError("Corpus is too short for the given max_len. No sequences were created.")

    dataset = tf.data.Dataset.from_tensor_slices(sequences)

    def create_masked_views(seq):
        # x_target is the original, unmodified sequence
        x_target = seq

        # Determine the number of tokens to mask
        num_to_mask = tf.cast(tf.cast(max_len, tf.float32) * mask_fraction, tf.int32)

        # Randomly select indices to mask
        mask_indices = tf.random.shuffle(tf.range(max_len))[:num_to_mask]
        mask_indices = tf.expand_dims(mask_indices, axis=1) # Required for scatter_nd_update

        # Get the ID for the [MASK] token
        mask_token_id = tokenizer(['[MASK]'])[0]

        # Create x_online by replacing tokens at mask_indices with the mask token
        x_online = tf.tensor_scatter_nd_update(
            x_target,
            mask_indices,
            tf.fill([num_to_mask], mask_token_id)
        )

        # For the loss calculation, we need indices relative to the batch.
        # This will be paired with batch indices in the train_step.
        return (x_online, x_target), mask_indices

    dataset = dataset.map(create_masked_views, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(buffer_size=1024).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return dataset

# --- Phase 2: Supervised Fine-tuning Data ---
def generate_finetuning_data(qa_pairs, tokenizer, max_len, batch_size):
    """
    Generates a tf.data.Dataset for Phase 2 fine-tuning.
    """
    questions = [pair['q'] for pair in qa_pairs]
    answers = [pair['a'] for pair in qa_pairs]

    start_token = tokenizer(['[START]'])
    end_token = tokenizer(['[END]'])

    # Tokenize questions for the encoder input
    tokenized_questions_ragged = tokenizer(tf.strings.bytes_split(questions, 'UTF-8'))
    tokenized_questions = tokenized_questions_ragged.to_list()

    # Tokenize answers and create decoder inputs/targets
    tokenized_answers_ragged = tokenizer(tf.strings.bytes_split(answers, 'UTF-8'))
    tokenized_answers = tokenized_answers_ragged.to_list()

    decoder_inputs = [tf.concat([start_token, ans], axis=0) for ans in tokenized_answers]
    decoder_targets = [tf.concat([ans, end_token], axis=0) for ans in tokenized_answers]

    mask_token_id = tokenizer(['[MASK]'])[0]
    # Pad all sequences to the specified max_len
    encoder_inputs = tf.keras.preprocessing.sequence.pad_sequences(
        tokenized_questions, maxlen=max_len, padding='post', value=mask_token_id
    )
    decoder_inputs = tf.keras.preprocessing.sequence.pad_sequences(
        decoder_inputs, maxlen=max_len, padding='post', value=mask_token_id
    )
    decoder_targets = tf.keras.preprocessing.sequence.pad_sequences(
        decoder_targets, maxlen=max_len, padding='post', value=mask_token_id
    )

    dataset = tf.data.Dataset.from_tensor_slices(
        ((encoder_inputs, decoder_inputs), decoder_targets)
    )
    dataset = dataset.shuffle(1024).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return dataset
