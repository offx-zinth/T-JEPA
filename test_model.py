import unittest
import tensorflow as tf
import numpy as np
from t_jepa.model import TJEPA
from data import get_tokenizer, generate_pretraining_data, generate_finetuning_data

class TestTJEPA(unittest.TestCase):

    def setUp(self):
        """Set up a small model and data for testing."""
        self.config = {
            'num_layers_enc': 1, 'embed_dim': 32, 'num_heads_enc': 2, 'ff_dim_enc': 64,
            'max_len': 24, 'predictor_hidden_dim': 32, 'tau': 0.99,
            'num_layers_dec': 1, 'num_heads_dec': 2, 'ff_dim_dec': 64,
            'batch_size': 2,
        }
        # Ensure corpus is long enough for at least one batch
        self.pretrain_corpus = "a small corpus for testing purposes that is definitely long enough to form a sequence"
        self.qa_pairs = [{'q': 'test q', 'a': 'test a'}]

        self.tokenizer = get_tokenizer(self.pretrain_corpus + " " + self.qa_pairs[0]['q'] + " " + self.qa_pairs[0]['a'])
        self.config['vocab_size'] = self.tokenizer.vocabulary_size()

    def _create_model(self):
        """Helper method to instantiate the TJEPA model with test config."""
        return TJEPA(
            num_layers_enc=self.config['num_layers_enc'], embed_dim=self.config['embed_dim'],
            num_heads_enc=self.config['num_heads_enc'], ff_dim_enc=self.config['ff_dim_enc'],
            vocab_size=self.config['vocab_size'], max_len=self.config['max_len'],
            predictor_hidden_dim=self.config['predictor_hidden_dim'], tau=self.config['tau'],
            num_layers_dec=self.config['num_layers_dec'], num_heads_dec=self.config['num_heads_dec'],
            ff_dim_dec=self.config['ff_dim_dec']
        )

    def test_01_model_instantiation(self):
        """Test if the TJEPA model can be instantiated without errors."""
        model = self._create_model()
        self.assertIsInstance(model, TJEPA)
        print("\nPASS: Model instantiation test.")

    def test_02_output_shapes(self):
        """Test if the model's sub-components produce outputs with correct shapes."""
        model = self._create_model()

        # Test encoder output shape
        input_data = np.random.randint(0, self.config['vocab_size'], (self.config['batch_size'], self.config['max_len']))
        enc_output = model.encoder.online_encoder(input_data)
        self.assertEqual(enc_output.shape, (self.config['batch_size'], self.config['max_len'], self.config['embed_dim']))

        # Test decoder output shape
        dec_input = np.random.randint(0, self.config['vocab_size'], (self.config['batch_size'], self.config['max_len']))
        dec_output, _ = model.decoder(dec_input, enc_output, training=False, look_ahead_mask=None)
        self.assertEqual(dec_output.shape, (self.config['batch_size'], self.config['max_len'], self.config['vocab_size']))
        print("PASS: Output shapes test.")

    def test_03_phase1_training_step(self):
        """Test if a single training step of Phase 1 executes without errors."""
        model = self._create_model()
        pretrain_dataset = generate_pretraining_data(
            self.pretrain_corpus, self.tokenizer, self.config['max_len'], self.config['batch_size']
        )

        encoder_core = model.encoder
        encoder_core.compile(optimizer='adam', loss='mse')

        history = encoder_core.fit(pretrain_dataset, epochs=1, verbose=0)
        self.assertIn('loss', history.history)
        self.assertTrue(np.isfinite(history.history['loss'][0]))
        print("PASS: Phase 1 training step test.")

    def test_04_phase2_training_step(self):
        """Test if a single training step of Phase 2 executes without errors."""
        model = self._create_model()
        finetune_dataset = generate_finetuning_data(
            self.qa_pairs, self.tokenizer, self.config['max_len'], self.config['batch_size']
        )

        model.freeze_encoder()
        model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))

        history = model.fit(finetune_dataset, epochs=1, verbose=0)
        self.assertIn('loss', history.history)
        self.assertTrue(np.isfinite(history.history['loss'][0]))
        print("PASS: Phase 2 training step test.")

if __name__ == '__main__':
    unittest.main()
