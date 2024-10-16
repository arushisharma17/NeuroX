import unittest
from unittest.mock import patch, MagicMock, mock_open
import numpy as np
import torch

# Import the script's functions
from transformers_extractors import (
    get_model_and_tokenizer,
    aggregate_repr,
    extract_sentence_representations,
)

class TestTransformersExtractor(unittest.TestCase):
    def setUp(self):
        # Common setup for the tests
        self.device = torch.device("cpu")
        self.model_name = "distilbert-base-uncased"
        self.sentence = "Hello, how are you?"

    @patch("transformers.AutoTokenizer.from_pretrained")
    @patch("transformers.AutoModel.from_pretrained")
    def test_get_model_and_tokenizer_encoder(self, mock_model_from_pretrained, mock_tokenizer_from_pretrained):
        # Mock the model and tokenizer
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_model_from_pretrained.return_value = mock_model
        mock_tokenizer_from_pretrained.return_value = mock_tokenizer

        model, tokenizer = get_model_and_tokenizer(self.model_name, device=self.device)

        mock_model_from_pretrained.assert_called_once_with(
            self.model_name, output_hidden_states=True
        )
        mock_tokenizer_from_pretrained.assert_called_once_with(self.model_name)
        self.assertEqual(model, mock_model)
        self.assertEqual(tokenizer, mock_tokenizer)

    @patch("transformers.AutoTokenizer.from_pretrained")
    @patch("transformers.AutoModelForCausalLM.from_pretrained")
    @patch("transformers.AutoConfig.from_pretrained")
    def test_get_model_and_tokenizer_decoder(
        self, mock_config_from_pretrained, mock_model_from_pretrained, mock_tokenizer_from_pretrained
    ):
        # Mock the config, model, and tokenizer
        mock_config = MagicMock()
        mock_config.is_decoder = True
        mock_config.model_type = "gpt2"
        mock_config_from_pretrained.return_value = mock_config

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_model_from_pretrained.return_value = mock_model
        mock_tokenizer_from_pretrained.return_value = mock_tokenizer

        model, tokenizer = get_model_and_tokenizer(self.model_name, device=self.device)

        mock_model_from_pretrained.assert_called_once_with(
            self.model_name,
            return_dict_in_generate=True,
            output_hidden_states=True,
            trust_remote_code=True,
        )
        mock_tokenizer_from_pretrained.assert_called_once_with(self.model_name)
        self.assertEqual(model, mock_model)
        self.assertEqual(tokenizer, mock_tokenizer)

    def test_aggregate_repr_first(self):
        # Create a dummy state tensor
        state = np.random.rand(12, 10, 768)
        start = 2
        end = 5
        aggregation = "first"
        result = aggregate_repr(state, start, end, aggregation)
        expected = state[:, start, :]
        np.testing.assert_array_equal(result, expected)

    def test_aggregate_repr_last(self):
        state = np.random.rand(12, 10, 768)
        start = 2
        end = 5
        aggregation = "last"
        result = aggregate_repr(state, start, end, aggregation)
        expected = state[:, end, :]
        np.testing.assert_array_equal(result, expected)

    def test_aggregate_repr_average(self):
        state = np.random.rand(12, 10, 768)
        start = 2
        end = 5
        aggregation = "average"
        result = aggregate_repr(state, start, end, aggregation)
        expected = np.average(state[:, start : end + 1, :], axis=1)
        np.testing.assert_array_almost_equal(result, expected)

    def test_aggregate_repr_empty_slice(self):
        state = np.random.rand(12, 10, 768)
        start = 5
        end = 2  # end < start
        aggregation = "average"
        result = aggregate_repr(state, start, end, aggregation)
        expected = np.zeros((state.shape[0], state.shape[2]))
        np.testing.assert_array_equal(result, expected)

    @patch("transformers.AutoTokenizer.from_pretrained")
    @patch("transformers.AutoModel.from_pretrained")
    def test_extract_sentence_representations(
        self, mock_model_from_pretrained, mock_tokenizer_from_pretrained
    ):
        # Mock model and tokenizer
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = [101, 7592, 1010, 2129, 2024, 2017, 102]
        mock_tokenizer.convert_ids_to_tokens.return_value = [
            "[CLS]",
            "hello",
            ",",
            "how",
            "are",
            "you",
            "[SEP]",
        ]
        mock_tokenizer.all_special_tokens = ["[CLS]", "[SEP]", "[UNK]"]
        mock_tokenizer.unk_token = "[UNK]"

        # Create dummy hidden states
        hidden_size = 768
        sequence_length = 7  # Number of tokens including special tokens
        num_layers = 13  # 12 layers + embeddings
        hidden_states = []
        for _ in range(num_layers):
            # Batch size x seq length x hidden size
            layer_hidden = torch.rand(1, sequence_length, hidden_size)
            hidden_states.append(torch.tensor(layer_hidden))

        mock_model.return_value = (None, None, hidden_states)
        mock_model_from_pretrained.return_value = mock_model
        mock_tokenizer_from_pretrained.return_value = mock_tokenizer

        model, tokenizer = get_model_and_tokenizer(self.model_name, device=self.device)

        final_hidden_states, detokenized = extract_sentence_representations(
            self.sentence,
            model,
            tokenizer,
            device=self.device,
            include_embeddings=True,
            aggregation="last",
            dtype="float32",
            include_special_tokens=False,
            tokenization_counts={},
        )

        # Check the shapes of the outputs
        expected_num_tokens = len(self.sentence.split())
        self.assertEqual(
            final_hidden_states.shape,
            (num_layers, expected_num_tokens, hidden_size),
        )
        self.assertEqual(len(detokenized), expected_num_tokens)

    def test_extract_sentence_representations_empty_sentence(self):
        # Edge case: Empty sentence
        model = MagicMock()
        tokenizer = MagicMock()
        tokenizer.encode.return_value = []
        tokenizer.convert_ids_to_tokens.return_value = []
        tokenizer.all_special_tokens = []
        tokenizer.unk_token = "[UNK]"

        model.return_value = (None, None, [])

        final_hidden_states, detokenized = extract_sentence_representations(
            "",
            model,
            tokenizer,
            device=self.device,
            include_embeddings=True,
            aggregation="last",
            dtype="float32",
            include_special_tokens=False,
            tokenization_counts={},
        )

        self.assertEqual(final_hidden_states.size, 0)
        self.assertEqual(len(detokenized), 0)

    def test_main_function_arguments(self):
        # Test argument parsing and main function integration
        # Since the main function is designed to be run as a script, we can test
        # the argument parsing separately if needed.
        pass  # Implement if necessary

# Run the tests
if __name__ == "__main__":
    unittest.main()
