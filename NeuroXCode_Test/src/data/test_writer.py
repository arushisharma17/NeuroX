import unittest
from unittest.mock import MagicMock, patch, mock_open, call
import numpy as np
import json
import h5py
import os
from writer import ActivationsWriter

class TestActivationsWriter(unittest.TestCase):
    def setUp(self):
        # Sample data for testing
        self.sentence_idx = 0
        self.extracted_words = ['Hello', 'world', '!']
        self.num_layers = 2
        self.embedding_size = 4
        self.activations = np.random.rand(
            self.num_layers,
            len(self.extracted_words),
            self.embedding_size
        ).astype('float32')

        # Common arguments
        self.filename_hdf5 = 'test_activations.hdf5'
        self.filename_json = 'test_activations.json'

    def tearDown(self):
        # Clean up created files if any
        if os.path.exists(self.filename_hdf5):
            os.remove(self.filename_hdf5)
        if os.path.exists(self.filename_json):
            os.remove(self.filename_json)
        # Remove any decomposed layer files
        for i in range(self.num_layers):
            hdf5_file = f'test_activations-layer{i}.hdf5'
            json_file = f'test_activations-layer{i}.json'
            if os.path.exists(hdf5_file):
                os.remove(hdf5_file)
            if os.path.exists(json_file):
                os.remove(json_file)

    @patch('h5py.File')
    def test_hdf5_activations_writer(self, mock_h5py_file):
        writer = ActivationsWriter.get_writer(self.filename_hdf5)
        writer.open(self.num_layers)
        writer.write_activations(self.sentence_idx, self.extracted_words, self.activations)
        writer.close()

        # Assert that h5py.File was called correctly
        mock_h5py_file.assert_called_once_with(self.filename_hdf5, 'w')
        # Ensure that datasets were created
        activations_file = mock_h5py_file.return_value
        activations_file.create_dataset.assert_any_call(
            str(self.sentence_idx),
            self.activations.shape,
            dtype='float32',
            data=self.activations
        )
        # Ensure sentence_to_index dataset was created
        activations_file.create_dataset.assert_any_call(
            'sentence_to_index',
            (1,),
            dtype=h5py.special_dtype(vlen=str)
        )
        activations_file.close.assert_called_once()

    @patch('builtins.open', new_callable=mock_open)
    def test_json_activations_writer(self, mock_file):
        writer = ActivationsWriter.get_writer(self.filename_json)
        writer.open(self.num_layers)
        writer.write_activations(self.sentence_idx, self.extracted_words, self.activations)
        writer.close()

        # Assert that open was called correctly
        mock_file.assert_called_once_with(self.filename_json, 'w', encoding='utf-8')
        # Retrieve the file handle
        file_handle = mock_file()
        # Check that data was written to the file
        self.assertTrue(file_handle.write.called)
        # Since data is written as JSON lines, we can check the write calls
        write_calls = file_handle.write.call_args_list
        self.assertEqual(len(write_calls), 1)
        # Parse the JSON content
        written_data = write_calls[0][0][0]
        json_data = json.loads(written_data)
        # Verify the structure of the JSON data
        self.assertEqual(json_data['linex_index'], self.sentence_idx)
        self.assertEqual(len(json_data['features']), len(self.extracted_words))
        for idx, feature in enumerate(json_data['features']):
            self.assertEqual(feature['token'], self.extracted_words[idx])
            self.assertEqual(len(feature['layers']), self.num_layers)
            for layer in feature['layers']:
                self.assertIn('index', layer)
                self.assertIn('values', layer)
                self.assertEqual(len(layer['values']), self.embedding_size)

    @patch('h5py.File')
    def test_hdf5_writer_with_decompose_layers(self, mock_h5py_file):
        writer = ActivationsWriter.get_writer(
            self.filename_hdf5,
            decompose_layers=True
        )
        writer.open(self.num_layers)
        writer.write_activations(self.sentence_idx, self.extracted_words, self.activations)
        writer.close()

        # Assert that h5py.File was called for each layer
        calls = [
            call(f'test_activations-layer{layer_idx}.hdf5', 'w')
            for layer_idx in range(self.num_layers)
        ]
        mock_h5py_file.assert_has_calls(calls, any_order=True)

    @patch('builtins.open', new_callable=mock_open)
    def test_json_writer_with_filter_layers(self, mock_file):
        # Only write layer 1
        filter_layers = '1'
        writer = ActivationsWriter.get_writer(
            self.filename_json,
            filter_layers=filter_layers
        )
        writer.open(self.num_layers)
        writer.write_activations(self.sentence_idx, self.extracted_words, self.activations)
        writer.close()

        # Retrieve the file handle
        file_handle = mock_file()
        # Parse the JSON content
        write_calls = file_handle.write.call_args_list
        written_data = write_calls[0][0][0]
        json_data = json.loads(written_data)
        # Verify that only the specified layer is written
        for feature in json_data['features']:
            self.assertEqual(len(feature['layers']), 1)
            self.assertEqual(feature['layers'][0]['index'], 1)

    def test_invalid_filetype(self):
        with self.assertRaises(NotImplementedError):
            writer = ActivationsWriter.get_writer('activations.unsupported')
            writer.open(self.num_layers)

    def test_hdf5_writer_without_proper_extension(self):
        with self.assertRaises(ValueError):
            writer = ActivationsWriter.get_writer('activations.txt', filetype='hdf5')
            writer.open(self.num_layers)

    def test_json_writer_without_proper_extension(self):
        with self.assertRaises(ValueError):
            writer = ActivationsWriter.get_writer('activations.txt', filetype='json')
            writer.open(self.num_layers)

    @patch('h5py.File')
    def test_hdf5_writer_multiple_sentences(self, mock_h5py_file):
        writer = ActivationsWriter.get_writer(self.filename_hdf5)
        writer.open(self.num_layers)
        # Write multiple sentences
        for idx in range(5):
            writer.write_activations(idx, self.extracted_words, self.activations)
        writer.close()

        # Ensure create_dataset is called for each sentence
        activations_file = mock_h5py_file.return_value
        self.assertEqual(activations_file.create_dataset.call_count, 6)  # 5 sentences + sentence_to_index

    @patch('builtins.open', new_callable=mock_open)
    def test_json_writer_multiple_sentences(self, mock_file):
        writer = ActivationsWriter.get_writer(self.filename_json)
        writer.open(self.num_layers)
        # Write multiple sentences
        for idx in range(5):
            writer.write_activations(idx, self.extracted_words, self.activations)
        writer.close()

        # Check that write is called 5 times
        file_handle = mock_file()
        self.assertEqual(file_handle.write.call_count, 5)

    def test_activations_writer_manager_init(self):
        writer = ActivationsWriter.get_writer(self.filename_hdf5)
        self.assertIsInstance(writer, ActivationsWriter)
        self.assertFalse(writer.decompose_layers)
        self.assertIsNone(writer.filter_layers)

    def test_activations_writer_add_writer_options(self):
        parser = argparse.ArgumentParser()
        ActivationsWriter.add_writer_options(parser)
        args = parser.parse_args([])
        self.assertTrue(hasattr(args, 'output_type'))
        self.assertTrue(hasattr(args, 'decompose_layers'))
        self.assertTrue(hasattr(args, 'filter_layers'))

if __name__ == '__main__':
    unittest.main()
