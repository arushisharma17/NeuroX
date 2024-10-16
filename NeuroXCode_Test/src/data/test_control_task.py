import unittest
from control_task import create_sequence_labeling_dataset

class TestControlTask(unittest.TestCase):

    def setUp(self):
        # Sample training tokens
        self.train_tokens = {
            'source': [['word1', 'word2', 'word3'], ['word2', 'word4']],
            'target': [['label1', 'label2', 'label1'], ['label2', 'label3']]
        }
        # Sample dev and test sources
        self.dev_source = [['word1', 'word5'], ['word3', 'word2']]
        self.test_source = [['word6', 'word2'], ['word1', 'word7']]

    def test_create_sequence_labeling_dataset_same_distribution(self):
        result = create_sequence_labeling_dataset(
            self.train_tokens,
            dev_source=self.dev_source,
            test_source=self.test_source,
            case_sensitive=True,
            sample_from='same'
        )
        self.assertEqual(len(result), 3)  # train, dev, test
        train_control = result[0]
        self.assertEqual(len(train_control['source']), len(self.train_tokens['source']))
        self.assertEqual(len(train_control['target']), len(self.train_tokens['source']))
        # Check that control labels are assigned
        self.assertTrue(all(len(s) == len(t) for s, t in zip(train_control['source'], train_control['target'])))

    def test_create_sequence_labeling_dataset_uniform_distribution(self):
        result = create_sequence_labeling_dataset(
            self.train_tokens,
            sample_from='uniform'
        )
        self.assertEqual(len(result), 1)  # Only train data
        train_control = result[0]
        self.assertEqual(len(train_control['source']), len(self.train_tokens['source']))

    def test_case_insensitive(self):
        train_tokens = {
            'source': [['Word1', 'word2', 'WORD3'], ['word2', 'word4']],
            'target': [['label1', 'label2', 'label1'], ['label2', 'label3']]
        }
        result = create_sequence_labeling_dataset(
            train_tokens,
            case_sensitive=False
        )
        word_types = set()
        for sent in train_tokens['source']:
            for tok in sent:
                word_types.add(tok.lower())
        self.assertEqual(len(result[0]['target']), len(train_tokens['source']))
        assigned_labels = set()
        for sent in result[0]['target']:
            for label in sent:
                assigned_labels.add(label)
        self.assertTrue(len(assigned_labels) <= len(word_types))

if __name__ == '__main__':
    unittest.main()
