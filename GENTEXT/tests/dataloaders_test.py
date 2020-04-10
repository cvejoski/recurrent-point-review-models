import unittest

from gentext.data.loaders import DataLoaderBarabasiRandom


class DataLoaderTest(unittest.TestCase):
    def test_barabasi_random(self):
        basic_event_data_loader = {
            "device": "cpu",
            "path_to_data": "./data",
            "path_to_vectors": "./embeddings",
            "batch_size": 32,
            "emb_dim": "barabasi_random.word2vec.1000.30d.txt"}
        data_loader = DataLoaderBarabasiRandom(**basic_event_data_loader)
        self.assertEqual(len(data_loader.train.dataset), 1000)
        self.assertEqual(len(data_loader.validate.dataset), 1000)


if __name__ == '__main__':
    unittest.main()
