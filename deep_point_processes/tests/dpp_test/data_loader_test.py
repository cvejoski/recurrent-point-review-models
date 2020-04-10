import unittest
from dpp.data.loaders import BasicEventDataLoader


class DataLoaderTest(unittest.TestCase):
    def test_apple_full_dataloader(self):
        basic_event_data_loader = {
            "data_path": "./data/OrderBook/amazon_full",
            "batch_size": 32,
            "bptt_size": 100,
            "num_workers": 1}
        data_loader = BasicEventDataLoader(**basic_event_data_loader)
        self.assertEqual(len(data_loader.train.dataset), 24)
        self.assertEqual(len(data_loader.validate.dataset), 1)


if __name__ == '__main__':
    unittest.main()
