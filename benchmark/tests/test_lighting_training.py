import unittest

import numpy as np

from benchmark.segmentation.lighting.train_v2 import augment_light, validation_transform


class LightingTrainingTest(unittest.TestCase):
    def test_progressive_augmentation_is_deterministic(self):
        image = np.full((32, 32, 3), 128, dtype=np.uint8)
        first, first_name = augment_light(image, np.random.default_rng(42), "progressive", 0.75)
        second, second_name = augment_light(image, np.random.default_rng(42), "progressive", 0.75)
        self.assertEqual(first_name, second_name)
        self.assertTrue(np.array_equal(first, second))
        self.assertEqual(first.dtype, np.uint8)

    def test_validation_lighting_preserves_shape(self):
        image = np.full((24, 40, 3), 128, dtype=np.uint8)
        for condition in ("normal", "dark_3x", "dark_10x", "low_light_noise", "overexposed"):
            transformed = validation_transform(image, condition, 7)
            self.assertEqual(transformed.shape, image.shape)
            self.assertEqual(transformed.dtype, np.uint8)


if __name__ == "__main__":
    unittest.main()
