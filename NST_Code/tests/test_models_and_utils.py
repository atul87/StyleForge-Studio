import os
import sys
import unittest
import torch
import tempfile
import shutil
from PIL import Image

# Ensure the parent directory (NST_Code) is in the python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from utils.models import VGGEncoder, Decoder
from utils.utils import (
    adaptive_instance_normalization,
    calc_mean_std,
    get_transform,
    ImageFolderDataset
)

class TestModelsAndUtils(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.vgg_path = os.path.join(parent_dir, 'vgg_normalised.pth')
        cls.decoder_path = os.path.join(parent_dir, 'experiment', 'final_exp', 'decoder_final.pth')
        
    def test_calc_mean_std(self):
        # Create a dummy tensor of shape [batch, channels, height, width]
        # values with known mean and variance
        feat = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]]) # shape: [1, 1, 2, 2]
        # Mean should be (1+2+3+4)/4 = 2.5
        # Variance: ((1-2.5)^2 + (2-2.5)^2 + (3-2.5)^2 + (4-2.5)^2)/4 = (2.25 + 0.25 + 0.25 + 2.25)/4 = 5.0/4 = 1.25
        # Std: sqrt(1.25 + 1e-5)
        mean, std = calc_mean_std(feat)
        self.assertEqual(mean.shape, (1, 1, 1, 1))
        self.assertEqual(std.shape, (1, 1, 1, 1))
        self.assertAlmostEqual(mean.item(), 2.5, places=4)
        self.assertAlmostEqual(std.item(), (1.25 + 1e-5)**0.5, places=4)

    def test_adaptive_instance_normalization(self):
        # Test shapes matching
        content_feat = torch.randn(2, 3, 16, 16)
        style_feat = torch.randn(2, 3, 16, 16)
        
        normalized = adaptive_instance_normalization(content_feat, style_feat)
        self.assertEqual(normalized.shape, (2, 3, 16, 16))
        
        # Test if normalized features mean/std matches style mean/std
        c_mean, c_std = calc_mean_std(normalized)
        s_mean, s_std = calc_mean_std(style_feat)
        
        self.assertTrue(torch.allclose(c_mean, s_mean, atol=1e-4))
        self.assertTrue(torch.allclose(c_std, s_std, atol=1e-4))

    def test_get_transform(self):
        transform = get_transform(size=512, crop=True, final_size=256)
        self.assertIsNotNone(transform)
        # Random inputs to check transformation output
        dummy_img = Image.new('RGB', (600, 600), color='red')
        tensor_img = transform(dummy_img)
        self.assertEqual(tensor_img.shape, (3, 256, 256))

        transform_no_crop = get_transform(size=0, crop=False, final_size=128)
        tensor_img_no_crop = transform_no_crop(dummy_img)
        self.assertEqual(tensor_img_no_crop.shape, (3, 128, 128))

    def test_image_folder_dataset(self):
        # Create temp dir with test images
        temp_dir = tempfile.mkdtemp()
        try:
            # Create a couple of images
            for i in range(2):
                img = Image.new('RGB', (100, 100), color='blue')
                img.save(os.path.join(temp_dir, f"test_{i}.jpg"))
            
            # Non-image file to test filtering
            with open(os.path.join(temp_dir, "ignore.txt"), "w") as f:
                f.write("hello")
                
            transform = get_transform(size=0, crop=False, final_size=50)
            dataset = ImageFolderDataset(temp_dir, transform=transform)
            
            self.assertEqual(len(dataset), 2)
            self.assertEqual(dataset[0].shape, (3, 50, 50))
        finally:
            shutil.rmtree(temp_dir)

    def test_vgg_encoder(self):
        if not os.path.exists(self.vgg_path):
            self.skipTest(f"vgg_normalised.pth not found at {self.vgg_path}. Skipping VGG network tests.")
            
        encoder = VGGEncoder(self.vgg_path)
        encoder.eval()
        
        dummy_input = torch.randn(1, 3, 256, 256)
        
        # Test is_test=False (returns h1, h2, h3, h4)
        h1, h2, h3, h4 = encoder(dummy_input, is_test=False)
        self.assertEqual(h1.shape[1], 64)
        self.assertEqual(h2.shape[1], 128)
        self.assertEqual(h3.shape[1], 256)
        self.assertEqual(h4.shape[1], 512)
        
        # Test is_test=True (returns h4)
        h4_only = encoder(dummy_input, is_test=True)
        self.assertEqual(h4_only.shape, h4.shape)

    def test_decoder(self):
        decoder = Decoder()
        decoder.eval()
        
        # Decoder accepts [batch, 512, h, w] and outputs [batch, 3, h*8, w*8]
        dummy_input = torch.randn(1, 512, 32, 32)
        output = decoder(dummy_input)
        self.assertEqual(output.shape, (1, 3, 256, 256))
        
        if os.path.exists(self.decoder_path):
            decoder.load_state_dict(torch.load(self.decoder_path, map_location='cpu', weights_only=True))
            # Test again with loaded state
            output_loaded = decoder(dummy_input)
            self.assertEqual(output_loaded.shape, (1, 3, 256, 256))
        else:
            print("Warning: decoder_final.pth not found. Skipping weight loading test.")

if __name__ == '__main__':
    unittest.main()
