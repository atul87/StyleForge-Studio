import os
import sys
import torch
import torch.nn as nn

# Add custom target directory to sys.path to find installed ONNX libraries
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
workspace_dir = os.path.dirname(project_dir)
sys.path.insert(0, os.path.join(workspace_dir, 'lib'))

# Append utils directory to path
sys.path.append(os.path.join(project_dir, 'utils'))



from models import VGGEncoder, Decoder

def calc_mean_std(feat, eps=1e-5):
    # Calculate mean and standard deviation for each channel
    size = feat.size()
    batch_size, channels = size[0], size[1]
    feat_flat = feat.reshape(batch_size, channels, -1)
    feat_mean = feat_flat.mean(dim=2).reshape(batch_size, channels, 1, 1)
    feat_var = feat_flat.var(dim=2, unbiased=False) + eps
    feat_std = feat_var.sqrt().reshape(batch_size, channels, 1, 1)
    return feat_mean, feat_std

def adaptive_instance_normalization(content_feat, style_feat):
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)
    normalized_content_feat = (content_feat - content_mean) / content_std
    return normalized_content_feat * style_std + style_mean

class StyleTransferModel(nn.Module):
    def __init__(self, encoder, decoder):
        super(StyleTransferModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, content_image, style_image, alpha):
        content_feat = self.encoder(content_image, is_test=True)
        style_feat = self.encoder(style_image, is_test=True)
        stylized_feat = adaptive_instance_normalization(content_feat, style_feat)
        # Apply alpha blending
        blended_feat = alpha * stylized_feat + (1.0 - alpha) * content_feat
        output_image = self.decoder(blended_feat)
        return output_image

def main():
    print("Loading PyTorch model weights...")
    vgg_path = os.path.join(project_dir, 'vgg_normalised.pth')
    decoder_path = os.path.join(project_dir, 'experiment', 'final_exp', 'decoder_final.pth')

    encoder = VGGEncoder(vgg_path)
    decoder = Decoder()
    decoder.load_state_dict(torch.load(decoder_path, map_location='cpu'))

    encoder.eval()
    decoder.eval()

    # Wrap the combined pipeline
    nst_model = StyleTransferModel(encoder, decoder)
    nst_model.eval()

    # Create dummy inputs
    # Shape: (batch, channels, height, width)
    # Using 512x512 as standard, but we'll use dynamic axes for flexibility
    dummy_content = torch.randn(1, 3, 512, 512)
    dummy_style = torch.randn(1, 3, 512, 512)
    dummy_alpha = torch.tensor([1.0], dtype=torch.float32)

    # Output directory
    static_models_dir = os.path.join(project_dir, 'static', 'models')
    os.makedirs(static_models_dir, exist_ok=True)
    onnx_path = os.path.join(static_models_dir, 'nst.onnx')

    print(f"Exporting model to ONNX format at {onnx_path}...")
    
    torch.onnx.export(
        nst_model,
        (dummy_content, dummy_style, dummy_alpha),
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['content', 'style', 'alpha'],
        output_names=['output'],
        dynamic_axes={
            'content': {0: 'batch_size', 2: 'height', 3: 'width'},
            'style': {0: 'batch_size', 2: 'height', 3: 'width'},
            'output': {0: 'batch_size', 2: 'height', 3: 'width'}
        }
    )

    print("Checking if ONNX model is valid...")
    try:
        import onnx
        model = onnx.load(onnx_path)
        onnx.checker.check_model(model)
        print("ONNX model is valid!")
    except ImportError:
        print("onnx library not installed, skipping onnx.checker validation.")
    except Exception as e:
        print(f"ONNX check failed: {e}")

if __name__ == '__main__':
    main()
