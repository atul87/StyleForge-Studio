import os
import sys
import time
import torch
from PIL import Image

# Add NST_Code directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from utils.models import VGGEncoder, Decoder
from utils.utils import adaptive_instance_normalization
from torchvision import transforms

def get_process_memory():
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024) # MB
    except ImportError:
        return None

def run_benchmark():
    print("=" * 60)
    print("StyleForge Studio CPU Performance Benchmarks")
    print("=" * 60)
    
    device = torch.device("cpu")
    print(f"Device: {device}")
    
    # 1. Model Loading Benchmark
    start_mem = get_process_memory()
    if start_mem:
        print(f"Initial Memory Usage: {start_mem:.2f} MB")
        
    t0 = time.time()
    vgg_path = os.path.join(parent_dir, 'vgg_normalised.pth')
    decoder_path = os.path.join(parent_dir, 'experiment', 'final_exp', 'decoder_final.pth')
    
    encoder = VGGEncoder(vgg_path).to(device)
    decoder = Decoder().to(device)
    decoder.load_state_dict(torch.load(decoder_path, map_location=device, weights_only=True))
    
    encoder.eval()
    decoder.eval()
    t_load = time.time() - t0
    
    end_mem = get_process_memory()
    print(f"Model Load Time: {t_load:.4f} seconds")
    if start_mem and end_mem:
        print(f"Memory after loading models: {end_mem:.2f} MB (Delta: {end_mem - start_mem:.2f} MB)")

    # 2. Inference Benchmark
    resolutions = [256, 512]
    num_warmup = 2
    num_runs = 5
    
    for res in resolutions:
        print(f"\nBenchmarking Style Transfer at {res}x{res} resolution...")
        
        content_transform = transforms.Compose([
            transforms.Resize((res, res)),
            transforms.ToTensor()
        ])
        
        # Load preset images
        c_image_path = os.path.join(parent_dir, 'content_data', 'brad_pitt.jpg')
        s_image_path = os.path.join(parent_dir, 'style_data', 'sketch.png')
        
        c_img = Image.open(c_image_path).convert('RGB')
        s_img = Image.open(s_image_path).convert('RGB')
        
        c_tensor = content_transform(c_img).unsqueeze(0).to(device)
        s_tensor = content_transform(s_img).unsqueeze(0).to(device)
        
        # Warmup
        for _ in range(num_warmup):
            with torch.no_grad():
                c_feats = encoder(c_tensor, is_test=True)
                s_feats = encoder(s_tensor, is_test=True)
                stylized_feats = adaptive_instance_normalization(c_feats, s_feats)
                _ = decoder(stylized_feats)
                
        # Benchmark Runs
        latencies = []
        for i in range(num_runs):
            t_start = time.time()
            with torch.no_grad():
                c_feats = encoder(c_tensor, is_test=True)
                s_feats = encoder(s_tensor, is_test=True)
                stylized_feats = adaptive_instance_normalization(c_feats, s_feats)
                _ = decoder(stylized_feats)
            t_end = time.time()
            latencies.append(t_end - t_start)
            
        avg_lat = sum(latencies) / len(latencies)
        min_lat = min(latencies)
        max_lat = max(latencies)
        
        print(f"  Runs: {num_runs}")
        print(f"  Average Latency: {avg_lat:.4f} seconds")
        print(f"  Min Latency:     {min_lat:.4f} seconds")
        print(f"  Max Latency:     {max_lat:.4f} seconds")
        
        # CPU FPS equivalent
        fps = 1.0 / avg_lat
        print(f"  Throughput:      {fps:.2f} frames/sec")

    if get_process_memory():
        print(f"\nFinal Memory Usage: {get_process_memory():.2f} MB")
    print("=" * 60)

if __name__ == '__main__':
    run_benchmark()
