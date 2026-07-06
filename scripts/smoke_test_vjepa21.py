
import time

import torch



torch.set_float32_matmul_precision("high")



device = torch.device("cuda:0")



print("Loading V-JEPA 2.1 ViT-B/16...")

encoder, predictor = torch.hub.load(

    ".",

    "vjepa2_1_vit_base_384",

    source="local",

    pretrained=True,

)



encoder = encoder.to(device).eval()

predictor = predictor.to(device).eval()



video = torch.rand(1, 3, 8, 384, 384, device=device)



torch.cuda.reset_peak_memory_stats(device)

torch.cuda.synchronize()

start_time = time.perf_counter()



with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):

    tokens = encoder(video)



torch.cuda.synchronize()

elapsed = time.perf_counter() - start_time



print()

print("Model loading and encoder forward: success")

print("Encoder type:", type(encoder).__name__)

print("Predictor type:", type(predictor).__name__)

print("Input shape:", tuple(video.shape))

print("Output token shape:", tuple(tokens.shape))

print("Output dtype:", tokens.dtype)

print("Elapsed seconds:", round(elapsed, 3))

print("Peak GPU memory MB:", round(torch.cuda.max_memory_allocated(device) / 1024**2, 1))

