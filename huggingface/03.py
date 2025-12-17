# 해당모델에 대해서 access 신청을 하고( 바로 승인 또는 몇시간 또는 며칠뒤에 승인이되는 경우가 있음)
# 허깅페이스에 로그인(브라우져 로그인 X)
# pip install huggingface_hub
    # CLI  : 터미널   huggingface-cli login --> 토큰입력
    # 코드상에서 : 
    # from huggingface_hub import login
    # login(token='hf-token')

import torch
import os
from diffusers import FluxPipeline
from huggingface_hub import login
from dotenv import load_dotenv
load_dotenv()

login(token=os.getenv('HF_TOKEN'))

pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power

prompt = "A cat holding a sign that says hello world"
image = pipe(
    prompt,
    height=1024,
    width=1024,
    guidance_scale=3.5,
    num_inference_steps=50,
    max_sequence_length=512,
    generator=torch.Generator("cpu").manual_seed(0)
).images[0]
image.save("flux-dev.png")
