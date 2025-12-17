# 허깅페이스의 무료 tier 로 추론서버를 사용
import os
from huggingface_hub import InferenceClient
from PIL import Image
from dotenv import load_dotenv
load_dotenv()

client = InferenceClient(
    provider="together",
    api_key=os.environ["HF_TOKEN"],
)

# output is a PIL.Image object
image = client.text_to_image(
    "20대 또는 30대 여성이 밝고 화창한 외부에서 즐겁게 데이트하기 위해  파운데이션을 사용하는 모습을 그려줘 ",
    model="black-forest-labs/FLUX.1-dev",
)
image.show()