ls -laL /home/xhr/.cache/huggingface/hub/models--microsoft--rad-dino/snapshots/*/

grep -rn "rad_dino_path" /data5/xhr/code/PriorRG --include="*.yaml" | head -3
find /data5 /data6 ~ -maxdepth 8 -type d \( -iname "*rad*dino*" -o -iname "*rad-dino*" \) 2>/dev/null
ls ~/.cache/huggingface/hub/ | grep -i dino
/data5/xhr/code/PriorRG/script_github/PriorRG/mimic_cxr/report-generation-gpt2/best_2026_09_29_20/configs/config_2026_09_29_20.yaml:34:rad_dino_path: microsoft/rad-dino
/data5/xhr/code/PriorRG/script_github/PriorRG/mimic_cxr/report-generation-gpt2/best_2026_09_29_15/configs/config_2026_09_29_15.yaml:34:rad_dino_path: /home/miao/data/dataset/checkpoints/microsoft/rad-dino
/data5/xhr/code/PriorRG/script_github/PriorRG/mimic_cxr/report-generation-gpt2/best_2026_09_29_09/configs/config_2026_09_29_09.yaml:34:rad_dino_path: microsoft/rad-dino
/home/xhr/.cache/huggingface/hub/.locks/models--microsoft--rad-dino
/home/xhr/.cache/huggingface/hub/models--microsoft--rad-dino
models--microsoft--rad-dino


编码器解码器的位置：
ENC = "microsoft/BiomedVLP-CXR-BERT-specialized"
DEC = "/data5/xhr/code/PriorRG/dataset/checkpoints/distilbert/distilgpt2"

图像编码器：IMG = "microsoft/rad-dino"
print("tokens:", out.last_hidden_state.shape)   # [1, 1+N, 768]，第 0 个是 CLS，其余是 patch token
print("global:", out.pooler_output.shape)       # [1, 768]
pooler_output：整张图一个向量,[1, 768] 是对整张图的全局特征，每张图只有一个 768 维向量，没有 token 这一维。在 RAD-DINO 里它通常就是 CLS token 经过 layernorm 后的结果
import os
os.environ["HF_HUB_OFFLINE"] = "1"      # 必须在 import transformers 之前

import torch
from PIL import Image
from transformers import AutoModel, AutoImageProcessor

IMG = "microsoft/rad-dino"
proc = AutoImageProcessor.from_pretrained(IMG)
vit = AutoModel.from_pretrained(IMG).eval()

path = "/data5/xhr/code/PriorRG/data-demo/4f4c142-ff4415c6-17466d42-d7531983-33acac69.jpg"          # 换成一张真实的胸片路径
img = Image.open(path).convert("RGB")   # MIMIC 的 jpg 是灰度，要转 RGB
x = proc(images=img, return_tensors="pt")
print("input:", x["pixel_values"].shape)

with torch.no_grad():
    out = vit(**x)

print("tokens:", out.last_hidden_state.shape)   # [1, 1+N, 768]，第 0 个是 CLS，其余是 patch token
print("global:", out.pooler_output.shape)       # [1, 768]
pooler_output：整张图一个向量[1, 768] 是对整张图的全局特征，每张图只有一个 768 维向量，没有 token 这一维。在 RAD-DINO 里它通常就是 CLS token 经过 layernorm 后的结果
