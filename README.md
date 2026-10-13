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
