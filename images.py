from datasets import load_dataset
from pathlib import Path
from PIL import Image

import torch
from diffusers import Flux2KleinPipeline

import numpy as np
import json


# 1. Загрузка датасета и распаковка файлов
ds = load_dataset("NarchAI1992/Roof_house", cache_dir=r"E:\Projects\sber_de\dataset")

input_dir = Path(r"E:\Projects\sber_de\input_images")
output_dir = Path(r"E:\Projects\sber_de\output_images")
input_dir.mkdir(parents=True, exist_ok=True)
output_dir.mkdir(parents=True, exist_ok=True)

for i, picture in enumerate(ds["train"]):
    img = picture["image"]
    img.thumbnail((256, 256), Image.Resampling.LANCZOS)
    input_path = input_dir / f"image_{i}.png"
    img.save(input_path)


# 2. Подготовка промптов
prompt_basket = [
    "Add a giant bird in the foreground",
    "Add a blue balloon coming from a door",
    "Change color of the main object to purple",
    "Add two happy children playing in the foreground",
    "Add a slogan related to the content of the picture",
    "Add a broken glass effect to the picture",
    "Repaint the picture in the style of Malevich",
    "Turn every object in a picture into a fruit",
    "Add a rain effect to the picture",
    "Add an anime effect to the picture"
    ]

prompt_basket = [p + ". Remove the watermarks from the picture and any contact information." for p in prompt_basket]


# 3. Загрузка модели и запуск пайплайна обработки файла
pipe = Flux2KleinPipeline.from_pretrained("black-forest-labs/FLUX.2-klein-base-4B", torch_dtype=torch.bfloat16, cache_dir="E:\\huggingface_cache")
pipe.enable_model_cpu_offload()  

results = []

for file in input_dir.iterdir():
    if file.is_file():
        rand_num = np.random.randint(len(prompt_basket))
        prompt = prompt_basket[rand_num]

        input_image = Image.open(file).convert("RGB")

        image = pipe(
            prompt=prompt,
            image=input_image,
            guidance_scale=4.0,
            num_inference_steps=20,
            generator=torch.Generator(device="cuda").manual_seed(0)
        ).images[0]
        
        image.save(output_dir / f"{file.stem}_upd.png")

        data_json = {
            "input_id": file.name,
            "prompt": prompt,
            "output_id": f"{file.stem}_upd.png"
        }

        results.append(data_json)

# 4. Сохранение результатов в json
with open("final_result.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4, ensure_ascii=False)


# 5. Техническая валидация
errors = []

input_ids, output_ids = set(), set()

for i, res in enumerate(results):
    input_id = res.get("input_id")
    output_id = res.get("output_id")
    prompt = res.get("prompt")

    if not input_id:
        errors.append(f"JSON record {i}: missing input_id")
    if not output_id:
        errors.append(f"JSON record {i}: missing output_id")
    if not prompt:
        errors.append(f"JSON record {i}: missing prompt")

    if input_id:
        if input_id in input_ids:
            errors.append(f"Duplicate input_id: {input_id}")
        input_ids.add(input_id)
    if output_id:    
        if output_id in output_ids:
            errors.append(f"Duplicate output_id: {output_id}")
        output_ids.add(output_id)

    if input_id and output_id:
        input_path = input_dir / input_id
        output_path = output_dir / output_id

        if not input_path.exists():
            errors.append(f"Missing input file: {input_path}")
        else:
            try:
                with Image.open(input_path) as img:
                    if img.size[0] == 0 or img.size[1] == 0:
                        errors.append(f"Invalid input image size: {input_path}")
            except Exception as e:
                errors.append(f"Cannot open input image {input_path}: {e}")

        if not output_path.exists():
            errors.append(f"Missing output file: {output_path}")
        else:
            try:
                with Image.open(output_path) as img:
                    if img.size[0] == 0 or img.size[1] == 0:
                        errors.append(f"Invalid output image size: {output_path}")
            except Exception as e:
                errors.append(f"Cannot open output image {output_path}: {e}")

print(f"Checked images: {len(results)}")

if errors:
    print("Validation failed:")
    for err in errors:
        print(f"- {err};")
else:
    print("Validation passed")