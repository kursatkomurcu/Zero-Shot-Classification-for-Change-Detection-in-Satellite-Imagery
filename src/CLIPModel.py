import torch
import clip
from PIL import Image
import numpy as np

classes = ["landscape", "forest", "building", "road",
            "vehicle", "bridge", "river", "lake",
            "farmland", "airport", "runway", "ship",
            "railway", "parking lot", "cloud", "wind turbine",
            "stadium", "school", "hospital", "industrial site",
            "park", "beach", "mountain", "glacier",
            "desert", "volcano", "crater", "island",
            "wetland", "quarry", "dam", "residential area"]

def loadCLIPModel(pretrainedModel="ViT-B/32", device="cpu"):
    model, preprocess = clip.load(pretrainedModel, device=device)
    return model, preprocess

def CLIP(model, preprocess, img1_path, img2_path, classes, device="cpu"):
    image_1 = preprocess(Image.open(img1_path)).unsqueeze(0).to(device)
    image_2 = preprocess(Image.open(img2_path)).unsqueeze(0).to(device) 

    text_tokens = clip.tokenize(classes).to(device)

    with torch.no_grad():
        logits_per_image_1 = model(image_1, text_tokens)[0]
        probs_1 = logits_per_image_1.softmax(dim=-1).cpu().numpy()

        logits_per_image_2 = model(image_2, text_tokens)[0] 
        probs_2 = logits_per_image_2.softmax(dim=-1).cpu().numpy()

    difference = np.abs(probs_2 - probs_1)

    return difference
