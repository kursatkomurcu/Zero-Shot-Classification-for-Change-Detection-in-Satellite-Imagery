from CLIPModel import *
import pandas as pd
import os
import ast

torch.cuda.empty_cache()

classes = ["landscape", "forest", "building", "road",
            "vehicle", "bridge", "river", "lake",
            "farmland", "airport", "runway", "ship",
            "railway", "parking lot", "cloud", "wind turbine",
            "stadium", "school", "hospital", "industrial site",
            "park", "beach", "mountain", "glacier",
            "desert", "volcano", "crater", "island",
            "wetland", "quarry", "dam", "residential area"]

image_1_folder = 'augmented_S2Looking/Image1'
image_2_folder = 'augmented_S2Looking/Image2'

results_df = pd.DataFrame(columns=['img', 'differences'])

# SAVING PROCESS FOR WHOLE DATASET
model, preprocess = loadCLIPModel()

for img in os.listdir(image_1_folder):
    img1_path = os.path.join(image_1_folder, img)
    img2_path = os.path.join(image_2_folder, img)

    if os.path.isfile(img1_path) and os.path.isfile(img2_path):
        differences = CLIP(model, preprocess, img1_path, img2_path, classes)
        differences = differences[0]

        results_df = results_df._append({'img': img, 'differences': differences.tolist()}, ignore_index=True)
        print(results_df)

results_df.to_csv('/media/kursat/TOSHIBA EXT16/projects/satellite/YENI/visual-language-model/zero-shot/augmented_S2Looking.csv', index=False)
