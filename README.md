# Zero-Shot Classification for Change Detection in Satellite Imagery 

This paper investigates the application of zero-shot classification using the Comparative Language-Image Pre-Training (CLIP) model for change detection in satellite imagery. 
Since traditional supervised learning methods require extensive labeled datasets for all classes of objects in satellite images, which are often scarce or unavailable, zero-shot learning offers a promising alternative. 
Through detailed analysis of three different satellite image datasets: LEVIR-CD, DSIFN, and S2Looking, 
the study evaluates the ability of the zero-shot classification model to detect and classify changes without prior exposure to specific target classes and compares it with traditional tree-based supervised learning algorithms. 
These findings highlight the potential of zero-shot learning as a powerful tool for monitoring, managing, and responding to global changes, underscoring its importance for remote sensing and Earth observation applications where fast, 
accurate change detection is critical.

For more details about the CLIP model, visit its GitHub page: [CLIP GitHub](https://github.com/openai/CLIP).

## Datasets
The study evaluates the zero-shot classification model's performance using the following satellite image datasets:
- **LEVIR-CD**: A large-scale building change detection dataset for urban and rural scenes. For more information, visit [LEVIR-CD Dataset](https://chenhao.in/LEVIR/).
- **DSIFN**: A dataset focusing on flood and inundation scenarios. For more details, visit [DSIFN Dataset](https://github.com/GeoZcx/A-deeply-supervised-image-fusion-network-for-change-detection-in-remote-sensing-images/tree/master/dataset).
- **S2Looking**: A diverse dataset that includes various types of land cover changes. For additional information, visit [S2Looking Dataset](https://github.com/S2Looking/Dataset).


## Python Codes and Notebooks:
**CLIPModel.py:** This code contains a function which runs CLIP Model on a image pair and calculates differences between probabilities of given 32 object names. 

**collect_data.py:** It runs CLIP Model for each image pair in given dataset and saves the difference arrays.

**data_augmentation.ipynb:** It augments 0 labeled (non change) images.

**classification.ipynb:** This notebooks a threshold optimization method and some machine learning algorithms which classify images according to output of CLIP Model


