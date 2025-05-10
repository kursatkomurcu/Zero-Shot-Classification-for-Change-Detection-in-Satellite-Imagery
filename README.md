# Zero-Shot Classification for Change Detection in Satellite Imagery 

[![Buy Me A Coffee](https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png)](https://www.buymeacoffee.com/kursatkomurcu)

This paper investigates the application of zero-shot classification using the Comparative Language-Image Pre-Training (CLIP) model for change detection in satellite imagery. 
Since traditional supervised learning methods require extensive labeled datasets for all classes of objects in satellite images, which are often scarce or unavailable, zero-shot learning offers a promising alternative. 
Through detailed analysis of three different satellite image datasets: LEVIR-CD, DSIFN, and S2Looking, 
the study evaluates the ability of the zero-shot classification model to detect and classify changes without prior exposure to specific target classes and compares it with traditional tree-based supervised learning algorithms. 
These findings highlight the potential of zero-shot learning as a powerful tool for monitoring, managing, and responding to global changes, underscoring its importance for remote sensing and Earth observation applications where fast, 
accurate change detection is critical.

For more details about the CLIP model, visit its GitHub page: [CLIP GitHub](https://github.com/openai/CLIP).

## Citiation

```
@INPROCEEDINGS{10586705,
  author={Kömürcü, Kürşat and Petkevičius, Linas},
  booktitle={2024 IEEE 11th Workshop on Advances in Information, Electronic and Electrical Engineering (AIEEE)}, 
  title={Zero Shot Classification for Change Detection in Satellite Imagery}, 
  year={2024},
  volume={},
  number={},
  pages={1-6},
  keywords={Electrical engineering;Earth;Analytical models;Zero-shot learning;Conferences;Supervised learning;Satellite images;Zero Shot Classification;CLIP Model;Satellite Image Analysis;Change Detection},
```

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

![dataset](https://github.com/kursatkomurcu/Zero-Shot-Classification-for-Change-Detection-in-Satellite-Imaginary/blob/main/imgs/dataset.png)

## Algorithm Comparison

|                          | **LEVIR-CD** |       |       |       | **DSIFN** |       |       |       | **S2Looking** |       |       |       | **Average** |       |       |       |
|--------------------------|:------------:|:-----:|:-----:|:-----:|:---------:|:-----:|:-----:|:-----:|:-------------:|:-----:|:-----:|:-----:|:-----------:|:-----:|:-----:|:-----:|
|                          |     Acc      |  F1   | Rec   | Prec  |    Acc    |  F1   | Rec   | Prec  |      Acc      |  F1   | Rec   | Prec  |    Acc      |  F1   | Rec   | Prec  |
| **Threshold Optimization** |    0.9369    | 0.9439| 1.0   | 0.9496|    0.9648  | 0.9669| 1.0   | 0.9664|     0.9507     | 0.9535| 1.0   | 0.9985|    0.9508    | 0.9547| 1.0   | 0.9715|
| **Decision Tree**          |    0.875     | 0.8703| 0.8468| 0.8952|    0.9606  | 0.9609| 0.9570| 0.9647|     0.9374     | 0.9419| 0.9428| 0.9410|    0.9244    | 0.9243| 0.9155| 0.9336|
| **Random Forest**          |    0.9464    | 0.9473| 0.9729| 0.9230|    0.9757  | 0.9764| 0.9950| 0.9586|     0.9697     | 0.9718| 0.9695| 0.9741|    0.9639    | 0.9651| 0.9791| 0.9519|
| **Gradient Boosting**      |    0.9508    | 0.9502| 0.9459| 0.9545|    0.9732  | 0.9739| 0.9870| 0.9611|     0.9723     | 0.9741| 0.9676| 0.9806|    0.9654    | 0.9660| 0.9668| 0.9654|
| **XGBoost**                |    0.9553    | 0.9553| 0.9639| 0.9469|    0.9782  | 0.9788| 0.9930| 0.9650|     0.9743     | 0.9760| 0.9714| 0.9807|    0.9692    | 0.97  | 0.9761| 0.9642|


|                          | **LEVIR-CD** |       |       |       | **DSIFN** |       |       |       | **S2Looking** |       |       |       | **Average** |       |       |       |
|--------------------------|:------------:|:-----:|:-----:|:-----:|:---------:|:-----:|:-----:|:-----:|:-------------:|:-----:|:-----:|:-----:|:-----------:|:-----:|:-----:|:-----:|
|                          |     Acc      |  F1   | Rec   | Prec  |    Acc    |  F1   | Rec   | Prec  |      Acc      |  F1   | Rec   | Prec  |    Acc      |  F1   | Rec   | Prec  |
| **Threshold Optimization** |    0.9375    | 0.9669| 1.0   | 0.9767|    0.9352  | 0.9665| 1.0   | 0.9642|     0.94       | 0.9690| 1.0   | 0.9960|    0.9375    | 0.9674| 1.0   | 0.9789|
| **Decision Tree**          |    0.8593    | 0.9203| 0.8888| 0.9541|    0.9294  | 0.9623| 0.9504| 0.9746|     0.947      | 0.9727| 0.9488| 0.9978|    0.9119    | 0.9517| 0.9293| 0.9755|
| **Random Forest**          |    0.9296    | 0.9620| 0.9743| 0.95  |    0.9617  | 0.9802| 0.9969| 0.9640|     0.971      | 0.9852| 0.9729| 0.9979|    0.9541    | 0.9758| 0.9813| 0.9706|
| **Gradient Boosting**      |    0.8984    | 0.9432| 0.9230| 0.9642|    0.9588  | 0.9785| 0.9907| 0.9667|     0.974      | 0.9868| 0.9759| 0.9979|    0.9437    | 0.9695| 0.9632| 0.9762|
| **XGBoost**                |    0.9296    | 0.9613| 0.9572| 0.9655|    0.9764  | 0.9877| 0.9969| 0.9787|     0.978      | 0.9888| 0.9799| 0.9979|    0.9613    | 0.9792| 0.978 | 0.9807|


![flowchart](https://github.com/kursatkomurcu/Zero-Shot-Classification-for-Change-Detection-in-Satellite-Imaginary/blob/main/imgs/clip.png)
Clip Model Diagram

![histogram](https://github.com/kursatkomurcu/Zero-Shot-Classification-for-Change-Detection-in-Satellite-Imaginary/blob/main/imgs/histogram.png)
Histogram of the datasets and optimized thresholds

