# Land Use and Land Cover (LULC) Classification using Deep Learning

## Overview
This project implements Land Use and Land Cover (LULC) classification using deep learning models, specifically VGG16, ResNet50, and EfficientNet. The classification is performed on the EuroSAT dataset, which contains satellite imagery labeled into different land use categories.

## Dataset
**EuroSAT**: The dataset consists of 27,000 images (RGB and multispectral) with 10 classes:
- Annual Crop
- Forest
- Herbaceous Vegetation
- Highway
- Industrial
- Pasture
- Permanent Crop
- Residential
- River
- Sea/Lake

More details about the dataset can be found [here](https://github.com/phelber/EuroSAT).

## Models Used
The following pre-trained convolutional neural networks (CNNs) are fine-tuned for LULC classification:
- **VGG16**: A deep CNN with 16 layers, known for its simplicity and effectiveness in image classification.
- **ResNet50**: A residual network with 50 layers, capable of handling vanishing gradient problems.
- **EfficientNet**: A highly efficient model that balances depth, width, and resolution for optimal performance.

## Implementation Steps
1. **Data Preprocessing**:
   - Resize images to 64x64.
   - Normalize pixel values.
   - Apply data augmentation.
   
2. **Model Training**:
   - Use pre-trained models with ImageNet weights.
   - Fine-tune the models on the EuroSAT dataset.
   - Use categorical cross-entropy loss and Adam optimizer.

3. **Evaluation**:
   - Compare performance using accuracy, precision, recall, and F1-score.
   - Generate confusion matrices to analyze misclassifications.

## Results
- The models are compared based on classification accuracy and computational efficiency.
- ResNet50 and EfficientNet often outperform VGG16 in terms of accuracy.
- EfficientNet provides a good balance between accuracy and computational cost.

## Dependencies
To run this project, install the following dependencies:
```bash
pip install tensorflow keras numpy pandas matplotlib scikit-learn
```

## Usage
Run the training script using:
```bash
python train.py
```
Modify hyperparameters and model selection in the configuration file as needed.

## Acknowledgments
- EuroSAT dataset creators
- TensorFlow and Keras libraries for deep learning
- Pre-trained models from the Keras Applications module

## License
This project is open-source and available under the MIT License.

