# Hands_and_Health
Exploration of machine learning in healthcare.

# Hand Health Prediction: Non-Invasive Color Analysis Using Machine Learning

![Project Banner](https://github.com/AKaradsheh/Hands_and_Health/blob/62b402ba2269acbd570a3b0f6f87eb7584e73673/images_and_graphs/hand_marked2.png)
*Figure: Example hand image with midpoints marked for RGB sampling (green circles). Healthy hands show brighter tones.*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Made with Jupyter](https://img.shields.io/badge/Made%20with-Jupyter-orange.svg)](https://jupyter.org/)

## Overview

This Capstone project develops a machine learning pipeline to classify hand health as "healthy" (pink tones indicating good circulation) or "unhealthy" (bluish hues signaling potential issues like hypoxia or poor blood flow). Using non-invasive RGB color analysis from hand images, the model enables early detection for at-risk individuals (e.g., cardiovascular/respiratory conditions). 

Key highlights:
- **Data Story**: Processed ~127 images per radius (1, 3, 5 pixels) from healthy/unhealthy folders, extracting RGB at finger midpoints via MediaPipe. Healthy hands are ~15-30 points brighter (esp. green channel), confirmed by t-tests (p < 0.001).
- **Best Model**: Random Forest (RF) at radius 5 achieves **92% test accuracy** and **83% CV**—outperforming Logistic Regression (LR), Regularized LR, and Neural Network (NN).
- **Impact**: Potential mobile app integration for remote screening, reducing healthcare burdens.

Repo Structure:
- `data/`: CSVs with RGB features (e.g., `hand_color_data_midpoints_radius_5.csv`).
- `notebooks/`: Jupyter files for data processing (`photos_marking1.ipynb`), EDA (`EDA_radii135_datasets.ipynb`), modeling (`Hands_ML{1-4}.ipynb`).
- `images_and_graphs/`: Visuals (e.g., marked hands, PCA plots, accuracy graphs).
- `model_comparison.csv`: Metrics file for all models/radii.

## Installation

1. Clone the repo:
   ```
   git clone https://github.com/yourusername/hand-health-prediction.git
   cd hand-health-prediction
   ```

2. Set up environment (Python 3.8+):
   ```
   python -m venv env
   source env/bin/activate  # On Windows: env\\Scripts\\activate
   pip install -r requirements.txt
   ```
   *requirements.txt:*
   ```
   mediapipe
   opencv-python
   pandas
   numpy
   scikit-learn
   tensorflow
   matplotlib
   seaborn
   ```

3. Download datasets/images (if not included due to size/privacy).

## Usage

1. **Data Processing**: Run `photos_marking1.ipynb` to generate marked images and CSVs.
   ```
   jupyter notebook photos_marking1.ipynb
   ```

2. **EDA**: Explore in `EDA_radii135_datasets.ipynb` (e.g., pairplots, correlations).

3. **Modeling**: Train/evaluate in `Hands_ML3.ipynb` (RF focus).
   - Load CSV: `df = pd.read_csv('../data/hand_color_data_midpoints_radius_5.csv')`
   - Train RF: Achieves 92% accuracy.
   - Metrics: View `model_comparison.csv`.

4. **Prediction**: Example script for new images:
   ```python
   import pandas as pd
   from sklearn.ensemble import RandomForestClassifier
   from joblib import load  # Assuming model saved as .joblib

   model = load('rf_model_radius5.joblib')
   new_data = pd.DataFrame(...)  # Extract RGB from new image
   prediction = model.predict(new_data)
   print("Healthy" if prediction[0] == 1 else "Unhealthy")
   ```

## Data

- **Sources**: Healthy (~70 images) from stock sites (Pexels/Unsplash); Unhealthy (~57) from clinic (bluish tones).
- **Processing**: MediaPipe detects landmarks; RGB averaged at midpoints (thumb: 2 segments; others: 3) with radii 1/3/5. Imputed NaNs; ~84 features.
- **Stats**: Healthy brighter (e.g., Avg_G ~160 vs. ~140); t-tests significant (p < 0.001).
- **Balance**: ~58% unhealthy; stratified splits used.


## Models and Results

Four models trained per radius (PCA: 20 components, ~97% variance):

- **Random Forest (Best)**: 92% test accuracy; CV rises to 83% at R5. Low errors (1 FN)—prioritizes unhealthy detection.
- Others: Reg LR (85-88% test), NN (81-88% test, variable), Initial LR (81-88% test, unstable).

Metrics (`model_comparison.csv` averages):

| Model                  | Radius | Test Acc | CV Acc (± std) |
|------------------------|--------|----------|----------------|
| Initial LR             | 5      | 0.81     | 0.71 (±0.11)  |
| Regularized LR         | 5      | 0.85     | 0.72 (±0.10)  |
| **Random Forest**      | **5**  | **0.92** | **0.83 (±0.12)** |
| Neural Network         | 5      | 0.83     | 0.76 (±0.06)  |




Predictions vs. Actual: RF aligns well (92% correct on test; e.g., R5: 14/15 unhealthy true negatives). Errors rare, often borderlines.

## Contributors

- Adli Karadsheh (Developer)

## License

MIT License. See [LICENSE](LICENSE) for details.
