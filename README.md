# ML Safety — CARLA Perception Safety Evaluation

This repository contains the implementation and experimental results for the **Introduction to Machine Learning Safety** coursework. The project evaluates three independent binary CARLA perception classifiers:

- `has_pedestrian`
- `has_traffic_light`
- `has_vehicle`

The evaluation covers model training and validation, clean performance, CARLA condition shifts, calibration, temperature scaling, OOD detection, FGSM robustness, ODD k-projection coverage, and additional safety analyses.

---

## 1. Dataset

The experiments use the CARLA RGB dataset with the following splits:

```text
Data2/
├── validation/
│   ├── labels.csv
│   └── rgb-front/
├── test/
│   ├── labels.csv
│   └── rgb-front/
├── test-fog/
│   ├── labels.csv
│   └── rgb-front/
├── test-night/
│   ├── labels.csv
│   └── rgb-front/
└── test-town-01/
    ├── labels.csv
    └── rgb-front/
```

The `CarlaDataset` constructs the image filename from the `frame` column, e.g. `frame=260` → `000260.jpg`.

The final evaluation uses the complete **3,600-image test set**, rather than a 100-image subset.

---

## 2. Environment

Experiments were executed primarily in Google Colab using Google Drive for dataset/model storage.

Main libraries:

```text
Python
PyTorch
Torchvision
Scikit-learn
Pandas
NumPy
Matplotlib
PIL
```

A CUDA runtime was used when available.

---

## 3. Model Architecture

Each task uses a separate pretrained ResNet-18 model with the final fully connected layer replaced by one output logit:

```python
import torch.nn as nn
from torchvision import models

def create_model():
    model = models.resnet18(
        weights=models.ResNet18_Weights.DEFAULT
    )
    model.fc = nn.Linear(
        model.fc.in_features,
        1
    )
    return model
```

Binary predictions are obtained with:

```python
probability = torch.sigmoid(logit)
prediction = int(probability > 0.5)
```

---

## 4. Final Training Configuration

The final models were retrained after correcting the train/validation dataset assignment.

```python
criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.0001
)

batch_size = 32
epochs = 4
```

The three classifiers were trained independently using the correct datasets:

```text
Training → labels_train
Validation → labels_validation
Test → labels_test
```

The test set was not used for model selection.

Final checkpoints:

```text
Data2/model/
├── has_pedestrian_model.pth
├── has_traffic_light_model.pth
└── has_vehicle_model.pth
```

---

## 5. Preprocessing — Important Reproducibility Detail

Two explicit transforms are used in the final pipeline. This prevents the model-inference preprocessing from being accidentally mixed with the FGSM pixel-space attack.

### Model transform

Used for training, validation, clean evaluation, reliability diagrams/ECE, temperature scaling, and normal model inference:

```python
model_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
```

### Attack transform

Used to obtain images in the original `[0,1]` pixel space for FGSM generation:

```python
attack_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
```

The final FGSM pipeline is:

```text
raw image [0,1]
      ↓
FGSM perturbation
      ↓
clip to [0,1]
      ↓
normalize
      ↓
trained ResNet-18
```

This separation is important because the trained ResNet expects normalized inputs, while the FGSM perturbation is bounded in pixel space.

---

## 6. Clean Evaluation

Clean evaluation uses the final checkpoint, the complete test set, `model_transform`, and a 0.5 probability threshold.

Recall is prioritized because false negatives correspond to missed safety-critical objects.

Recall is calculated as:

```text
Recall = TP / (TP + FN)
```

Confusion matrices were also inspected to validate unexpected recall values.

---

## 7. Evaluation Across CARLA Conditions

Each model was evaluated on:

```text
Validation
Test
Test-Fog
Test-Night
Test-Town-01
```

This evaluates sensitivity to weather, lighting, and environment changes.

---

## 8. Final Clean-Test Metrics

The final clean recall values used as the robustness baseline are:

| Model | Recall (clean) | Threshold analysis |
|---|---:|---:|
| Pedestrian | **0.4972** | ≥ 0.90 |
| Traffic light | **0.9853** | ≥ 0.85 |
| Vehicle | **0.8989** | ≥ 0.85 |

The threshold analysis indicates that the traffic-light and vehicle classifiers satisfy the selected recall thresholds, while the pedestrian classifier does not.

---

## 9. Reliability Diagrams and ECE

Reliability diagrams were generated for all three classifiers using `model_transform`.

Final single-model ECE values:

| Model | ECE (single) |
|---|---:|
| Pedestrian | **0.6803** |
| Traffic light | **0.2705** |
| Vehicle | **0.1879** |

The pedestrian classifier shows the largest calibration error.

---

## 10. Temperature Scaling

Temperature scaling was performed by selecting the temperature on the **validation set** and then evaluating calibration on the held-out test set. Both sets use `model_transform`.

For logit `z`:

```text
p_T = sigmoid(z / T)
```

Final ECE results:

| Model | ECE (single) | ECE (temperature-scaled) |
|---|---:|---:|
| Pedestrian | **0.6803** | **0.5306** |
| Traffic light | **0.2705** | **0.2601** |
| Vehicle | **0.1879** | **0.1751** |

Temperature scaling improves ECE for all three classifiers, but the resulting ECE values remain substantially above a strict 0.05 target.

---

## 11. OOD Detection

OOD detection was evaluated using a k-NN detector.

Final reported evidence:

| Model | Detector | AUROC | Met? |
|---|---|---:|:---:|
| Pedestrian | k-NN (4 neighbours) | **0.99352 (All OOD)** | ✓ |

The reported pedestrian detector AUROC values were approximately **0.9871 for fog**, **0.9999 for night**, and **0.9935 overall**. The overall AUROC exceeds the 0.90 criterion used in the evaluation.

---

## 12. FGSM Adversarial Attack

FGSM follows the required formulation:

```text
x_adv = x + ε · sign(∇x L(y, f(x)))
```

The tested perturbation magnitudes are:

```python
epsilons = [0.01, 0.05, 0.10]
```

The perturbation is generated in raw pixel space, clipped to `[0,1]`, and the resulting image is normalized before model inference.

### Perturbation validation

The maximum perturbation was independently checked:

```text
ε = 0.01 → maximum ≈ 0.0100
ε = 0.05 → maximum ≈ 0.0500
ε = 0.10 → maximum ≈ 0.1000
```

Individual positive and negative examples were also inspected to verify decision-boundary crossings.

For the pedestrian classifier at ε = 0.01:

```text
Average clean BCE loss       = 1.0575
Average adversarial BCE loss = 6.1565
```

The increased adversarial loss provides additional evidence that the perturbations successfully increase the model loss.

---

## 13. FGSM Robustness — Exercise 8.5

The complete **3,600-image test set** is used for clean and adversarial evaluation.

Recall drop is calculated as:

```text
Recall Drop = Clean Recall − Adversarial Recall
```

Final reported results at ε = 0.05:

| Model | Recall (clean) | Recall (FGSM) | Recall Drop |
|---|---:|---:|---:|
| Pedestrian | **0.4972** | **0.0042** | **99.16%** |
| Traffic light | **0.9853** | **0.0000** | **100.00%** |
| Vehicle | **0.8989** | **0.1770** | **80.31%** |

These results indicate substantial vulnerability to the tested FGSM perturbation.

Confusion matrices were inspected during the investigation to verify whether the recall reductions were caused by positive samples becoming false negatives rather than by an incorrect recall implementation.

---

## 14. ODD Definition and k-Projection Coverage

For the k-projection experiment, the ODD was represented using three dimensions aligned with the available CARLA scenarios:

```python
odd_description = {
    "weather": ["clear", "fog"],
    "lighting": ["day", "night"],
    "environment": ["standard_town", "town_01"]
}
```

The four evaluated scenario categories cover clear/day, fog/day, clear/night, and clear/day in Town-01 conditions.

The `odd-coverage` implementation was used for `k ∈ {1,2,3}`.

| k | Covered | Total | Coverage |
|---:|---:|---:|---:|
| 1 | 6 | 6 | **100%** |
| 2 | 9 | 12 | **75%** |
| 3 | 4 | 8 | **50%** |

The test set therefore has complete 1-way coverage, but only 75% 2-way and 50% 3-way coverage. Performance in uncovered ODD combinations is consequently not directly verified.

---

## 15. Class Imbalance and Safety-Critical Recall

The CARLA test data contains unequal positive and negative classes. Because accuracy can hide poor detection of the positive safety-critical class, recall and confusion matrices are emphasized.

The final evaluation therefore reports recall alongside accuracy, precision, and F1 rather than relying on accuracy alone.

---

## 16. Cost-Sensitive Threshold Analysis

The baseline classification threshold was 0.5. For safety verification, the following minimum recall criteria were used: pedestrian ≥ 0.90, traffic light ≥ 0.85, and vehicle ≥ 0.85. These values are evaluation criteria rather than the model's classification threshold.
The reported thresholds were:

| Model | Threshold |
|---|---:|
| Pedestrian | **≥ 0.90** |
| Traffic light | **≥ 0.85** |
| Vehicle | **≥ 0.85** |

Threshold selection should be based on validation data and fixed before final test evaluation.

---

## 17. Additional Validation of Unexpected FGSM Results

The initially observed severe FGSM degradation was not accepted without investigation. The following checks were performed:

1. **Confusion matrices** were inspected to identify the source of recall reduction.
2. **Perturbation magnitude** was independently measured and verified to remain bounded by ε.
3. **Positive and negative individual samples** were inspected for decision-boundary crossings.
4. **Average BCE loss** was compared between clean and adversarial inputs.
5. The evaluation pipeline was checked for consistency of model checkpoint, test dataset, threshold, and preprocessing.
6. The final preprocessing was separated into `model_transform` and `attack_transform` to avoid the earlier normalization mismatch.

These checks provide supporting evidence for the reported adversarial degradation.

---

## 18. Reproducibility Workflow

To reproduce the final evaluation:

1. Start a CUDA-enabled Google Colab runtime if available.
2. Mount Google Drive.
3. Place the `Data2` directory at `/content/drive/MyDrive/Data2/`.
4. Verify all dataset splits and their `labels.csv` files.
5. Define `CarlaDataset`.
6. Define `model_transform` and `attack_transform` separately.
7. Define the ResNet-18 `create_model()` function.
8. Train each classifier for **4 epochs** using `labels_train` and validate using `labels_validation`.
9. Save the three model checkpoints.
10. Evaluate clean performance on the complete test set using `model_transform`.
11. Evaluate Validation, Test, Test-Fog, Test-Night, and Test-Town-01.
12. Generate reliability diagrams and calculate ECE using `model_transform`.
13. Select temperature using validation data and calculate calibrated test ECE.
14. Run the OOD detector using the final model/checkpoint setup.
15. Generate FGSM examples for ε = 0.01, 0.05, and 0.10 using `attack_transform`.
16. Verify perturbation bounds and inspect clean/adversarial examples.
17. Evaluate adversarial recall on the complete test set.
18. Calculate recall drops and inspect confusion matrices.
19. Run k-projection coverage for `k = 1, 2, 3`.
20. Perform class-imbalance and threshold analyses.

---

## 19. Final Safety Summary

The final experiments demonstrate that strong clean performance does not by itself establish safety.

- The pedestrian classifier has substantially lower clean recall (**0.4972**) than the traffic-light (**0.9853**) and vehicle (**0.8989**) classifiers.
- FGSM at ε = 0.05 causes severe recall degradation for all three classifiers, with reported recall drops of **99.16%**, **100.00%**, and **80.31%**.
- Temperature scaling improves calibration but does not reduce ECE below the strict target used in the evaluation.
- The k-NN OOD detector shows strong reported separation for the evaluated OOD scenarios, with overall AUROC **0.99352**.
- ODD coverage is complete for 1-way combinations but falls to **75% for 2-way** and **50% for 3-way** combinations.
- Class imbalance makes recall particularly important for safety-critical interpretation.
- Threshold selection should account for asymmetric false-positive and false-negative consequences.

Overall, the evaluation highlights limitations in adversarial robustness, calibration, and higher-order ODD coverage despite strong performance on some clean and OOD metrics.
