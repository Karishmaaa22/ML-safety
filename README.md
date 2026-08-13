# ML Safety -- CARLA Perception Safety Evaluation

This repository contains the implementation and experimental results for
the **Introduction to Machine Learning Safety** coursework. The project
evaluates three binary CARLA perception classifiers for:

-   `has_pedestrian`
-   `has_traffic_light`
-   `has_vehicle`

The repository covers model training, clean-data evaluation,
calibration, OOD analysis, adversarial robustness using FGSM, ODD
k-projection coverage, and additional safety-oriented investigations.

------------------------------------------------------------------------

## 1. Project Overview

The perception system uses forward-facing CARLA RGB images. Each image
has three binary labels:

-   whether a pedestrian is present,
-   whether a traffic light is present,
-   whether a vehicle is present.

A separate binary classifier is trained for each label using
**ResNet-18**.

The safety evaluation considers:

1.  Model training and validation
2.  Clean test-set performance
3.  Evaluation across different CARLA test conditions
4.  Reliability diagrams and Expected Calibration Error (ECE)
5.  Temperature scaling
6.  OOD detection
7.  FGSM adversarial robustness
8.  ODD k-projection coverage for `k ∈ {1,2,3}`
9.  Additional safety analyses including class imbalance, ODD coverage
    gaps, and threshold selection

------------------------------------------------------------------------

## 2. Dataset Structure

The experiments use the following CARLA dataset splits:

``` text
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

The image filename is generated from the `frame` column:

``` text
frame 0      → 000000.jpg
frame 260    → 000260.jpg
```

The dataset class loads each RGB image and returns the selected binary
label.

------------------------------------------------------------------------

## 3. Environment

The experiments were primarily executed in Google Colab with Google
Drive used for dataset and model storage.

Main libraries:

``` text
Python
PyTorch
Torchvision
Scikit-learn
Pandas
NumPy
Matplotlib
PIL
```

A CUDA-enabled runtime was used when available.

Check the runtime with:

``` python
import torch

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
```

------------------------------------------------------------------------

# 4. Dataset and DataLoader

The CARLA dataset is implemented using a custom PyTorch `Dataset`.

``` python
from torch.utils.data import Dataset
from PIL import Image
import os
import torch

class CarlaDataset(Dataset):

    def __init__(
        self,
        dataframe,
        image_dir,
        label_column,
        transform=None
    ):
        self.dataframe = dataframe
        self.image_dir = image_dir
        self.label_column = label_column
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):

        row = self.dataframe.iloc[idx]

        image_name = f"{row['frame']:06d}.jpg"

        image_path = os.path.join(
            self.image_dir,
            image_name
        )

        image = Image.open(
            image_path
        ).convert("RGB")

        label = torch.tensor(
            row[self.label_column],
            dtype=torch.float32
        )

        if self.transform:
            image = self.transform(image)

        return image, label
```

The image preprocessing used for the ResNet-18 models is:

``` python
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
```

------------------------------------------------------------------------

# 5. Model Architecture

Three independent binary classifiers are trained.

``` python
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

The pretrained ResNet-18 backbone is retained and the final fully
connected layer is replaced with a single output neuron.

The output is a **logit**. Binary predictions are obtained using:

``` python
probability = torch.sigmoid(logit)
prediction = int(probability > 0.5)
```

------------------------------------------------------------------------

# 6. Training

Each of the three classifiers is trained independently.

Tasks:

``` python
tasks = [
    "has_pedestrian",
    "has_traffic_light",
    "has_vehicle"
]
```

Training configuration:

``` python
criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-4,
    weight_decay=1e-4
)

batch_size = 32
epochs = 10
```

The trained models are saved separately:

``` text
Data2/model/
├── has_pedestrian_model.pth
├── has_traffic_light_model.pth
└── has_vehicle_model.pth
```

A saved model can be loaded with:

``` python
model = create_model().to(device)

model.load_state_dict(
    torch.load(
        "/content/drive/MyDrive/Data2/model/has_pedestrian_model.pth",
        map_location=device
    )
)

model.eval()
```

Replace `has_pedestrian` with the appropriate task when loading another
classifier.

------------------------------------------------------------------------

# 7. Clean Evaluation

Clean evaluation is performed using the complete test set rather than a
small random subset.

For binary classification:

``` python
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

def evaluate_model(model, data_loader, device):

    model.eval()

    all_labels = []
    all_predictions = []

    model.to(device)

    with torch.no_grad():

        for images, labels in data_loader:

            images = images.to(device)

            outputs = model(images)

            predictions = (
                torch.sigmoid(outputs) > 0.5
            ).int()

            all_predictions.extend(
                predictions.cpu().numpy().flatten()
            )

            all_labels.extend(
                labels.cpu().numpy().flatten()
            )

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    precision = precision_score(
        all_labels,
        all_predictions,
        zero_division=0
    )

    recall = recall_score(
        all_labels,
        all_predictions,
        zero_division=0
    )

    f1 = f1_score(
        all_labels,
        all_predictions,
        zero_division=0
    )

    return accuracy, precision, recall, f1
```

Recall is particularly important because false negatives correspond to
missed safety-critical objects.

------------------------------------------------------------------------

# 8. Evaluation Across Test Conditions

The test data contains several conditions:

``` text
test
test-fog
test-night
test-town-01
```

Each trained model is evaluated separately on each split.

The resulting metrics are stored in a DataFrame and visualized using
metric-versus-split plots.

This allows performance changes caused by weather, lighting, and
environment changes to be investigated.

------------------------------------------------------------------------

# 9. Confusion Matrix Analysis

Confusion matrices were used to investigate unexpected recall values.

For a binary classifier:

``` text
                 Predicted
                 0       1
Actual 0        TN      FP
Actual 1        FN      TP
```

Recall is:

``` text
Recall = TP / (TP + FN)
```

Confusion matrices were particularly important during the FGSM
investigation because they showed whether a recall of zero was caused by
a metric implementation error or by all positive samples becoming false
negatives.

Example:

``` text
[[  23   993]
 [2584     0]]
```

gives:

``` text
TN = 23
FP = 993
FN = 2584
TP = 0
```

and therefore:

``` text
Recall = 0 / (0 + 2584) = 0
```

------------------------------------------------------------------------

# 10. Calibration and Reliability Diagrams

Model confidence was evaluated using reliability diagrams.

The reliability diagram compares:

``` text
Confidence → Accuracy
```

against the ideal calibration line:

``` text
Accuracy = Confidence
```

The experiments produced reliability diagrams for:

-   `has_pedestrian`
-   `has_traffic_light`
-   `has_vehicle`

The observed ECE values before calibration were:

``` text
Pedestrian       0.7271
Traffic light    0.2690
Vehicle          0.1940
```

The pedestrian model showed the largest calibration error.

------------------------------------------------------------------------

# 11. Temperature Scaling

Temperature scaling was investigated using:

``` python
temperatures = [0.5, 1.0, 2.0]
```

For a model logit `z`:

``` text
p_T = sigmoid(z / T)
```

The temperature changes confidence without changing the underlying model
weights.

Probability distributions were plotted for the different temperature
values.

Temperature selection should be based on validation/calibration data
rather than using the final test set to choose a value.

------------------------------------------------------------------------

# 12. OOD Detection

OOD evaluation was performed to investigate whether the perception
system can distinguish in-distribution data from shifted or
out-of-distribution conditions.

The evaluation included confidence-based/OOD scoring and comparison of
ID and shifted data.

The OOD results should be reproduced from the corresponding OOD
evaluation cells in the notebook, using the same model checkpoints and
dataset splits.

------------------------------------------------------------------------

# 13. FGSM Adversarial Evaluation

FGSM was implemented according to:

``` text
x_adv = x + ε sign(∇x L(x,y))
```

The implementation is:

``` python
import torch
import torch.nn.functional as F

def fgsm_attack(image, epsilon, gradient):

    perturbed = (
        image + epsilon * gradient.sign()
    )

    perturbed = torch.clamp(
        perturbed,
        0,
        1
    )

    return perturbed
```

The adversarial example generation is:

``` python
def generate_adversarial_example(
    model,
    image,
    label,
    epsilon
):

    model.eval()

    image = image.unsqueeze(0).to(device)

    label = torch.tensor(
        [[float(label)]],
        device=device
    )

    image.requires_grad = True

    output = model(image)

    loss = F.binary_cross_entropy_with_logits(
        output,
        label
    )

    model.zero_grad()

    loss.backward()

    gradient = image.grad

    adv_image = fgsm_attack(
        image,
        epsilon,
        gradient
    )

    return adv_image.squeeze(0).detach()
```

The tested perturbation magnitudes were:

``` python
epsilons = [0.01, 0.05, 0.10]
```

The complete test set was used for the final adversarial recall
evaluation.

------------------------------------------------------------------------

# 14. FGSM Pipeline Validation

Because the initial adversarial recall results were unusually severe,
the FGSM pipeline was independently investigated.

The following checks were performed:

### 14.1 Perturbation magnitude

The perturbation was checked using:

``` python
difference = (
    adv_image - image
).abs()

print(
    difference.max().item()
)

print(
    difference.mean().item()
)
```

Observed maximum perturbations were approximately:

``` text
ε = 0.01 → 0.0100
ε = 0.05 → 0.0500
ε = 0.10 → 0.1000
```

confirming that the perturbations remain bounded by ε.

### 14.2 Individual adversarial examples

Correctly classified positive and negative images were inspected.

For example, a pedestrian-positive sample changed from:

``` text
Clean prediction       = 1
Adversarial prediction = 0
```

while a negative sample changed from:

``` text
Clean prediction       = 0
Adversarial prediction = 1
```

### 14.3 Loss increase

For the pedestrian classifier at ε = 0.01:

``` text
Average clean loss       = 1.0575
Average adversarial loss = 6.1565
```

The substantial increase in loss provides additional evidence that the
FGSM perturbation is successfully increasing the classification loss.

------------------------------------------------------------------------

# 15. Final FGSM Recall Results

The full 3,600-image test set was used.

### Clean recall

``` text
Pedestrian       0.0666
Traffic light    0.9652
Vehicle          0.2341
```

### Adversarial recall

  Model             Clean Recall   ε=0.01   ε=0.05   ε=0.10
  --------------- -------------- -------- -------- --------
  Pedestrian              0.0666   0.0000   0.0000   0.0014
  Traffic light           0.9652   0.0000   0.0000   0.0000
  Vehicle                 0.2341   0.0007   0.0000   0.0000

Corresponding absolute recall drops:

  Model             ε=0.01   ε=0.05   ε=0.10
  --------------- -------- -------- --------
  Pedestrian        0.0666   0.0666   0.0652
  Traffic light     0.9652   0.9652   0.9652
  Vehicle           0.2333   0.2341   0.2341

These results indicate substantial vulnerability to FGSM perturbations.

------------------------------------------------------------------------

# 16. ODD Definition for k-Projection Coverage

For the k-projection experiment, the ODD was represented using three
dimensions that correspond directly to the available CARLA test
scenarios:

``` python
odd_description = {
    "weather": ["clear", "fog"],
    "lighting": ["day", "night"],
    "environment": ["standard_town", "town_01"]
}
```

The four test scenario categories were represented as:

``` python
test_scenarios = [
    {
        "weather": "clear",
        "lighting": "day",
        "environment": "standard_town"
    },
    {
        "weather": "fog",
        "lighting": "day",
        "environment": "standard_town"
    },
    {
        "weather": "clear",
        "lighting": "night",
        "environment": "standard_town"
    },
    {
        "weather": "clear",
        "lighting": "day",
        "environment": "town_01"
    }
]
```

------------------------------------------------------------------------

# 17. k-Projection Coverage

The `odd-coverage` implementation is used to calculate coverage for:

``` text
k = 1
k = 2
k = 3
```

The resulting coverage is:

    k   Covered   Total   Coverage
  --- --------- ------- ----------
    1         6       6   **100%**
    2         9      12    **75%**
    3         4       8    **50%**

The results show that all individual ODD conditions are represented,
while coverage decreases for higher-order combinations. Several
combinations therefore remain unverified.

------------------------------------------------------------------------

# 18. Additional Safety Analyses

## 18.1 Class imbalance

The test set contains unequal proportions of positive and negative
samples.

For example, the pedestrian test set contains:

``` text
Positive samples = 706
Negative samples = 2894
```

Therefore, accuracy alone is not an adequate safety metric. Recall is
prioritized because false negatives correspond to missed safety-critical
objects.

Clean-test recalls were:

``` text
Pedestrian       0.0666
Traffic light    0.9652
Vehicle          0.2341
```

------------------------------------------------------------------------

## 18.2 ODD coverage gaps

The k-projection results demonstrate:

``` text
1-way coverage = 100%
2-way coverage = 75%
3-way coverage = 50%
```

Therefore, complete 1-way coverage does not imply that higher-order
interactions between operating conditions have been tested.

Performance in the uncovered combinations should not be assumed without
additional testing.

------------------------------------------------------------------------

## 18.3 Cost-sensitive threshold

The baseline classifier uses:

``` python
prediction = int(probability > 0.5)
```

From a safety perspective, a fixed threshold of 0.5 is not necessarily
optimal because false positives and false negatives have asymmetric
consequences.

A cost-optimal threshold `τ*` should be selected using validation data
and a safety-relevant cost function that penalizes critical false
negatives appropriately. The selected threshold should then be fixed
before final test evaluation.

------------------------------------------------------------------------

# 19. Reproducibility Checklist

To reproduce the complete evaluation:

1.  Start a CUDA-enabled Google Colab runtime if GPU evaluation is
    desired.
2.  Mount Google Drive.
3.  Place the CARLA `Data2` directory at:

``` text
/content/drive/MyDrive/Data2/
```

4.  Verify the five dataset splits:
    -   `validation`
    -   `test`
    -   `test-fog`
    -   `test-night`
    -   `test-town-01`
5.  Define the `CarlaDataset` class.
6.  Define the image transformation.
7.  Define `create_model()`.
8.  Train the three classifiers or load the saved checkpoints.
9.  Run clean evaluation.
10. Run evaluation across the available test splits.
11. Generate reliability diagrams and calculate ECE.
12. Run temperature-scaling analysis.
13. Run the OOD evaluation using the corresponding notebook cells.
14. Run the FGSM implementation with ε = 0.01, 0.05 and 0.10.
15. Validate FGSM perturbation magnitude and loss increase.
16. Evaluate adversarial recall using the complete test set.
17. Download/import `kprojection.py` from the `odd-coverage` repository.
18. Define the three-dimensional ODD.
19. Compute k-projection coverage for `k = 1, 2, 3`.
20. Use the resulting evidence in the safety-case report.

------------------------------------------------------------------------

# 20. Main Safety Findings

The experiments demonstrate several important safety concerns:

-   Clean performance differs substantially between the three perception
    classifiers.
-   The pedestrian and vehicle classifiers have low clean recall.
-   The traffic-light classifier has high clean recall but loses all
    positive-class recall under FGSM.
-   FGSM perturbations were independently validated through perturbation
    bounds, confusion matrices, individual examples, and loss increase.
-   Calibration is imperfect for all three models, with the largest ECE
    observed for the pedestrian classifier.
-   ODD coverage decreases from 100% at 1-way interactions to 50% at
    3-way interactions.
-   Higher-order ODD combinations remain unverified.
-   A fixed probability threshold of 0.5 does not explicitly account for
    the asymmetric safety consequences of false positives and false
    negatives.

These results demonstrate that **high clean-data performance alone is
insufficient to establish ML safety**. Robustness, calibration, ODD
coverage, and system-level handling of model uncertainty must also be
considered.
