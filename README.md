# Fashion MNIST: Clothing Classification (CNN and VGG16) with a Streamlit App

A computer vision learning project: two neural networks for recognizing clothing items from the **Fashion MNIST** dataset, and a **Streamlit** web app where you can upload your own image, get the predicted class with probabilities, and view the training charts of each model.

## Features

- Model selection: **convolutional network (CNN)** or **VGG16**.
- Upload your own image (`jpg`, `jpeg`, `png`) and classify it with one click.
- Display of the predicted class and the probabilities for all 10 classes.
- **Loss** and **accuracy** charts (training and validation sets) for both models.

## Dataset

[Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist): 70,000 black-and-white images of 28×28 pixels, 10 classes:

`T-shirt/top`, `Trouser`, `Pullover`, `Dress`, `Coat`, `Sandal`, `Shirt`, `Sneaker`, `Bag`, `Ankle boot`.

## Models

| Model | Input | Architecture |
|---|---|---|
| **CNN** (`fashion_mnist_model.keras`) | 28×28, grayscale | 3 × `Conv2D` (32, 64, 64 filters, 3×3, ReLU), 2 × `MaxPooling2D`, `Dropout(0.2)` after each convolutional block, `Flatten`, `Dense(64)`, `Dense(10, softmax)` |
| **VGG16** (`vgg16_fashion_mnist_model.keras`) | 32×32, RGB | VGG16 base, `Flatten`, `Dense(256)`, `Dropout(0.3)`, `Dense(10, softmax)` |

Automatic learning rate reduction was used during training (from roughly `1e-3` to `1e-6`).

## Results

Summary of the last epoch based on the saved training history:

| Model | Epochs | Accuracy (training) | Accuracy (validation) | Loss (validation) |
|---|---|---|---|---|
| CNN | 52 | 94.4% | 92.6% | 0.209 |
| VGG16 | 34 | 92.0% | 88.3% | 0.334 |

The training histories are in the files `history_model_fashion.json` (CNN) and `history_conv.json` (VGG16).

## Repository Structure

```
.
├── app.py                           # Streamlit app
├── fashion_mnist_model.keras        # trained CNN
├── vgg16_fashion_mnist_model.keras  # trained VGG16-based model
├── history_model_fashion.json       # CNN training history
├── history_conv.json                # VGG16 training history
└── README.md
```

## Running

1. Clone the repository:

   ```bash
   git clone https://github.com/SHEV-4/fashion-mnist-classifier.git
   cd <repository-name>
   ```

2. (Recommended) create a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate      # Windows: venv\Scripts\activate
   ```

3. Install the dependencies (the models were saved with Keras 3):

   ```bash
   pip install streamlit tensorflow matplotlib pandas streamlit-option-menu pillow numpy
   ```

4. Start the app:

   ```bash
   streamlit run app.py
   ```

   The app will open in your browser.

## How to Use

1. Choose a model in the top menu: **VGG 16** or **Convolutional**.
2. In the sidebar, choose a chart: **Loss functions** or **Accuracy**.
3. Upload an image via **Choose an image...** and click **Test**.
4. Get the predicted class and the probabilities for all classes.

> **Tip.** Fashion MNIST contains clothing items on a dark background. The best results come from images with a single item on a plain background and no extra details.

## Technologies

Python, TensorFlow / Keras, Streamlit, streamlit-option-menu, NumPy, Pandas, Matplotlib, Pillow.

## Author

[SHEV-4](https://github.com/SHEV-4)
