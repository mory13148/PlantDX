# 🌱 PlantDx

**PlantDx** is a deep learning application for detecting plant diseases from leaf images.

It uses **EfficientNet-B0** fine-tuned on the **PlantVillage** dataset and provides disease information and treatment recommendations in French.

## 🚀 Live Demo

👉 [**Try PlantDx online**](https://plantdx.streamlit.app/)

## 🚀 Features

* 🌿 Plant disease classification
* 🧠 EfficientNet-B0
* 🔄 Test-Time Augmentation (TTA)
* 🎯 Confidence-based `UNKNOWN` detection
* 🇫🇷 Disease information in French
* 💊 Treatment recommendations
* 🖥️ Streamlit web application
* ☁️ Model hosted on Hugging Face Hub

## 📊 Results

| Metric              |        Result |
| ------------------- | ------------: |
| Validation Accuracy |    **96.45%** |
| Test Accuracy       |    **96.67%** |
| Classes             |        **38** |
| Input Size          | **224 × 224** |

## 🧠 Model

PlantDx uses **EfficientNet-B0**, pretrained on ImageNet and fine-tuned for **38 plant disease classes**.

Test-Time Augmentation is applied using:

* Original image
* Horizontal flip
* +8° rotation
* -8° rotation

The predictions from these transformations are averaged to obtain the final prediction.

If the confidence is below the configured threshold, the result is classified as `UNKNOWN`.

The trained model is hosted on **Hugging Face Hub** and downloaded automatically when the application starts.

## 🌿 Supported Plants

PlantDx supports diseases affecting:

* Apple
* Blueberry
* Cherry
* Corn
* Grape
* Orange
* Peach
* Pepper
* Potato
* Raspberry
* Soybean
* Squash
* Strawberry
* Tomato

## 🛠️ Tech Stack

* Python
* PyTorch
* Torchvision
* EfficientNet-B0
* Streamlit
* Pillow
* NumPy
* Hugging Face Hub

## 📁 Project Structure

```text
PlantDX/
│
├── app/
│   └── app.py
│
├── src/
│   ├── inference.py
│   └── diseases.json
│
├── notebooks/
│
├── .gitignore
├── README.md
└── requirements.txt
```

The PlantVillage dataset and model weights are not stored in the GitHub repository.

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/mory13148/PlantDX.git
cd PlantDX
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

## ▶️ Run Locally

```bash
streamlit run app/app.py
```

The application will open in your browser.

## 📷 How It Works

1. Upload an image of a plant leaf.
2. Click **Analyser la feuille**.
3. The image is processed by the EfficientNet-B0 model.
4. PlantDx predicts the disease.
5. The confidence score is displayed.
6. Low-confidence predictions are classified as `UNKNOWN`.
7. Disease information and treatment recommendations are displayed in French.

## 📚 Dataset

This project uses the **PlantVillage** dataset.

Reference:

> Mohanty, S. P., Hughes, D. P., & Salathé, M. (2016). Using Deep Learning for Image-Based Plant Disease Detection.

## ⚠️ Disclaimer

PlantDx is an educational and portfolio project.

Predictions should not be considered a substitute for professional agricultural diagnosis.

## 👨‍💻 Author

**Mory Adama DEMBELE**

Machine Learning • Data Science • Artificial Intelligence

📍 Bamako, Mali

---

⭐ If you find this project interesting, feel free to explore the repository.
