# Scene Recognition with Deep CNNs

An end-to-end PyTorch application for indoor scene classification on the MIT Indoor67 dataset. This project explores multiple deep learning approaches for indoor scene recognition and deploys the final distilled model through an interactive Streamlit application for real-time inference.

---

## Technologies

- Python
- PyTorch
- Torchvision
- Streamlit
- NumPy
- Pandas
- scikit-learn

---
## 🎥 Live Demo

<p align="center">

<a href="tests/results/demo.mp4">
  <img src="https://img.shields.io/badge/▶️%20Watch-Demo-2ea44f?style=for-the-badge" alt="Watch Demo">
</a>

</p>

<p align="center">
An interactive Streamlit application for indoor scene classification using the trained PyTorch model.
</p>
---
## Features

- Interactive Streamlit application for image classification
- Modular PyTorch inference pipeline
- Knowledge distillation for lightweight deployment
- Model evaluation with accuracy, Macro F1, and confusion matrix
- Error analysis and misclassification visualization
---

## Dataset

- **Dataset:** MIT Indoor67
- **Classes:** 67 indoor scene categories
- **Images:** 15,620
- **Input Size:** 224 × 224 RGB

---

## Project Workflow

The notebook includes the complete machine learning pipeline:

- Dataset preparation and preprocessing
- DenseNet-style CNN trained from scratch
- ConvNeXt-Tiny linear probing
- ConvNeXt-Tiny fine-tuning
- Knowledge distillation to a lightweight student network
- Model comparison and benchmarking
- Error analysis and confusion matrix generation
- Model export for deployment
- Streamlit-based inference application

---

## Running the Application

```bash
git clone https://github.com/<your-username>/Scene-Recognition-with-Deep-CNNs.git
cd Scene-Recognition-with-Deep-CNNs

pip install -r requirements.txt
streamlit run app.py
```

Upload an indoor scene image to receive the predicted scene category and confidence score.

---

## Future Improvements

- Docker deployment
- GitHub Actions CI/CD
- Vision Transformer (ViT) models
- Cloud deployment

---

## References

- MIT Indoor Scene Recognition Dataset: http://web.mit.edu/torralba/www/indoor.html
