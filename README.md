# Adversarial Attacks on Digits 🎯

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)

This project focuses on exploring **adversarial attacks** on handwritten digits using a Convolutional Neural Network (CNN). The goal is to identify the most vulnerable digit when Gaussian noise is added to the images. The project involves training a CNN model on the MNIST dataset, adding noise to the test images, and evaluating the model's performance to determine which digit is most susceptible to misclassification.

---

## Table of Contents 📚
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Adversarial Attack](#adversarial-attack)
- [Results](#results)
- [Conclusion](#conclusion)
- [Requirements](#requirements)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

---

## Introduction 🧠

Adversarial attacks involve adding small, often imperceptible perturbations to input data to cause machine learning models to make incorrect predictions. In this project, we focus on handwritten digits from the MNIST dataset and investigate how adding **Gaussian noise** affects the model's accuracy, particularly identifying which digit is most vulnerable to such attacks.

---

## Dataset 📊

The [MNIST dataset](http://yann.lecun.com/exdb/mnist/) consists of 70,000 grayscale images of handwritten digits (0-9), each of size 28x28 pixels. The dataset is split into:
- **60,000 training images**
- **10,000 test images**

---

## Model Architecture 🏗️

The CNN model used in this project has the following architecture:

- **Convolutional Layers**: Two convolutional layers with 32 filters each, followed by batch normalization and ReLU activation.
- **Max Pooling**: Max pooling layers to reduce spatial dimensions.
- **Dropout**: Dropout layers to prevent overfitting.
- **Fully Connected Layers**: Dense layers with 256, 128, and 84 units, followed by batch normalization and ReLU activation.
- **Output Layer**: A softmax layer with 10 units for classification.

The model is trained using the **Adam optimizer** and **categorical cross-entropy loss**.

---

## Adversarial Attack ⚔️

To simulate an adversarial attack, **Gaussian noise** is added to the test images. The noise is controlled by the standard deviation (`sigma`) parameter. The model's performance is then evaluated on the noisy images to determine the impact on accuracy and identify the most vulnerable digit.

---

## Results 📈

After adding Gaussian noise with a sigma value of **0.4**, the model's accuracy drops, and the most vulnerable digit is identified as **"1"**. Below is a summary of the results:

| Digit | Accuracy Drop |
|-------|---------------|
| 0     | Low           |
| 1     | **High**      |
| 2     | Medium        |
| ...   | ...           |

For more details, refer to the [classification report](#) in the notebook.

---

## Conclusion 🏁

This project demonstrates the susceptibility of machine learning models to adversarial attacks, even with simple noise addition. The findings highlight the importance of **robust model training** and the need for defenses against adversarial perturbations.

---

## Requirements 📦

To run this project, you need the following Python packages:

- TensorFlow
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

You can install the required packages using the following command:

```bash
pip install -r requirements.txt
```
## Usage 🚀

### Clone the Repository
```bash
git clone https://github.com/your-username/adversarial-attacks-on-digits.git
cd LIME_INCEPTIONv3
```
### Run the Jupyter Notebook
```bash
jupyter notebook Adversarial_attacks_on_digits.ipynb
```
Follow the instructions in the notebook to train the model, perform adversarial attacks, and analyze the results.
### Contributing 🤝
Contributions are welcome! Follow these steps:
1. Fork the repository
2. Create a new branch:
   ```bash
   git checkout -b feature-name
   ```
3. Make your changes and commit:
   ```bash
   git commit -m "Add new feature"
   ```
4. Push to your branch:
  ```bash
  git push origin feature-name
  ```
5. Open a pull request on GitHub
### License 📚
This project is open-source.
### 💡 Need Help?
Feel free to reach out to me at [satyapavan30@gmail.com] for any questions or suggestions!
