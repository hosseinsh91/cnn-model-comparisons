# Deep Learning Architectures for CIFAR-10

## **📌 Project Overview**
This project explores **multiple deep learning architectures** for **image classification** on the **CIFAR-10 dataset** using **TensorFlow/Keras**. It compares **Convolutional Neural Networks (CNNs)**, evaluates **learning rate strategies**, and implements **transfer learning** to analyze how different techniques impact model performance.

### **🚀 Key Features**
✅ **Implementation of CNN models with different layer structures**  
✅ **Comparison of models with and without pooling layers**  
✅ **Evaluation of Exponential Decay vs. OneCycle Learning Rate Schedulers**  
✅ **Transfer learning with VGG16 (frozen and partially trainable layers)**  
✅ **Performance analysis of model accuracy and training time**  

---

## **📌 Dataset: CIFAR-10**
- The **CIFAR-10 dataset** consists of **60,000 images** in **10 categories**, including:
  - Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck
- Each image is **32x32 pixels with 3 color channels**.

### **📌 Data Preprocessing**
- The dataset is **normalized** by scaling pixel values to `[0,1]`:
  ```python
  (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
  x_train, x_test = x_train / 255.0, x_test / 255.0
