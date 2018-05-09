# Recurrent Attention Model for Image Classification
---
## Description
Convolutional Neural Network is proved to be an effective mechanism to perform object-classification task. Yet, it imposes tremendous computational overhead while processing the image as it needs to read entire image to emit the prediction. [(Mnih et al. 2014)](https://arxiv.org/pdf/1406.6247.pdf) developed a Recurrent Attention Model (RAM) mimicking the retina-like structure that focuses attention to only those parts of the image which provide maximum information to accomplish the classification task. As demonstrated by the authors, this model achieves considerable accuracy when tested on MNIST dataset in three different settings, namely, ‘Centered Digits’, ‘Non-Centered Digits’ and ‘Cluttered Non-Centered Digits’. In this project, I implement this model as it is and study the results.

## Architecture
![Architecture](images/architecture.png "Architecture of the Recurrent Attention Model")

The architecture consists of Glimpse Sensor, Glimpse Network, Core Network, Action Network and Location Network.
