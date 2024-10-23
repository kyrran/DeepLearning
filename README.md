# Deep Learning Coursework

This repository contains the coursework for the Deep Learning module, including two key projects focused on convolutional neural networks (CNNs) and generative models.

## Coursework 1: Convolutional Neural Networks (CNNs)

For this coursework, the focus was on building and training various CNN components from scratch, along with applying them to a real-world high-dimensional classification task.

### Key Accomplishments:
- Built and implemented convolutional layers, max-pooling layers, and batch normalization layers from scratch using PyTorch.
- Successfully implemented a simplified ResNet-18 model with high classification accuracy.
- Achieved high accuracy on benchmark datasets by refining the model's architecture and hyperparameters.
- Explored the effects of batch normalization and max-pooling on model performance.
- Conducted thorough evaluations of model performance, including visualizations of training and validation progress, to better understand the behavior of deep learning models.


### Notebooks:
- **[Coursework 1 Notebook](https://github.com/kyrran/DeepLearning/blob/main/dl_cw_1.ipynb)**: This notebook walks through the step-by-step implementation of CNN components, ResNet-18, and the detailed analysis of model performance.


---

## Coursework 2: Generative Models

In the second coursework, the emphasis shifted to understanding and implementing generative models using popular datasets such as MNIST and CIFAR-10.

### Part 1: Variational Autoencoder (VAE)
- Implemented a VAE using the MNIST dataset.
- Focused on understanding the variational inference process and how the model learns latent representations of data.
- Evaluated the quality of generated images by varying the latent space sampling.

### Part 2: Deep Convolutional Generative Adversarial Network (DCGAN)
- Built a DCGAN for generating images using the CIFAR-10 dataset.
- Explored the training dynamics of adversarial networks and experimented with different network configurations.
- Some focus was placed on generating high-quality images while keeping an emphasis on understanding the model's behavior and learning process.

### Notebooks:
- **[Coursework 2 VAE & DCGAN Notebook](https://github.com/kyrran/DeepLearning/blob/main/dl_cw2_2024.ipynb)**: This notebook contains the full implementation and analysis of both the Variational Autoencoder and the DCGAN models, including training results and visualizations of the generated images.
  
### Key Tasks:
- Implemented a VAE for generating MNIST digits and analyzed its latent space behavior.
- Implemented a DCGAN for generating CIFAR-10 images, experimented with different architectures, and tuned the generator to improve image quality.
- Visualized the training loss and latent space representation for VAE, observing how the KL divergence and reconstruction loss converge over time.

### Results:
- **VAE Training Results**: The reconstruction loss decreased while KL divergence showed the expected upward trend, indicating convergence. Posterior collapse was mitigated by tuning the beta parameter, balancing between reconstruction and generation quality.
- **DCGAN Training**: Encountered and analyzed mode collapse during training, where the generator produced limited image diversity. Improved model performance by tuning BatchNorm layers and adjusting the architecture.

### Qualitative Results:
- T-SNE visualizations of the latent representations for the VAE show distinct clusters, indicating the model learned meaningful representations of the data.
- DCGAN generated high-quality images after resolving mode collapse, but still displayed some repeated patterns in output images.

### Challenges and Solutions:
- **Posterior Collapse in VAE**: Initially observed during early training stages but successfully mitigated by adjusting the beta parameter to 0.9, preventing the model from overfitting to the prior.
- **Mode Collapse in DCGAN**: Identified as the generator produced similar images repetitively. Resolved by adjusting the model architecture, specifically incorporating BatchNorm, which reduced the collapse significantly.

### Datasets:
- MNIST: [Link to Dataset](https://en.wikipedia.org/wiki/MNIST_database)
- CIFAR-10: [Link to Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)

---
