# MachineLearningPlayground
Implementation of basic mathematical pattern recognition/machine learning techniques for fun

## Setup
- Download with pip
```Prompt
pip install MLplayground
```

- Download with git
```Prompt
git clone https://github.com/lujiazho/MachineLearningPlayground.git
```

## Basic Algorithms
- [x] :star: [Support Vector Classifier (SVC)](Tutorials/SVC.ipynb) - ([Math Derivaton](Math_Derivation/SVC.pdf))
- [x] :smiley: [Support Vector Regressor (SVR)](Tutorials/SVR.ipynb) - ([Math Derivaton](Math_Derivation/SVR.pdf))
- [x] :1234: [Ridge Regression](Tutorials/Ridge.ipynb) - ([Math Derivaton](Math_Derivation/Ridge_Regression.pdf))
- [x] :mortar_board: [Nearest Mean](Tutorials/NearestMean.ipynb) - ([Math Derivaton](Math_Derivation/K-means_n_Nearest-means.pdf))
- [x] :closed_book: [K-Means](Tutorials/KMeans.ipynb) - ([Math Derivaton](Math_Derivation/K-means_n_Nearest-means.pdf))
- [x] :green_book: [K-Nearest Neighbors (KNN)](Tutorials/KNN.ipynb) - ([Math Derivaton](Math_Derivation/KNN.pdf))
- [x] :eyes: [Perceptron Learning](Tutorials/Perceptron.ipynb) - ([Math Derivaton](Math_Derivation/Perceptron_Learning_n_Gradient_Descent.pdf))
- [x] :camera: [MSE techniques (classification&Regression)](Tutorials/MSE.ipynb) - ([Math Derivaton](Math_Derivation/MSE_techniques.pdf))
- [x] :mahjong: [Density Estimation (Non-parametric)](Tutorials/DenEstimate_NP.ipynb) - ([Math Derivaton](Math_Derivation/Density_Estimation.pdf))
- [x] :busts_in_silhouette: [Density Estimation (parametric)](Tutorials/DenEstimate_P.ipynb) - ([Math Derivaton](Math_Derivation/Density_Estimation.pdf))
- [x] :bar_chart: [ANN](Tutorials/ANN.ipynb) - ([Math Derivaton](Math_Derivation/ANN.pdf))
- [x] :snake: [PCA](Tutorials/PCA.ipynb) - ([Math Derivaton](Math_Derivation/Feature_Reduction.pdf))


## Convolutional Neural Networks (CNN)

Training time on Colab of multiple implementation of CNN with parameters: epochs=20, batch=2.

|Model / Dataset (imgs)|Loops<br><sup>CPU<br>(s/epoch)|NumPy<br><sup>CPU<br>(s/epoch)|CuPy<br><sup>GPU<br>(s/epoch)|Loops+Numba<br><sup>CPU<br>(s/epoch)|Img2col<br><sup>CPU<br>(s/epoch)|Img2col+Numba<br><sup>CPU<br>(s/epoch)
|---                     |---                           |---          |---         |---           |---          |---
|Baseline / Digits (1k)|255|24|19|2|2|**1.5**
|Lenet / Digits (1k)|464|72|63|4.5|**4**|**4**
|Lenet / Cifar-10 (100)|184.5|13.5|12|0.9|**0.6**|0.7

**Junior versions**
- [x] 💻 [CNN 1.0](Tutorials/CNN/CNN1.0.ipynb) - No Batch No Channel
- [x] 🌱 [CNN 2.0](Tutorials/CNN/CNN2.0.ipynb) - No Batch But Channel

**Senior versions:** Include both batch & channel
- [x] 💬 [CNN 3.0](Tutorials/CNN/CNN3.0.ipynb) - NumPy Array accelerated
- [x] 🔭 [CNN loops](Tutorials/CNN/CNN_loops_numba.ipynb) - Loops + Numba accelerated
- [x] ✨ [CNN img2col](Tutorials/CNN/CNN_img2col_numba.ipynb) - Img2col Function accelerated

**Math Derivation**
- My version: [CNN](Math_Derivation/CNN.pdf), [Img2col](Math_Derivation/img2col.pdf)
- A better tutorial from [Microsoft](https://microsoft.github.io/ai-edu/)

## ML Playground
- [x] :surfer: [Digit Recognizer](Tutorials/_Project_1_digit_recognizer.ipynb)
- [x] :fireworks: [Auto Encoder](Tutorials/_Project_2_auto_encoder.ipynb)
- [x] :pencil2: [Neural Network Language Model (NNLM)](Tutorials/_Project_3_NNLanguageModel.ipynb) - ([Model Structure](Math_Derivation/NLP_NNLM.pdf))
- [x] :bulb: [Word2Vec (Skip-gram)](Tutorials/_Project_4_Word2Vec(Skip-gram).ipynb) - ([Model Structure](Math_Derivation/NLP_word2vec_skipgram.pdf))
- [ ] :mega: []()
