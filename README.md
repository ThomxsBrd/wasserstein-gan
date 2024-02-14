# Wassertein Gan Synthesis for Time Series with Complex Temporal Dynamics: Frugal Architectures and Arbitrary Sample-Size Generation

Beroud, T., Abry, P., Malevergne, Y., Senneret, M., Perrin, G. and Macq, J., 2023, June. Wassertein Gan Synthesis for Time Series with Complex Temporal Dynamics: Frugal Architectures and Arbitrary Sample-Size Generation. In ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) (pp. 1-5). IEEE. [link](https://ieeexplore.ieee.org/document/10094897)

## Introduction 

The purpose of this project was to study the potential of Deep Learning techniques for quantitative management. More precisely, it was a question of studying how these tools allow the automatic generation of baskets of financial series, i.e. multivariate synthetic series that reproduce as well as possible the key statistics of the financial series, which will have to be selected from the existing literature or constructed.

## Analysis of the Synthetic Data
Refer to the PyMultiFracs documentation which implements wavelet based multifractal analysis of 1D signals: [PyMultiFracs Documentation](https://neurospin.github.io/pymultifracs/) & [PyMultiFracs GitHub Pages](https://github.com/neurospin/pymultifracs).

#### Installing the analysis package
```
pip install -U https://github.com/neurospin/pymultifracs/archive/master.zip
```

## Training set presentation

- **Synthesizing a fractional Brownian motion**

Sample from the Training Set for a fBm with $H=0.8 \$:

![Example Data_train](https://user-images.githubusercontent.com/103654041/166447468-0b6104a9-d1e8-42d2-8244-40e286d27dc1.png)

- **Synthesizing a multifractal random walk**

Sample from the Training Set for a mrw with $H=0.8 , \ \lambda = \sqrt{0.06} \$:

![Example trainsing Set mrw](https://user-images.githubusercontent.com/103654041/172889176-d95c8a3c-0e4a-491f-9138-ad2ca69d76f3.png)

- **Synthesizing a skewed multifractal random walk**

Sample from the Training Set for a smrw with $H=0.8 , \ \lambda = \sqrt{0.06}, \ K_0 = 0.05, \ \alpha = 0.1 \$:

![smrw](https://user-images.githubusercontent.com/103654041/178753864-10f1e202-e3ac-464a-b264-bb3b5189e53b.png)