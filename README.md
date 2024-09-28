
# "Self-supervised normalizing flow for jointing low-light enhancement and deblurring"
Lingyan Li, Chunzi Zhu, Jiale Chen, Baoshun Shi, Qiusheng Lian

## paper
https://doi.org/10.1007/s00034-024-02723-0

## Abatract
Abstract Low-light image enhancement algorithms have been widely developed. Nevertheless, using long exposure under low-light conditions will lead to motion blurs of the captured images, which presents a challenge to address low-light enhancement and deblurring jointly. A recent effort called LEDNet addresses these issues by designing a encoder-decoder pipeline. However, LEDNet relies on paired data during training, but capturing low-blur and normal-sharp images of the same visual scene simultaneously is challenging. To overcome these challenges, we propose a self-supervised normalizing flow called SSFlow for jointing low-light enhancement and deblurring. SSFlow consists of two modules: an orthogonal channel attention U-Net (OAtt-UNet) module for extracting features, and a normalizing flow for correcting color and denoising (CCD flow). During the training of the SSFlow, the two modules are connected to each other by a color map. Concretely, OAtt-UNet module is a variant of U-Net consisting of an encoder and a decoder. OAtt-UNet module takes a low-light blurry image as input, and incorporates an orthogonal channel attention block into the encoder to improve the representation ability of the overall network. The filter adaptive convolutional layer is integrated into the decoder, applying a dynamic convolution filter to each element of the feature for effective deblurring. To extract color information and denoise, the CCD flow makes full use of the powerful learning ability of the normalizing flow. We construct an unsupervised loss function, continuously optimizing the network by using the consistent color map between the two modules in the color space. The effectiveness of our proposed network is demonstrated through both qualitative and quantitative experiments. Code is available at https://github.com/shibaoshun/SSFlow.

## Pre-trained Models
pretrained model: https://pan.baidu.com/s/14EbyudgxDK3bFJJY9bLSxA
key:iqtn


## Environment:
```
python=3.8
cuda=11.3
torch=1.12
numpy=1.23.3
```
## Datasets:
```
LOL，LOL-Blur are datasets that include Ground truth, and they need to be tested using the test.py file.
LOL-Blur: "LEDNet: Joint low-light enhancement and deblurring in the dark"
LOL: "Deep retinex decomposition for low-light enhancement"
LIME and MEF are datasets containing natural low-light images, which need to be tested using the test_unpaired.py file.
LIME: " LIME: Low-light image enhancement via illumination map estimation"
MEF: "Perceptual quality assessment for multi-exposure image fusion. IEEE Transactions on Image Processing"
 ```
## Else
 The testing code was originally written by Zhu Chunzi, a master student of Dr. Shi Baoshun, and was later summarized and added some annotations by Zhao Yingshuai, also a master student of Dr. Shi Baoshun in Sep.28th 2024.
 If you find this code helpful and use it in your research, please cite the associated paper and use it only for academic research purposes.

## Citation
If our work is useful for your research, please consider citing:
```
@article{Li2024SelfSupervisedNF,
  title={Self-Supervised Normalizing Flow for Jointing Low-Light Enhancement and Deblurring},
  author={Lingyan Li and Chunzi Zhu and Jiale Chen and Baoshun Shi and Qiusheng Lian},
  journal={Circuits Syst. Signal Process.},
  year={2024},
  volume={43},
  pages={5727-5748},
  url={https://api.semanticscholar.org/CorpusID:270178160}
}
```
## Contact
If you have any questions, please feel free to reach me out at shibaoshun@ysu.edu.cn.
