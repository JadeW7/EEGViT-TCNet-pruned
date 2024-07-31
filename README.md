## EEGViT-TCNet on Cleaned EEGEyeNet Dataset

## Overview
EEGViT is a hybrid Vision Transformer (ViT) incorporated with Depthwise Convolution in patch embedding layers. This work is based on 
Dosovitskiy, et al.'s ["An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"](https://arxiv.org/abs/2010.11929). After finetuning EEGViT pretrained on ImageNet, it achieves a considerable improvement over the SOTA on the Absolute Position task in EEGEyeNet dataset.

This repository consists of five models: ViT pretrained and non-pretrained; EEGViT pretrained and non-pretrained; EEGViT-TCNet pretrained. The pretrained weights of ViT layers are loaded from [huggingface.co](https://huggingface.co/docs/transformers/model_doc/vit).

## Dataset download
Download data for EEGEyeNet absolute position task
```bash
wget -O "./dataset/Position_task_with_dots_synchronised_min.npz" "https://osf.io/download/ge87t/"
```
For more details about EEGEyeNet dataset, please refer to ["EEGEyeNet: a Simultaneous Electroencephalography and Eye-tracking Dataset and Benchmark for Eye Movement Prediction"](https://arxiv.org/abs/2111.05100) and [OSF repository](https://osf.io/ktv7m/)

## Installation

### Requirements

First install the general_requirements.txt

```bash
pip3 install -r general_requirements.txt 
```

### Pytorch Requirements

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
```

For other installation details and different cuda versions, visit [pytorch.org](https://pytorch.org/get-started/locally/).

## Remaining Appendix Citations for "Enhancing EEG Data Quality: A Comprehensive Review of Outlier Detection and Cleaning Methods"

- Mayeli, Ahmad, et al. "Automated pipeline for EEG artifact reduction (APPEAR) recorded during fMRI." *Journal of Neural Engineering* 18, no. 4 (2021): 0460b4. Available at: [https://iopscience.iop.org/article/10.1088/1741-2552/ac1037/meta](https://iopscience.iop.org/article/10.1088/1741-2552/ac1037/meta)

- Harishvijey, A., and J. Benadict Raja. "Automated technique for EEG signal processing to detect seizure with optimized Variable Gaussian Filter and Fuzzy RBFELM classifier." *Biomedical Signal Processing and Control* 74 (2022): 103450. Available at: [https://www.sciencedirect.com/science/article/abs/pii/S1746809421010478](https://www.sciencedirect.com/science/article/abs/pii/S1746809421010478)

- Bajaj, Nikesh, et al. "Automatic and tunable algorithm for EEG artifact removal using wavelet decomposition with applications in predictive modeling during auditory tasks." *Biomedical Signal Processing and Control* 55 (2020): 101624. Available at: [https://www.sciencedirect.com/science/article/abs/pii/S1746809419302058](https://www.sciencedirect.com/science/article/abs/pii/S1746809419302058)

- Haresign, I. Marriott, et al. "Automatic classification of ICA components from infant EEG using MARA." *Developmental Cognitive Neuroscience* 52 (2021): 101024. Available at: [https://www.sciencedirect.com/science/article/pii/S1878929321001146](https://www.sciencedirect.com/science/article/pii/S1878929321001146)

- Yasoda, K., et al. "Automatic detection and classification of EEG artifacts using fuzzy kernel SVM and wavelet ICA (WICA)." *Soft Computing* 24, no. 21 (2020): 16011-16019. Available at: [https://link.springer.com/article/10.1007/s00500-020-04920-w](https://link.springer.com/article/10.1007/s00500-020-04920-w)

- Phadikar, Souvik, Nidul Sinha, and Rajdeep Ghosh. "Automatic EEG eyeblink artefact identification and removal technique using independent component analysis in combination with support vector machines and denoising autoencoder." *IET Signal Processing* 14, no. 6 (2020): 396-405. Available at: [https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/iet-spr.2020.0025](https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/iet-spr.2020.0025)

- Lopes, Fábio, et al. "Automatic electroencephalogram artifact removal using deep convolutional neural networks." *IEEE Access* 9 (2021): 149955-149970. Available at: [https://ieeexplore.ieee.org/abstract/document/9605576](https://ieeexplore.ieee.org/abstract/document/9605576)

- Phadikar, Souvik, Nidul Sinha, and Rajdeep Ghosh. "Automatic eyeblink artifact removal from EEG signal using wavelet transform with heuristically optimized threshold." *IEEE Journal of Biomedical and Health Informatics* 25, no. 2 (2020): 475-484. Available at: [https://ieeexplore.ieee.org/abstract/document/9095264](https://ieeexplore.ieee.org/abstract/document/9095264)

- Vidal, Marc, Mattia Rosso, and Ana M. Aguilera. "Bi-smoothed functional independent component analysis for EEG artifact removal." *Mathematics* 9, no. 11 (2021): 1243. Available at: [https://www.mdpi.com/2227-7390/9/11/1243](https://www.mdpi.com/2227-7390/9/11/1243)

- Ranjan, Rakesh, Bikash Chandra Sahana, and Ashish Kumar Bhandari. "Cardiac artifact noise removal from sleep EEG signals using hybrid denoising model." *IEEE Transactions on Instrumentation and Measurement* 71 (2022): 1-10. Available at: [https://ieeexplore.ieee.org/abstract/document/9855513](https://ieeexplore.ieee.org/abstract/document/9855513)

- Hwaidi, Jamal F., and Thomas M. Chen. "Classification of motor imagery EEG signals based on deep autoencoder and convolutional neural network approach." *IEEE Access* 10 (2022): 48071-48081. Available at: [https://ieeexplore.ieee.org/abstract/document/9766103](https://ieeexplore.ieee.org/abstract/document/9766103)

- Martini, Michael L., et al. "Deep anomaly detection of seizures with paired stereoelectroencephalography and video recordings." *Scientific Reports* 11, no. 1 (2021): 7482. Available at: [https://www.nature.com/articles/s41598-021-86891-y](https://www.nature.com/articles/s41598-021-86891-y)

- Mashhadi, Najmeh, et al. "Deep learning denoising for EOG artifacts removal from EEG signals." *2020 IEEE Global Humanitarian Technology Conference (GHTC)*. IEEE, 2020. Available at: [https://ieeexplore.ieee.org/abstract/document/9342884](https://ieeexplore.ieee.org/abstract/document/9342884)

- Brophy, Eoin, et al. "Denoising EEG signals for real-world BCI applications using GANs." *Frontiers in Neuroergonomics* 2 (2022): 805573. Available at: [https://www.frontiersin.org/articles/10.3389/fnrgo.2021.805573/full](https://www.frontiersin.org/articles/10.3389/fnrgo.2021.805573/full)

- Çınar, Salim. "Design of an automatic hybrid system for removal of eye-blink artifacts from EEG recordings." *Biomedical Signal Processing and Control* 67 (2021): 102543. Available at: [https://www.sciencedirect.com/science/article/abs/pii/S1746809421001403](https://www.sciencedirect.com/science/article/abs/pii/S1746809421001403)

- Rogasch, Nigel C., Mana Biabani, and Tuomas P. Mutanen. "Designing and comparing cleaning pipelines for TMS-EEG data: A theoretical overview and practical example." *Journal of Neuroscience Methods* 371 (2022): 109494. Available at: [https://www.sciencedirect.com/science/article/abs/pii/S0165027022000218](https://www.sciencedirect.com/science/article/abs/pii/S0165027022000218)

- Charupanit, Krit, et al. "Detection of anomalous high‐frequency events in human intracranial EEG." *Epilepsia Open* 5, no. 2 (2020): 263-273. Available at: [https://onlinelibrary.wiley.com/doi/full/10.1002/epi4.12397](https://onlinelibrary.wiley.com/doi/full/10.1002/epi4.12397)

- Abdi-Sargezeh, Bahman, et al. "EEG artifact rejection by extracting spatial and spatio-spectral common components." *Journal of Neuroscience Methods* 358 (2021): 109182. Available at: [https://www.sciencedirect.com/science/article/abs/pii/S0165027021001175](https://www.sciencedirect.com/science/article/abs/pii/S0165027021001175)

- Lee, Sangmin S., Kiwon Lee, and Guiyeom Kang. "EEG artifact removal by Bayesian deep learning ,ICA." *2020 42nd Annual International Conference of the IEEE Engineering in Medicine , Biology Society (EMBC)*. IEEE, 2020. Available at: [https://ieeexplore.ieee.org/abstract/document/9175785](https://ieeexplore.ieee.org/abstract/document/9175785)

- Salo, Karita S-T., et al. "EEG artifact removal in TMS studies of cortical speech areas." *Brain Topography* 33 (2020): 1-9. Available at: [https://link.springer.com/article/10.1007/s10548-019-00724-w](https://link.springer.com/article/10.1007/s10548-019-00724-w)

- Zangeneh Soroush, Morteza, et al. "EEG artifact removal using sub-space decomposition, nonlinear dynamics, stationary wavelet transform and machine learning algorithms." *Frontiers in Physiology* 13 (2022): 910368. Available at: [https://www.frontiersin.org/journals/physiology/articles/10.3389/fphys.2022.910368
