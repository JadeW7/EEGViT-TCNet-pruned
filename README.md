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

## Remaining Appendix Citations for Literature Review on EEG Data Cleaning Methodologies
 \item Harishvijey, A., and J. Benadict Raja. "Automated technique for EEG signal processing to detect seizure with optimized Variable Gaussian Filter and Fuzzy RBFELM classifier." \textit{Biomedical Signal Processing and Control} 74 (2022): 103450. Available at: \url{https://www.sciencedirect.com/science/article/abs/pii/S1746809421010478}
    
  \item Bajaj, Nikesh, et al. "Automatic and tunable algorithm for EEG artifact removal using wavelet decomposition with applications in predictive modeling during auditory tasks." \textit{Biomedical Signal Processing and Control} 55 (2020): 101624. Available at: \url{https://www.sciencedirect.com/science/article/abs/pii/S1746809419302058}
    
  \item Haresign, I. Marriott, et al. "Automatic classification of ICA components from infant EEG using MARA." \textit{Developmental Cognitive Neuroscience} 52 (2021): 101024. Available at: \url{https://www.sciencedirect.com/science/article/pii/S1878929321001146}
    
  \item Yasoda, K., et al. "Automatic detection and classification of EEG artifacts using fuzzy kernel SVM and wavelet ICA (WICA)." \textit{Soft Computing} 24, no. 21 (2020): 16011-16019. Available at: \url{https://link.springer.com/article/10.1007/s00500-020-04920-w}

    \item Phadikar, Souvik, Nidul Sinha, and Rajdeep Ghosh. "Automatic EEG eyeblink artefact identification and removal technique using independent component analysis in combination with support vector machines and denoising autoencoder." \textit{IET Signal Processing} 14, no. 6 (2020): 396-405. Available at: \url{https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/iet-spr.2020.0025}
    
    \item Lopes, Fábio, et al. "Automatic electroencephalogram artifact removal using deep convolutional neural networks." \textit{IEEE Access} 9 (2021): 149955-149970. Available at: \url{https://ieeexplore.ieee.org/abstract/document/9605576}
    
    \item Phadikar, Souvik, Nidul Sinha, and Rajdeep Ghosh. "Automatic eyeblink artifact removal from EEG signal using wavelet transform with heuristically optimized threshold." \textit{IEEE Journal of Biomedical and Health Informatics} 25, no. 2 (2020): 475-484. Available at: \url{https://ieeexplore.ieee.org/abstract/document/9095264}
    
    \item Vidal, Marc, Mattia Rosso, and Ana M. Aguilera. "Bi-smoothed functional independent component analysis for EEG artifact removal." \textit{Mathematics} 9, no. 11 (2021): 1243. Available at: \url{https://www.mdpi.com/2227-7390/9/11/1243}
    
    \item Ranjan, Rakesh, Bikash Chandra Sahana, and Ashish Kumar Bhandari. "Cardiac artifact noise removal from sleep EEG signals using hybrid denoising model." \textit{IEEE Transactions on Instrumentation and Measurement} 71 (2022): 1-10. Available at: \url{https://ieeexplore.ieee.org/abstract/document/9855513}
    
    \item Hwaidi, Jamal F., and Thomas M. Chen. "Classification of motor imagery EEG signals based on deep autoencoder and convolutional neural network approach." \textit{IEEE Access} 10 (2022): 48071-48081. Available at: \url{https://ieeexplore.ieee.org/abstract/document/9766103}
    
    \item Martini, Michael L., et al. "Deep anomaly detection of seizures with paired stereoelectroencephalography and video recordings." \textit{Scientific Reports} 11, no. 1 (2021): 7482. Available at: \url{https://www.nature.com/articles/s41598-021-86891-y}
    
    \item Mashhadi, Najmeh, et al. "Deep learning denoising for EOG artifacts removal from EEG signals." \textit{2020 IEEE Global Humanitarian Technology Conference (GHTC)}. IEEE, 2020. Available at: \url{https://ieeexplore.ieee.org/abstract/document/9342884}
    
    \item Brophy, Eoin, et al. "Denoising EEG signals for real-world BCI applications using GANs." \textit{Frontiers in Neuroergonomics} 2 (2022): 805573. Available at: \url{https://www.frontiersin.org/articles/10.3389/fnrgo.2021.805573/full}

    \item Çınar, Salim. "Design of an automatic hybrid system for removal of eye-blink artifacts from EEG recordings." \textit{Biomedical Signal Processing and Control} 67 (2021): 102543. Available at: \url{https://www.sciencedirect.com/science/article/abs/pii/S1746809421001403}

    \item Rogasch, Nigel C., Mana Biabani, and Tuomas P. Mutanen. "Designing and comparing cleaning pipelines for TMS-EEG data: A theoretical overview and practical example." \textit{Journal of Neuroscience Methods} 371 (2022): 109494. Available at: \url{https://www.sciencedirect.com/science/article/abs/pii/S0165027022000218}
    
    \item Charupanit, Krit, et al. "Detection of anomalous high‐frequency events in human intracranial EEG." \textit{Epilepsia Open} 5, no. 2 (2020): 263-273. Available at: \url{https://onlinelibrary.wiley.com/doi/full/10.1002/epi4.12397}
    
    \item Abdi-Sargezeh, Bahman, et al. "EEG artifact rejection by extracting spatial and spatio-spectral common components." \textit{Journal of Neuroscience Methods} 358 (2021): 109182. Available at: \url{https://www.sciencedirect.com/science/article/abs/pii/S0165027021001175}
    
    \item Lee, Sangmin S., Kiwon Lee, and Guiyeom Kang. "EEG artifact removal by Bayesian deep learning ,ICA." \textit{2020 42nd Annual International Conference of the IEEE Engineering in Medicine , Biology Society (EMBC)}. IEEE, 2020. Available at: \url{https://ieeexplore.ieee.org/abstract/document/9175785}
    
    \item Salo, Karita S-T., et al. "EEG artifact removal in TMS studies of cortical speech areas." \textit{Brain Topography} 33 (2020): 1-9. Available at: \url{https://link.springer.com/article/10.1007/s10548-019-00724-w}
    
    \item Zangeneh Soroush, Morteza, et al. "EEG artifact removal using sub-space decomposition, nonlinear dynamics, stationary wavelet transform and machine learning algorithms." \textit{Frontiers in Physiology} 13 (2022): 910368. Available at: \url{https://www.frontiersin.org/journals/physiology/articles/10.3389/fphys.2022.910368/full}
    
    \item Delorme, Arnaud. "EEG is better left alone." \textit{Scientific Reports} 13, no. 1 (2023): 2372. Available at: \url{https://www.nature.com/articles/s41598-023-27528-0}
    
    \item Islam, Md Shafiqul, et al. "EEG mobility artifact removal for ambulatory epileptic seizure prediction applications." \textit{Biomedical Signal Processing and Control} 55 (2020): 101638. Available at: \url{https://www.sciencedirect.com/science/article/abs/pii/S1746809419302198}

    \item Zhang, Hao Lan, et al. "EEG self-adjusting data analysis based on optimized sampling for robot control." \textit{Electronics} 9, no. 6 (2020): 925. Available at: \url{https://www.mdpi.com/2079-9292/9/6/925}
    
    \item Kaur, Chamandeep, et al. "EEG Signal denoising using hybrid approach of Variational Mode Decomposition and wavelets for depression." \textit{Biomedical Signal Processing and Control} 65 (2021): 102337. Available at: \url{https://www.sciencedirect.com/science/article/abs/pii/S1746809420304511}
    
    \item Sawangjai, Phattarapong, et al. "EEGANet: Removal of ocular artifacts from the EEG signal using generative adversarial networks." \textit{IEEE Journal of Biomedical and Health Informatics} 26, no. 10 (2021): 4913-4924. Available at: \url{https://ieeexplore.ieee.org/abstract/document/9627782}
    
    \item Zhang, Haoming, et al. "EEGdenoiseNet: A benchmark dataset for deep learning solutions of EEG denoising." \textit{Journal of Neural Engineering} 18, no. 5 (2021): 056057. Available at: \url{https://iopscience.iop.org/article/10.1088/1741-2552/ac2bf8/meta}
    
    \item Kumaravel, Velu Prabhakar, et al. "Efficient artifact removal from low-density wearable EEG using artifacts subspace reconstruction." \textit{2021 43rd Annual International Conference of the IEEE Engineering in Medicine , Biology Society (EMBC)}. IEEE, 2021. Available at: \url{https://ieeexplore.ieee.org/abstract/document/9629771}
    
    \item Yu, Junjie, et al. "Embedding decomposition for artifacts removal in EEG signals." \textit{Journal of Neural Engineering} 19, no. 2 (2022): 026052. Available at: \url{https://iopscience.iop.org/article/10.1088/1741-2552/ac63eb/meta}
    
    \item Maddirala, Ajay Kumar, and Kalyana C. Veluvolu. "Eye-blink artifact removal from single channel EEG with k-means and SSA." \textit{Scientific Reports} 11, no. 1 (2021): 11043. Available at: \url{https://www.nature.com/articles/s41598-021-90437-7}
    
    \item Yin, Jin, et al. "Frequency information enhanced deep EEG denoising network for ocular artifact removal." \textit{IEEE Sensors Journal} 22, no. 22 (2022): 21855-21865. Available at: \url{https://ieeexplore.ieee.org/abstract/document/9908307}
    
    \item Occhipinti, Edoardo, et al. "Hearables: Artefact removal in Ear-EEG for continuous 24/7 monitoring." \textit{2022 International Joint Conference on Neural Networks (IJCNN)}. IEEE, 2022. Available at: \url{https://ieeexplore.ieee.org/abstract/document/9892675}
    
    \item Leach, Sven, Georgia Sousouri, and Reto Huber. "‘High-Density-SleepCleaner’: An open-source, semi-automatic artifact removal routine tailored to high-density sleep EEG." \textit{Journal of Neuroscience Methods} 391 (2023): 109849. Available at: \url{https://www.sciencedirect.com/science/article/pii/S0165027023000687}

    \item Robbins, Kay A., et al. "How sensitive are EEG results to preprocessing methods: a benchmarking study." \textit{IEEE Transactions on Neural Systems and Rehabilitation Engineering} 28, no. 5 (2020): 1081-1090. Available at: \url{https://ieeexplore.ieee.org/abstract/document/9047940}
    
    \item Chuang, Chun-Hsiang, et al. "IC-U-Net: a U-Net-based denoising autoencoder using mixtures of independent components for automatic EEG artifact removal." \textit{NeuroImage} 263 (2022): 119586. Available at: \url{https://www.sciencedirect.com/science/article/pii/S1053811922007017}
    
    \item Gonsisko, Colton B., Daniel P. Ferris, and Ryan J. Downey. "iCanClean improves independent component analysis of mobile brain imaging with EEG." \textit{Sensors} 23, no. 2 (2023): 928. Available at: \url{https://www.mdpi.com/1424-8220/23/2/928}
    
    \item Klug, Marius, and Klaus Gramann. "Identifying key factors for improving ICA‐based decomposition of EEG data in mobile and stationary experiments." \textit{European Journal of Neuroscience} 54, no. 12 (2021): 8406-8420. Available at: \url{https://onlinelibrary.wiley.com/doi/full/10.1111/ejn.14992}

    \item Mathe, Mariyadasu, Mididoddi Padmaja, and Battula Tirumala Krishna. "Intelligent approach for artifacts removal from EEG signal using heuristic-based convolutional neural network." \textit{Biomedical Signal Processing and Control} 70 (2021): 102935. Available at: \url{https://www.sciencedirect.com/science/article/abs/pii/S1746809421005322}
    
    \item Bailey, N. W., et al. "Introducing RELAX (the Reduction of Electroencephalographic Artifacts): A fully automated pre-processing pipeline for cleaning EEG data-Part 1: Algorithm and Application to Oscillations." \textit{BioRxiv} (2022): 2022-03. Available at: \url{https://www.biorxiv.org/content/10.1101/2022.03.08.483548v1.abstract}
    
    \item Liu, Shengjie, et al. "Investigating data cleaning methods to improve performance of brain–computer interfaces based on stereo-electroencephalography." \textit{Frontiers in Neuroscience} 15 (2021): 725384. Available at: \url{https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2021.725384/full}
    
    \item Farhangi, Farbod. "Investigating the role of data preprocessing, hyperparameters tuning, and type of machine learning algorithm in the improvement of drowsy EEG signal modeling." \textit{Intelligent Systems with Applications} 15 (2022): 200100. Available at: \url{https://www.sciencedirect.com/science/article/pii/S2667305322000357}
    
    \item Seok, Dongyeol, et al. "Motion artifact removal techniques for wearable EEG and PPG sensor systems." \textit{Frontiers in Electronics} 2 (2021): 685513. Available at: \url{https://www.frontiersin.org/articles/10.3389/felec.2021.685513/full}
    
    \item Ranjan, Rakesh, Bikash Chandra Sahana, and Ashish Kumar Bhandari. "Motion artifacts suppression from EEG signals using an adaptive signal denoising method." \textit{IEEE Transactions on Instrumentation and Measurement} 71 (2022): 1-10. Available at: \url{https://ieeexplore.ieee.org/abstract/document/9676594}
    
    \item Kumaravel, Velu Prabhakar, et al. "NEAR: An artifact removal pipeline for human newborn EEG data." \textit{Developmental Cognitive Neuroscience} 54 (2022): 101068. Available at: \url{https://www.sciencedirect.com/science/article/pii/S1878929322000123}
    
    \item Dimigen, Olaf. "Optimizing the ICA-based removal of ocular EEG artifacts from free viewing experiments." \textit{NeuroImage} 207 (2020): 116117. Available at: \url{https://www.sciencedirect.com/science/article/pii/S1053811919307086}
    
    \item Liu, Yizhi, et al. "Paving the way for future EEG studies in construction: dependent component analysis for automatic ocular artifact removal from brainwave signals." \textit{Journal of Construction Engineering and Management} 147, no. 8 (2021): 04021087. Available at: \url{https://ascelibrary.org/doi/abs/10.1061/%28ASCE%29CO.1943-7862.0002097}
    
    \item Pilcevic, Dejan, et al. "Performance evaluation of metaheuristics-tuned recurrent neural networks for electroencephalography anomaly detection." \textit{Frontiers in Physiology} 14 (2023): 1267011. Available at: \url{https://www.frontiersin.org/journals/physiology/articles/10.3389/fphys.2023.1267011/full}
    
    \item Islam, Md Kafiul, Parviz Ghorbanzadeh, and Amir Rastegarnia. "Probability mapping based artifact detection and removal from single-channel EEG signals for brain–computer interface applications." \textit{Journal of Neuroscience Methods} 360 (2021): 109249. Available at: \url{https://www.sciencedirect.com/science/article/abs/pii/S0165027021001849}
    
    \item Varone, Giuseppe, et al. "Real-time artifacts reduction during TMS-EEG co-registration: a comprehensive review on technologies and procedures." \textit{Sensors} 21, no. 2 (2021): 637. Available at: \url{https://www.mdpi.com/1424-8220/21/2/637}
    
    \item Bailey, N. W., et al. "RELAX part 2: A fully automated EEG data cleaning algorithm that is applicable to Event-Related-Potentials." \textit{Clinical Neurophysiology} 149 (2023): 202-222. Available at: \url{https://www.sciencedirect.com/science/article/abs/pii/S1388245723000287}

    \item Gorjan, Dasa, et al. "Removal of movement-induced EEG artifacts: current state of the art and guidelines." \textit{Journal of Neural Engineering} 19, no. 1 (2022): 011004. Available at: \url{https://iopscience.iop.org/article/10.1088/1741-2552/ac542c/meta}

    \item Mahmood, Danyal, Humaira Nisar, and Yap Vooi Voon. "Removal of physiological artifacts from electroencephalogram signals: a review and case study." \textit{2021 IEEE 9th Conference on Systems, Process and Control (ICSPC 2021)}. IEEE, 2021. Available at: \url{https://ieeexplore.ieee.org/abstract/document/9689094}
    
    \item Mumtaz, Wajid, Suleman Rasheed, and Alina Irfan. "Review of challenges associated with the EEG artifact removal methods." \textit{Biomedical Signal Processing and Control} 68 (2021): 102741. Available at: \url{https://www.sciencedirect.com/science/article/pii/S1746809421003384}
    
    \item Ho, Thi Kieu Khanh, and Narges Armanfard. "Self-supervised learning for anomalous channel detection in eeg graphs: Application to seizure analysis." \textit{Proceedings of the AAAI Conference on Artificial Intelligence} Vol. 37, No. 7 (2023). Available at: \url{https://ojs.aaai.org/index.php/AAAI/article/view/25952}
    
    \item Mutanen, Tuomas P., et al. "Source-based artifact-rejection techniques for TMS–EEG." \textit{Journal of Neuroscience Methods} 382 (2022): 109693. Available at: \url{https://www.sciencedirect.com/science/article/pii/S0165027022002199}
    
    \item Ke, Jinjing, Jing Du, and Xiaowei Luo. "The effect of noise content and level on cognitive performance measured by electroencephalography (EEG)." \textit{Automation in Construction} 130 (2021): 103836. Available at: \url{https://www.sciencedirect.com/science/article/abs/pii/S0926580521002879}
    
    \item Bertazzoli, Giacomo, et al. "The impact of artifact removal approaches on TMS–EEG signal." \textit{NeuroImage} 239 (2021): 118272. Available at: \url{https://www.sciencedirect.com/science/article/pii/S1053811921005486}
    
    \item Anders, Phillipp, et al. "The influence of motor tasks and cut-off parameter selection on artifact subspace reconstruction in EEG recordings." \textit{Medical , Biological Engineering , Computing} 58 (2020): 2673-2683. Available at: \url{https://link.springer.com/article/10.1007/s11517-020-02252-3}
    
    \item Hernandez-Pavon, Julio C., et al. "TMS combined with EEG: Recommendations and open issues for data collection and analysis." \textit{Brain Stimulation} 16, no. 2 (2023): 567-593. Available at: \url{https://www.sciencedirect.com/science/article/pii/S1935861X23016960}
    
    \item Aghaei-Lasboo, Anahita, et al. "Tripolar concentric EEG electrodes reduce noise." \textit{Clinical Neurophysiology} 131, no. 1 (2020): 193-198. Available at: \url{https://www.sciencedirect.com/science/article/abs/pii/S1388245719312854}
    
    \item Saba-Sadiya, Sari, et al. "Unsupervised EEG artifact detection and correction." \textit{Frontiers in Digital Health} 2 (2021): 608920. Available at: \url{https://www.frontiersin.org/journals/digital-health/articles/10.3389/fdgth.2020.608920/full}
    
    \item Saini, Manali, Udit Satija, and Madhur Deo Upadhayay. "Wavelet based waveform distortion measures for assessment of denoised EEG quality with reference to noise-free EEG signal." \textit{IEEE Signal Processing Letters} 27 (2020): 1260-1264. Available at: \url{https://ieeexplore.ieee.org/abstract/document/9133280}
    
    \item Gajbhiye, Pranjali, et al. "Wavelet domain optimized Savitzky–Golay filter for the removal of motion artifacts from EEG recordings." \textit{IEEE Transactions on Instrumentation and Measurement} 70 (2020): 1-11. Available at: \url{https://ieeexplore.ieee.org/abstract/document/9272776}
    
    \item Klug, Marius, and Niels A. Kloosterman. "Zapline‐plus: A Zapline extension for automatic and adaptive removal of frequency‐specific noise artifacts in M/EEG." \textit{Human Brain Mapping} 43, no. 9 (2022): 2743-2758. Available at: \url{https://onlinelibrary.wiley.com/doi/full/10.1002/hbm.25832}
