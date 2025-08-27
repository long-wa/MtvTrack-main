# MtvTrack : Robust Visual Tracking via Modeling Time-Variant State of the Target

:fire: Project Introduction  

We provide the VQPPF module interface proposed in the MtvTrack paper, along with instructions for training and inference tracking algorithms. The module folder contains implementations of the proposed methods, while the APIInterfaceUsageExamples folder provides examples of VQPPF tracking. You can follow these examples to use the VQPPF module to predict time-varying target states.  

:bulb: Introduction to the use of the proposed method  
The overall pipeline of MtvTrack is shown in the figure below. The figure uses the One-Stream One-Stage [OSTrack](https://github.com/botaoye/OSTrack) tracker as the baseline tracker. Many other One-Stream One-Stage trackers can also use a similar architecture as shown in the figure to embed and integrate VQPPF. In addition, other types of trackers can also be embedded with VQPPF modules. For example, for the Siamese-trackers, the VQPPF module can be used for the output of the backbone network. Similarly, the CNN-Transformer-based trackers is also the same; Two-Stream Two-Stage trackers such as SwinTrack can use VQPPF before the input of the encoder; the autoregressive tracker ArTrack can also use VQPPF before the Pix2Seq Head.  
![Overall Pipeline](https://github.com/long-wa/MtvTrack-main/blob/master/assert/pipeline.png)
:robot: Environment Configuration  
Just configure it according to the baseline tracker. For example, you can configure the environment as in [OSTrack](https://github.com/botaoye/OSTrack).
# Acknowledgement  
Thanks to the following repo for providing us with a lot of convenience to implement our method.  
[OSTrack](https://github.com/botaoye/OSTrack)  
[SwinTrack](https://github.com/LitingLin/SwinTrack)  
[ArTrack](https://github.com/MIV-XJTU/ARTrack)
