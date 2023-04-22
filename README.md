# Multi-scale feature merging strategy for image classification
![Python 3.6.5](https://img.shields.io/badge/python-3.6.5-green.svg?style=plastic)
![PyTorch 1.2](https://img.shields.io/badge/PyTorch-1.12.0-orange?style=plastic)
![License CC BY-NC-SA](https://img.shields.io/badge/license-CC_BY--NC--SA--green.svg?style=plastic)

## Profile
This repository is **especially suitable for course project presentation** (the PowerPoint Slide is also attached) with somewhat innovation showing your understanding towards neural network. <br/>

All data are included in the repository, with a simple clone and correct environment dependency, you can easily work on it !

## Introduction
Based on Resnet50 pretraining model and somewhat necessary strategy like data augment, we aim to develop an image claasification model. Considering the sight that the semantic information of image feature will get more rich when the layer goes deeper, we introduce an Attention-based feature fusing component to get a comprehensive feature. The ablation test is also performed to showcase the effectiveness of this strategy, obtaining a 97.2% accuracy on the included dataset.


<p align="center">
<img src="multi-scale fusing component.png" height = "400" alt="" align=center />
<br><br>
<b>Figure 1.</b> The illustration of Diviner framework.
</p>



## Contact
If you have any questions, feel free to contact YuGuang Yang through Email (moujieguang@gmail.com) or Github issues. Pull requests are highly welcomed!

## Acknowlegements
Si Liu Professor (https://scholar.google.com/citations?hl=zh-CN&user=-QtVtNEAAAAJ) for introducing the basic concept of deep learning
