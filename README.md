# 6756_ad
6756 Final Project on Autonomous Driving

We trained a behavior cloning agent using a modified ResNet50 (pretrained on ImageNet), with a forecasting model providing other agent's next move. We achieve better result on open-loop evaluations than baseline, particularly in angle errors.

## Prediction
We use EfficientNetV3 for prediction of other agents.

## Planning
We use ResNet50 (pretrained on ImageNet) for behavioral planning of AV.

## Installation
You need to install l5kit. There are potential errors and solutiosn to the installation.
- Problem: unclear why python version matters
Solution: python version: 3.8

- Problem: l5kit install protobuf is 4.x, files protobuf generated is too old
Solution: pip install protobuf==3.2

- Problem: l5kit install numpy 1.19, file uses numpy.typing
Solution: pip install numpy==1.20