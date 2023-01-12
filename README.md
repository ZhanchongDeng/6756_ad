# 6756_ad
6756 Final Project on Autonomous Driving

We trained a behavior cloning agent using a modified ResNet50 (pretrained on ImageNet), with a forecasting model providing other agent's next move. We achieve better result on open-loop evaluations than baseline, particularly in angle errors.

## Prediction
We use EfficientNetV3 for prediction of other agents.

## Planning
We use ResNet50 (pretrained on ImageNet) for behavioral planning of AV.
