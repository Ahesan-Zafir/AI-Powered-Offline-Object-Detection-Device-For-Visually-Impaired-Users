# Overview of the System
The proposed system consists of the following components:
1.	Hardware : A Raspberry Pi serves as the central processing unit, interfaced with a Pi camera for capturing video input and headphones for delivering auditory output.
2.	Software : TensorFlow's Object Detection API is used to train and deploy the object detection model. The Single Shot Multibox Detector (SSD) with MobileNet V2 architecture is chosen for its balance of speed and accuracy, making it ideal for edge devices.
3.	Text-to-Speech (TTS) : Detected objects are announced using TTS libraries, converting text labels into spoken words that are relayed to the user via headphones.
4.	Training Data : The model is trained on a custom dataset comprising images of everyday objects such as cups, jugs, flasks, chairs, tables, and other items relevant to visually impaired users.

