# AI-Powered-Offline-Object-Detection-Device-For-Visually-Impaired-Users
Object detection has emerged as a transformative technology with applications
 spanning diverse fields, from autonomous vehicles to healthcare. This project
 focuses on leveraging object detection to assist visually impaired individuals in
 navigating their surroundings more effectively. The proposed system integrates
 a Raspberry Pi with a Pi camera and headphones, utilizing TensorFlow’s Object
 Detection API to identify objects in real-time. Once an object is detected, the
 system employs Text-to-Speech (TTS) technology to announce the object’s name
 through the headphones, providing auditory feedback to the user. The model is
 trained using the Single Shot Multi box Detector (SSD) with Mobile Net V2 archi
tecture, ensuring lightweight yet accurate detection suitable for edge devices like
 the Raspberry Pi. This project not only demonstrates the practical implementa
tion of object detection but also highlights its potential to enhance accessibility
 for visually impaired individuals. By combining hardware, software, and machine
 learning, this device aims to empower blind users by providing them with real-time
 information about their environment.

# Problem Statement

Visually impaired individuals often rely on external assistance or rudimentary tools to navigate their surroundings. While existing technologies like GPS and voice assistants are helpful, they lack the granularity required to identify nearby objects in real-time. For example, a blind person walking down a street may encounter obstacles such as bicycles, trash bins, or low-hanging branches, which traditional mobility aids cannot identify. Additionally, indoor navigation poses challenges, as objects like chairs, tables, or kitchen items are difficult to locate without tactile feedback.

This project aims to address these challenges by developing a portable, affordable, and efficient assistive device. The system uses a Raspberry Pi coupled with a Pi camera to capture live video feeds, processes the images using TensorFlow's Object Detection API, and announces the names of detected objects through headphones using Text-to-Speech (TTS) technology. By providing real-time auditory feedback, the device enables visually impaired users to gain a better understanding of their environment, enhancing their independence and safety.

 # PROPOSED MODEL
 
 Based on the papers, we came to the conclusion of using Raspberry Pi 3B + for
 object detection by using the SSD-mobilenet V2 model along with TensorFlow lite
 as the framework. It was decided on the basis of the following reasons.
                        
    1. Object detection without deep learning algorithms, would mean thousands of lines
    of code which would be a cumbersome process. With lot of research
    going under Object Detection and lots of models becoming open-source, there
    will always be newer models in the future having better accuracy, lesser time
    for detection. But that is technological advancements, which needs to be
    dealt in a positive manner.

    2. A micro-controller like device with onsite computing capabilities will reduce
    the cost and time of the process by reducing the server latency time and also
    the need for employing servers for the same..
    
    3. An SSD-mobilenet based model gives better accuracy and lesser inference
    time compared to the other models. Based on the trade-off between time and
    accuracy, it is safer to select this model

 # OVERVIEW OF PROJECT
   
    Step1:
    The first step involves capturing the images for training, we have collected 150
    images of cup, flask, jug as our dataset, which were later split into test images and
    train images
   
    Step2:
    After collecting the dataset we have to label the images further in order to teach
    the machine, this is performed by using a software tool called Lableimg.
   
    Step3:
    After labeling, particular algorithm should be chosen for the training purpose, we
    have chosen SSD MOBILE NET MODEL v2, which is made to run with 2000
    steps in i5 processor machine.
   
    Step4:
    After successful training the test images were given so as to understand the model’s
    accuracy and speed, in our case the chosen model has provided us with an accuracy
    of 96% and has detected all the three labelled datasets.
    
    Step5:
    After successful test results, now the real time object detection should be carried
    out. Through the webcam in our machine we were able to detect the images with
    greater accuracy.
    
    Step6:
    The final step is to convert the model to TFLITE model so that it works efficiently
    on our low processing device Raspberry Pi, the end results were with same accuracy
    as obtained in i5 machine, but the frames per second is slightly less i.e 0.9 frames
    per second.

    
   # Refer Project Report to get systematic steps and Procedure followed.
