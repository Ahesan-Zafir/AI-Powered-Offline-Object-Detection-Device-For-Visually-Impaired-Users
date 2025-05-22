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
