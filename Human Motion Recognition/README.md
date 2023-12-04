## Human Motion Recognition (HMR)
Bonus Project from **ECSE 444** Microprocessors

Data collection credit to Chenyi Xu
Supporting motion types: Still; Walking; Jumping; Running; Breaststroke; Clapping

This model works on 3D motion data ((x, y, z) from like accelerometer) in csv files. An exemplary dataset can be found in 'olympics.zip'. The program automatically standardizes the data points, which are then trained with a CNN and tested using a confusion matrix.
