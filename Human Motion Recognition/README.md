## Human Motion Recognition (HMR)
Bonus Project from **ECSE 444** Microprocessors

Data collection credit to Chenyi Xu
Supported motion types in our dataset (olympics.zip): Still; Walking; Jumping; Running; Breaststroke; Clapping. Note that this exemplary dataset is quite small and can be inaccurate.

This model works on 3D motion data (like accelerometer) from csv files. It automatically standardizes (unify encodings, remove invalid rows, fragmentation) the data points, which are then trained with a CNN and tested using a confusion matrix.
You may customize frame stride (default 26 data points since our sampling rate is 26Hz) and overlapping (13 data points, 50%).
