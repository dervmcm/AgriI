1. The is a CNN model aka Convolutional Neural Network. I choose this model because it allows timeline data.

2. I use castle 05's data. If you want to use another castle, you need to download two files: accel-__.csv and halter-__.csv. And you will also need to change the input files in the code

3. I built this model in Spyder. To run this file in your computer, you will need to download python and many of its libraries. For easy use, please download  Anaconda if you haven't done it already. Spyder is included within Anaconda

4. After downloading Anaconda, you will need to install tensorflow to build  Neural Network model. Refer to this documentation for instructions: https://docs.anaconda.com/anaconda/user-guide/tasks/tensorflow/.

If your computer is limited in memory, just type this in Anaconda Prompt:
conda activate base
pip install tensorflow

5. You will also need pandas library. If you follow documentation's instructions, the new environment doesn't include this library. Type the following commands in Anconda Prompt to download it:
conda activate tf
pip install pandas

If you install tensorflow to base (my commands in 4), ignore this step.

6. You can play with the model to see where it goes. I will try to comment all the numbers that you can change.

7. Just to remind you, I haven't organize the prediction, so this model doesn't predict the actions at any moment but rather the actions in the span of 10 seconds.