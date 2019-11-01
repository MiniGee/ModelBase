# Model Base

These are the base classes that are used in my other machine learning projects with Keras.

Note: This project was made for the purpose of my learning experience and is not intended to actually be used

## Summary

The [ModelBase](https://github.com/MiniGee/ModelBase/blob/master/model_base.py) class provides a <code>load</code> function to load weights for a certain epoch, and a <code>train</code> function that starts a training session and prints a progress bar, keeps track of loss data, and saves weights every epoch. It also keeps track of testing metrics.

The [LoaderBase](https://github.com/MiniGee/ModelBase/blob/master/loader_base.py) class is the base loader class that takes care of preprocessing and loading training data. It has an overridable <code>_load_file</code> function that allows the user define how to load a file. It also handles splitting training and test data, and randomizing each batch.

The utils file will contain all utility classes needed for the projects. So far, it just has the <code>MovingAvg</code> class, which calculates the average over the n most recent values.
