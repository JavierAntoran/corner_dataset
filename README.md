# corner_dataset

Simple dataset used for viewing how a multi layer perceptron organises
information throughout its activation space.

The dataset consists of 4096 32x32 pixel images belonging
to 4 different classes. Each class shows two lines forming a right angle
pointing towards a different corner of the image. Line length, thickness and possition vary across images. Low amplitude
gaussian noise is added to images.

The dataset can be loaded using numpy:
```python
x = np.load('data/angle_ims.npy')
y = np.load('data/angle_targets.npy')
```


 ![Alt text](pics/100_samples.png?raw=true "some examples")
