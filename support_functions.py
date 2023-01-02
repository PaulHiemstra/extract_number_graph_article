import pandas as pd
import numpy as np
from plotnine import *
import io
from tensorflow.keras.utils import load_img, img_to_array, array_to_img
from tensorflow.image import resize

def ggplot_to_numpy_array(gg, grayscale=True, resize_to=None, ggsave_args={}):
    '''
    Return a ggplot object (plotnine) as a numpy array. 
    Optionally reduce the image to grayscale first. 
    Any args to gg.save can be passed to `ggsave_args` as a dictonary
    '''
    buffer = io.BytesIO()
    gg.save(buffer, **ggsave_args)
    im = load_img(buffer)  

    if grayscale:
        im = im.convert('L')
    arr = img_to_array(im)
    if not resize_to is None:
        arr = resize(arr, resize_to) 

    return arr

def create_feature_label_pair(size=24, grayscale=True, resize_to=None, ggtheme_args={}, ggsave_args={}):
    '''
    Create a graph that plots random data into a bar graph of size `size`. 

    `ggsave_args` are passed on to ggplot.save, 
      see https://plotnine.readthedocs.io/en/stable/generated/plotnine.ggplot.html for details

    It returns the label (the actual gas usage), the plotnine plot and a numpy array
    containing a numerical representation of that png data. 

    Example usage:

       label, plot, features = create_feature_label_pair(24, grayscale=True, ggsave_args={'width': 3, 'height': 3, 'dpi': 30})
    '''
    plot_data = pd.DataFrame({'hour': np.arange(0,size), 
                              'gas_usage': np.random.uniform(0,5, size=size)})

    gg = (
        ggplot(plot_data) + geom_bar(aes(x='factor(hour)', y='gas_usage'), stat='identity') + theme(**ggtheme_args)
    )

    return [plot_data['gas_usage'], gg, ggplot_to_numpy_array(gg, grayscale=grayscale, resize_to=resize_to, ggsave_args=ggsave_args)]