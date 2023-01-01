import pandas as pd
import numpy as np
from plotnine import *
import io
from tensorflow.keras.utils import load_img, img_to_array, array_to_img

def ggplot_to_numpy_array(gg, grayscale=True, ggsave_args={}):
    '''
    Return a ggplot object (plotnine) as a numpy array. 
    Optionally reduce the image to grayscale first. 
    '''
    buffer = io.BytesIO()
    gg.save(buffer, **ggsave_args)
    im = load_img(buffer)  
    if grayscale:
        im = im.convert('L')
        return np.squeeze(img_to_array(im))  # Remove the size 1 color dimension we get because of grayscale
    return img_to_array(im)

def plot_image_array(ar):
    '''
    Helper function to plot an image stored as a numpy array. `array_to_img` needs
    a color channel, i.e. a shape of (x,y,no_colors). Normally this is 3, but for gray
    scale images I chose to omit the color channel altogether. To be able to plot the 
    array however we need to add a dummy color channel of lenght one using `expand_dims`. 
    '''
    if len(ar.shape) < 3:
        ar = np.expand_dims(ar, axis=2)
    return array_to_img(ar)

def create_feature_label_pair(size=24, grayscale=True, ggsave_args={}):
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
        ggplot(plot_data) + geom_bar(aes(x='factor(hour)', y='gas_usage'), stat='identity') + scale_y_continuous()
    )

    return [plot_data['gas_usage'], gg, ggplot_to_numpy_array(gg, grayscale=grayscale, ggsave_args=ggsave_args)]