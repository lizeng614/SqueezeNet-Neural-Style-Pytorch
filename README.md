# **Lightweight Neural Style on Pytorch**

This is a lightweight Pytorch implementation of [A Neural Algorithm of Artistic Style](http://arxiv.org/abs/1508.06576) using pretrained [SqueezeNet](https://arxiv.org/abs/1602.07360).

SqueezeNet has AlexNet-level accuracy with 50x fewer parameters. which reduces the computational time for Neural Style and yet has comparable results. It is also included in Torchvison so no explicit download is needed. Simply run the implementation will automatically download a 5MB SqueezeNet from Pytorch website.

Notice that this implementation aims at small size and fast training time on less capable machine like my own Laptop, it **should not beat** the result of using pretrained VGG as used in the original paper. No GPU is needed for this implementation. At test time it takes **less than 1.20s** for each training step for a 616x372 image on a intel core i5 Laptop.

## Examples:

<div align = 'center'>
<img src="https://raw.githubusercontent.com/lizeng614/pytorch-neural-style/master/img_data/img/shanghai.jpg" height="372px">
</div>

<div align = 'center'>
<img src="https://raw.githubusercontent.com/lizeng614/pytorch-neural-style/master/img_data/output/s_ms02.jpg" 372px> <img src="https://raw.githubusercontent.com/lizeng614/pytorch-neural-style/master/img_data/style/display/the_scream.jpg" height="372px">
</div>

<div align = 'center'>
<img src="https://raw.githubusercontent.com/lizeng614/pytorch-neural-style/master/img_data/output/s_s.jpg" height="372px"> <img src="https://raw.githubusercontent.com/lizeng614/pytorch-neural-style/master/img_data/style/display/stary_night.jpg" height="372px">
</div>

<div align = 'center'>
<img src="https://raw.githubusercontent.com/lizeng614/pytorch-neural-style/master/img_data/output/s_lm_01.jpg" height="372px"> <img src="https://raw.githubusercontent.com/lizeng614/pytorch-neural-style/master/img_data/style/display/la_muse.jpg" height="372px">
</div>

<div align = 'center'>
<img src="https://raw.githubusercontent.com/lizeng614/pytorch-neural-style/master/img_data/output/s_w01.jpg" height="372px"> <img src="https://raw.githubusercontent.com/lizeng614/pytorch-neural-style/master/img_data/style/display/wave_1.jpg" height="372px">
</div>

## Usage
Basic usage:
```
python neural_style.py --style <image.jpg> --content <image.jpg>
```
use ```--standard-train True```to perform a suggested training process, which is 500 steps Adam of learning rate 0.1 fellowed by another 500 steps Adam of learning rate 0.01

You can run the Ipython Notebook Version. This is convenient if you want play with some parameters, you can even try different layer of SqueezeNet.This implementation only used 7 layers from it's 12 defined

## Advanced Neural Style
for more advanced implementation on good GPU machine please check Fast-Neural-style: [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](http://cs.stanford.edu/people/jcjohns/eccv16/), codes are here: [torch](https://github.com/jcjohnson/fast-neural-style) or [tensorflow](https://github.com/lengstrom/fast-style-transfer).
