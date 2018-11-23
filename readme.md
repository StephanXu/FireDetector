# Fire Detector

> 基于Caffe深度学习框架的火灾烟雾检测系统
> 武汉工程大学 徐梓涵 张苏沛 肖澳文

## 简介

Fire Detector是火灾及烟雾检测解决方案。主要用于判断静态镜头拍摄画面是否有烟雾、明火出现。它基于OpenCV图形库与Caffe深度学习平台建立。

## 训练集

烟雾与正常

- http://smoke.ustc.edu.cn/datasets.htm

火焰

- https://collections.durham.ac.uk/collections/r1ww72bb497

- https://github.com/steffensbola/furg-fire-dataset

- https://zenodo.org/record/836749

ImageNet

- http://signal.ee.bilkent.edu.tr/VisiFire/Demo/SampleClips.html


## 调用方法

为了调用方便，我们增加了以下接口便于使用：

```
usage: ./FireDetector --video=string [options] ... 
options:
  -d, --deploy        The deploy file of model. (string [=../example/example_deploy.prototxt])
  -m, --model         The '.caffemodel' file of model. (string [=../example/example_model.caffemodel])
      --mean          Mean file (string [=../example/example_mean.binaryproto])
  -v, --video         The video to be detect. (string)
      --save          Save the results.
  -o, --output        Path to save the video processed.(need 'save' to be enabled) (string [=])
      --view          Show the process while processing video.
  -s, --sample        Capacity of sample frames for motion detecting. (int [=8])
  -b, --block_size    Size of motion blocks. (int [=48])
  -?, --help          print this message
```