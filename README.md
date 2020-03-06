# Fire Detector

> 基于Caffe深度学习框架的火灾烟雾检测系统
>

## 介绍

### 烟火识别，清晰准确

Fire Detector 是基于深度学习技术的火灾及烟雾检测解决方案。主要用于判断静态镜头拍摄画面是否有烟雾、明火出现。利用神经网络对图像输入的抽象和学习能力，通过对视频的动态图像进行分析和计算，得到当前场景的火情状态与该状态的具体图像方位。

### 不管什么颜色的烟火

得益于Motion Block技术，佐以对图像的灰度处理，Fire Detector 的核心神经网络能够同时兼容各种阶段火情的识别。Fire Detector可以更早地发现火情，预防火灾的发生。

### 准确轻盈，正是核心所在

高达92%准确率的核心卷积神经网络为 Fire Detector 提供了精准的判别引擎。基于 MobileNet 网络结构使得存储整个神经网络仅仅只需要十余兆字节空间，运行它更是只需要更低的内存和性能开销。

经过训练后的网络权重文件仅仅只有**12.3MB**大小。而精确度却能够达到**94%。**

### 文件

- 运行效果：链接：https://pan.baidu.com/s/1-u626h0VbHV1XbjJ8wiDoQ  提取码：tmoy 
- 模型：example/example_model.caffemodel

## 设计

Fire Detector 为烟火识别提供了两套解决方案，利用卷积神经网络将视频的整帧图像作为输入进行识别，并给出明火（Fire）、正常（Normal）和烟雾（Smoke）三种现实情况的概率，在输出的图像当中使用红色、绿色和蓝色对三种情况进行区别，可以帮助用户更加清晰地辨别。

### Motion Block

Fire Detector 也提供一种细分画面的方案，在此方案中 Fire Detector 将对是将每一帧图像分区进行多帧采样，得到有画面变化的区域，继而判别其中的具体情况，所得的结果将在图中用色块标明。这种技术专门针对固定位置的家庭摄像头设计，可以有效保证识别的准确度，同样也一定程度上解决了当画面足够大而明火、烟雾过小的情况下特征损失的问题。

## 使用指南

Fire Detector 主要针对监控系统拍摄的监控画面进行识别和分析，主要针对监控视频的输入与分析。采用定制化后的Caffe框架进行计算。因为编译依赖的关系较为复杂，非常建议用户直接使用已经编译的Fire Detector文件。

### Dependencies

- CUDA（测试环境为CUDA 8.0.361）
- Cudnn（测试环境为Cudnn 6.0.21）
- 编译依赖（若直接使用编译完成的 Fire Detector 则无需满足此处依赖）
  - CMake >= 3.2
  - GCC （Fire Detector采用C++11标准编写，故编译器需要能够支持C++11特性，我们在G++ 5.4版本上测试正常）
- Caffe （Fire Detector 需要支持Depth-wise convolution运算的Caffe来进行特别的卷积运算，相关的支持文件将在项目文件中给出）
- OpenCV（我们的测试环境使用的版本为OpenCV 3.4.3）

### Bug Shoot

通常在启动的时候有可能找不到Caffe的链接库路径，报错类似如下信息：

```
./FireDetector: error while loading shared libraries: libcaffe.so.1.0.0: cannot open shared object file: No such file or directory
```

此时请在`LD_LIBRARY_PATH`中添加Caffe的链接库路径即可，类似如下命令：

```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/caffe/build/lib
```

也可以将此命令加入到`~/.bashrc`和`/etc/profile`中方便以后使用。

### 调用方法

我们提供了下面的参数和选项供用户使用：

```
usage: ./FireDetector --video=string [options] ... 
options:
  -d, --deploy        The deploy file of model. (string [=../example/example_deploy.prototxt])
  -m, --model         The '.caffemodel' file of model. (string [=../example/example_model.caffemodel])
      --cmean         Use customed mean file
      --mean          Mean file (string [=../example/example_mean.binaryproto])
  -v, --video         The video to be detected. (string)
      --save          Save the results.
  -o, --output        Path to save the video processed.(need 'save' to be enabled) (string [=])
      --view          Show the process while processing video.
  -p, --parallel      Parallel processing.
  -t, --multi         The number of multithread (int [=1])
      --no_MB         Disable the motion block
  -s, --sample        Capacity of sample frames for motion detecting. (int [=8])
  -b, --block_size    Size of motion block. (int [=48])
  -?, --help          print this message

```

**Example**

仅预览，不保存结果：

```
./FireDetector -v ../example/videos/test.mp4 --view
```

预览，并保存结果：

```
./FireDetector -v ../example/videos/test.mp4 --view --save -o ../example/output.avi
```

### 参数说明

**参数**

| 参数名 | 简写 | 描述 |
| --- | --- | --- |
| --video | -v | 将要处理的视频文件 |

**配置**

| 配置名 | 简写 | 描述 |
| --- | --- | --- |
| --deploy | -d | 神经网络模型的部署文件（默认将采用默认模型） |
| --model | -m | 神经网络的权重文件（默认将采用与`--deploy`相匹配的默认模型） |
| --cmean | 无 | 是否使用外部均值文件（通过`--mean`指定） |
| --mean | 无 | 输入均值文件（默认将采用默认神经网络模型所使用的均值信息） |
| --save | 无 | (flag)保存渲染结果到视频文件 |
| --output | -o | 渲染结果的保存位置 |
| --view | 无 | (flag)在渲染时预览 |
| --parallel | -p | (flag)使用异步计算模式 |
| --multi | -t | 异步计算模式下的线程数 |
| --no_MB | 无 | 禁用Motion Block，可大幅提升运行速度 |
| --sample | -s | 视频采样帧容量（采样容量越大，计算时间越长） |
| --block_size | -b | Motion Block 的大小（Motion Block越大，鉴别越粗略，速度越快） |
| --help | -? | 显示帮助 |

### 编译选项

如果需要编译 Fire Detector ，请确保你的编译器支持C++11特性。我们推荐使用CMake编译 Fire Detector，所需要做的，只是确定如下环境变量并进行编译。

| 变量名 | 备注 |
| --- | --- |
| OpenCV_DIR | OpenCV库的路径（此路径下需包含OpenCVConfig.cmake）|
| Caffe_DIR | Caffe路径（此路径下需包含CaffeConfig.cmake）|
| Caffe_INCLUDE_DIRS | Caffe头文件路径，通常为`/path/to/caffe/include`|
| Caffe_BUILD_INCLUDE_DIRS | Caffe编译后头文件路径，通常为`/path/to/caffe/build/include`|

## 训练数据

### 数据集来源

在 Fire Detector 的开发过程中，我们从以下渠道获得了训练和测试数据集。

- 烟雾状态与正常状态
  - http://smoke.ustc.edu.cn/datasets.htm
- 明火
  - https://collections.durham.ac.uk/collections/r1ww72bb497
  - https://github.com/steffensbola/furg-fire-dataset
  - https://zenodo.org/record/836749
  - http://www.image-net.org
- 其他
  - http://signal.ee.bilkent.edu.tr/VisiFire/Demo/SampleClips.html

### 训练平台

Fire Detector 开发小组目前使用 Geforce GTX 1080Ti 进行训练。并在加入了Depth-wise卷积方法的Caffe中进行训练，同样，运行环境也需要支持Depth-wise卷积的Caffe深度学习框架。

## 开发计划

根据目前的开发进度，我们仍可以提供下列对 Fire Detector 进行更多优化的方案。

### 功能性改进

- Fire Detector 可以在当前的状况下将分类器推广运用到小型火灾和山火等大型火灾的识别和判断。
- Fire Detector 可以采用语义分割以更准确地寻找出火焰与烟雾。

### 性能改进

- 虽然已经针对 Motion Block 进行了异步处理优化，但 Fire Detector 仍可以通过异步并行技术对当前视频处理流程进行改进并大幅提升图像处理速度。
- 针对烟雾与明火均具有的情况，目前的Motion Block已经可以做到同时识别烟雾和明火，但仍然可以通过改进Motion Block的精确度来使这项特性更加完善。
