#include "classifier.hpp"
#include <string>
#include <vector>
#include <memory>
#include <caffe/caffe.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace caffe;
using namespace cv;

Cclassifier::Cclassifier(const std::string &deploy_file,
                         const std::string &caffemodel,
                         const std::vector<string> &labels)
{
    // set basic settings
    Caffe::set_mode(Caffe::GPU);

    // load network
    _net.reset(new Net<float>(deploy_file, TEST));
    _net->CopyTrainedLayersFrom(caffemodel);

    // get input specification
    Blob<float> *input_layer = _net->input_blobs()[0];
    _input_geometry = cv::Size(input_layer->width(), input_layer->height());
    _num_channels = input_layer->channels();

    // load labels
    _labels.assign(labels.begin(), labels.end());

    // [ATTENTION]here needS more checking...

    return;
}

std::vector<float> Cclassifier::Classify(const cv::Mat &img)
{
    Blob<float>* input_layer = _net->input_blobs()[0];
    input_layer->Reshape(1, _num_channels, _input_geometry.height, _input_geometry.width);
    _net->Reshape();

    //init img here...
    std::vector<cv::Mat> input_channels;
    float *input_data = input_layer->mutable_cpu_data();
    for (int i = 0; i < input_layer->channels(); ++i)
    {
        Mat channel(_input_geometry.height, _input_geometry.width, CV_32FC1, input_data);
        input_channels.push_back(channel);
        input_data += _input_geometry.width * _input_geometry.height;
    } //这个for循环将网络的输入blob同input_channels关联起来
    Mat sample_float;
    img.convertTo(sample_float, CV_32FC3);
    Mat sample_normalized;
    cv::subtract(sample_float, Scalar(129.52675, 128.78506, 116.44242), sample_normalized); //减去均值，均值可以自己求，也可以通过.binaryproto均值文件求出
    cv::split(sample_normalized, input_channels);                                          //将输入图片放入input_channels，即放入了网络的输入blob

    _net->Forward();
    Blob<float>* output_layer = _net->output_blobs()[0];
    const float *begin = output_layer->cpu_data(); //[ATTENTION] WHY???
    const float *end = begin + output_layer->channels();

    return std::vector<float>(begin, end);
}