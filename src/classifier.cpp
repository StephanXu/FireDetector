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
using namespace std;

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
    Blob<float> *input_layer = _net->input_blobs()[0];
    input_layer->Reshape(1, _num_channels, _input_geometry.height, _input_geometry.width);
    _net->Reshape();

    //init img here...
    int width = input_layer->width();
    int height = input_layer->height();
    std::vector<cv::Mat> input_channels;
    float *input_data = input_layer->mutable_cpu_data();
    for (int i = 0; i < input_layer->channels(); ++i)
    {
        Mat channel(height, width, CV_32FC1, input_data);
        input_channels.push_back(channel);
        input_data += width * height;
    } //将网络的输入blob同input_channels关联起来

    // make a resized copy for data
    cv::Mat img_resized;
    if (img.size() != _input_geometry)
    {
        cv::resize(img, img_resized, _input_geometry);
    }
    else
    {
        img_resized = img;
    }

    Mat sample_float;
    img_resized.convertTo(sample_float, CV_32FC3);

    Mat sample_normalized;
    // cv::subtract(sample_float, Scalar(103.939f, 116.779f, 123.68f), sample_normalized);
    cv::subtract(sample_float, _mean, sample_normalized);
    cv::split(sample_normalized, input_channels); // put image into input_channels(input to input blob)

    _net->Forward();
    Blob<float> *output_layer = _net->output_blobs()[0];
    const float *begin = output_layer->cpu_data(); //[ATTENTION] WHY???
    const float *end = begin + output_layer->channels();

    return std::vector<float>(begin, end);
}

/* Load the mean file in binaryproto format. */
void Cclassifier::SetMean(const std::string &mean_file)
{
    BlobProto blob_proto;
    ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);

    /* Convert from BlobProto to Blob<float> */
    Blob<float> mean_blob;
    mean_blob.FromProto(blob_proto);

    /* The format of the mean file is planar 32-bit float BGR or grayscale. */
    std::vector<cv::Mat> channels;
    float *data = mean_blob.mutable_cpu_data();
    for (int i = 0; i < _num_channels; ++i)
    {
        /* Extract an individual channel. */
        cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
        channels.push_back(channel);
        data += mean_blob.height() * mean_blob.width();
    }

    /* Merge the separate channels into a single image. */
    cv::Mat mean;
    cv::merge(channels, mean);

    /* Compute the global mean pixel value and create a mean image
   * filled with this value. */
    cv::Scalar channel_mean = cv::mean(mean);
    _mean = cv::Mat(_input_geometry, mean.type(), channel_mean);
}

std::tuple<int, float> Cclassifier::GetResult(const std::vector<float> &classify_result)
{
    auto max_item = max_element(classify_result.begin(), classify_result.end());
    return make_tuple(max_item - classify_result.begin(), *max_item);
}