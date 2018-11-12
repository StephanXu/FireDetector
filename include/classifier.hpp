#pragma once

#ifndef CLASSIFIER_H
#define CLASSIFIER_H

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>
#include <caffe/caffe.hpp>

class Cclassifier
{
  public:
  /*
  Cclassifier:
  */
    explicit Cclassifier(const std::string &deploy_file,
                         const std::string &caffemodel,
                         const std::vector<std::string> &labels);

    std::vector<float> Classify(const cv::Mat &img);

    void SetMean(const std::string &mean_file);

  private:
    // network
    std::shared_ptr<caffe::Net<float>> _net;
    // input data size
    cv::Size _input_geometry;
    // num of channels
    int _num_channels;
    // all labels
    std::vector<std::string> _labels;

    cv::Mat _mean;
};

#endif