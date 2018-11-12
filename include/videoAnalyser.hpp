#pragma once

#ifndef VIDEOANALYSER_HPP
#define VIDEOANALYSER_HPP

#include <string>
#include <queue>
#include <opencv2/opencv.hpp>

class videoAnalyser
{
  public:
    //constructor
    explicit videoAnalyser();
    ~videoAnalyser();
    explicit videoAnalyser(std::string filename);

  public:
    int imgLoad();
    int getQueueRemain();

  private:
    std::string _filename;
    std::queue<cv::Mat *> _queue;
    cv::VideoCapture _capt;
};

#endif