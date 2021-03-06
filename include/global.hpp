#pragma once

#ifndef MAIN_HPP
#define MAIN_HPP

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

using namespace std;

class Cglobal
{
  public:
    explicit Cglobal();

    int parse_params(int argc, char *argv[]);

    /* Constant */
    const float c_mask_alpha{0.5};
    const vector<cv::Scalar> c_mask_colors{cv::Scalar(0x00, 0x2c, 0xdd), cv::Scalar(0x50, 0xaf, 0x4c), cv::Scalar(0xb5, 0x51, 0x3f)};
    const cv::Scalar c_mean_value{cv::Scalar(103.939, 116.779, 123.68)};
    
    /* Basic Configurations */
    string m_deploy_file;
    string m_model_file;
    bool m_custom_mean;
    string m_mean_file;
    string m_video_file;
    bool m_save_result;
    string m_output_video;
    bool m_previous_wnd;
    int m_multithread_num;
    bool m_enable_multithread;
    bool m_disable_motion_block;

    /* Advance Configurations */
    int m_sample_capacity;
    int m_block_size;
};

extern Cglobal global;

#endif