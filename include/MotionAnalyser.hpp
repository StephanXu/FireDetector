#pragma once

#ifndef MOTIONANALYSER_HPP
#define MOTIONANALYSER_HPP

#include <vector>
#include <tuple>
#include <deque>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

class CMotionAnalyser
{
  public:
    vector<tuple<int, int, int, int>> _blocks;

    /* Generate blocks */
    int generate_blocks(tuple<int, int> size, int block_size);

    /*
    Init BackgroundSubtractorKNN object.
    Please call this function before you call any function except 'generate_blocks' whatever, or
    other function will return 0 that means failure.
    Arguments:
        frame_sample_count: the num of samples
        Dist2Threshold: A threshold of the squared distance between the pixel and
                        the sample to determine if the pixel is close to the sample. 
                        (This parameter does not affect background updates.)
        detectShadows: Set this param 'true' to detect shadows. (It costs much time)
    */
    void initialize_detect_object(const int frame_sample_count,
                                 const double Dist2Threshold = 400.0,
                                 const bool detectShadows = false);
    /*
    Detect motion status.
    It depense on '_bgMask' which will be generate automatically by calling 'feed_img' with param 'auto_apply' be true
    or generate manually by calling 'apply_img'
    Arguments:
        output: the cv::Mat receive motion status matrix
    */
    int detect_motion(Mat &output);

    int get_motion_blocks(vector<tuple<int, int, int, int>> &blockList,
                          const Mat &motion_map,
                          vector<tuple<int, int, int, int>> &destBlockList);

    int feed_img(const Mat &img, const bool auto_apply = true);

    /* Call this function to generate '_bgMask' manually */
    int apply_img();

  private:
    deque<Mat> _frameList;
    Ptr<BackgroundSubtractorKNN> _bgsubtrator;
    Mat _bgMask;

  private:
    const int _frame_sample_count{16};
};

class CMotionAnalyserOpticalFlow
{
  public:
    vector<Point2f> detect_motion_optical_flow(const cv::Mat &img_mat,
                                               const cv::Mat &pre_img_mat,
                                               const int block_size);
};


#endif