#include "MotionAnalyser.hpp"
#include <vector>
#include <tuple>
#include <deque>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

/* Definition: CMotionAnalyser */

int CMotionAnalyser::generate_blocks(tuple<int, int> size, int block_size)
{
    for (int y{}; y < std::get<1>(size) - block_size; y += block_size)
    {
        for (int x{}; x < std::get<0>(size) - block_size; x += block_size)
        {
            int w{block_size}, h{block_size};
            if (x + block_size > get<0>(size))
            {
                w = get<0>(size) - x;
            }
            if (y+block_size>get<1>(size))
            {
                h = get<1>(size) - y;
            }
            _blocks.push_back(make_tuple(x, y, w, h));
        }
    }
    return 0;
}

int CMotionAnalyser::initialize_detect_object(const int frame_sample_count,
                                              const double Dist2Threshold,
                                              const bool detectShadows)
{
    /* [ATTENTION] please add some control.*/
    _bgsubtrator = createBackgroundSubtractorKNN(_frame_sample_count, 800, false);
}

int CMotionAnalyser::detect_motion(Mat &output)
{
    if (!_bgsubtrator)
        return 0;

    Mat thres, dila;
    threshold(_bgMask, thres, 244, 255, THRESH_BINARY);
    erode(thres, thres, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)), Point(-1, -1), 2);
    dilate(thres, dila, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)), Point(-1, -1), 2);

    dila.copyTo(output);
    return 1;
}

int CMotionAnalyser::get_motion_blocks(vector<tuple<int, int, int, int>> &blockList,
                                       const Mat &motion_map,
                                       vector<tuple<int, int, int, int>> &destBlockList)
{
    Mat int_diff;
    float t11{}, t12{}, t21{}, t22{};
    int x{}, y{}, w{}, h{};
    integral(motion_map, int_diff);
    for (auto it{blockList.begin()}; it != blockList.end(); it++)
    {
        tie(x, y, w, h) = *it;
        t11 = int_diff.at<float>(y, x);
        t12 = int_diff.at<float>(y, x + w);
        t21 = int_diff.at<float>(y + h, x);
        t22 = int_diff.at<float>(y + w, x + h);
        if (t22 - t12 - t21 + t11 > 0)
        {
            destBlockList.push_back(*it);
        }
    }
    return 1;
}

int CMotionAnalyser::feed_img(const Mat &img, const bool auto_apply)
{
    if (!_bgsubtrator)
    {
        return 0;
    }

    _frameList.push_back(img);
    if (_frameList.size() > _frame_sample_count)
    {
        _frameList.pop_front();
    }

    if (auto_apply)
        _bgsubtrator->apply(img, _bgMask);
    return 1;
}

int CMotionAnalyser::apply_img()
{
    if (!_bgsubtrator)
    {
        return 0;
    }

    for (auto it{_frameList.begin()}; it != _frameList.end(); it++)
    {
        _bgsubtrator->apply(*it, _bgMask);
    }
    return 1;
}

/* Definition: CMotionAnalyserOpticalFlow */

vector<Point2f> CMotionAnalyserOpticalFlow::detect_motion_optical_flow(const cv::Mat &img_mat,
                                                                       const cv::Mat &pre_img_mat,
                                                                       const int block_size)
{
    Mat mgrey, mpre_grey;
    /* Convert image to grey */
    cvtColor(img_mat, mgrey, CV_BGR2GRAY);
    cvtColor(pre_img_mat, mpre_grey, CV_BGR2GRAY);

    vector<Point2f> curt_points, prev_points;
    vector<unsigned char> status;
    vector<float> err;
    vector<Point2f> moving_points;

    goodFeaturesToTrack(mgrey, prev_points, 500, 0.001, 10);
    calcOpticalFlowPyrLK(mpre_grey, mgrey, prev_points, curt_points, status, err, Size(10, 10), 3, TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 20, 0.03));

    {
        auto curt_it{curt_points.begin()};
        auto prev_it{prev_points.begin()};
        int c{};
        while (curt_it != curt_points.end())
        {
            if (status[c++] && (2 < (abs(prev_it->x - curt_it->x) + abs(prev_it->y - curt_it->y))))
            {
                moving_points.push_back(*curt_it);
            }
            curt_it++;
            prev_it++;
        }
    }

    return moving_points;
}
