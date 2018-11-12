
#include "videoAnalyser.hpp"
#include <opencv2/opencv.hpp>
#include <queue>

using namespace std;
using namespace cv;

int videoAnalyser::getQueueRemain()
{
    return _queue.size();
}

int videoAnalyser::imgLoad()
{
    cv::Mat *mat{new cv::Mat};
    if (!_capt.isOpened())
    {
        return 0;
    }
    _capt >> *mat;
    _queue.push(mat);
    return 1;
}