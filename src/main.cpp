//main.cpp
#include <opencv2/opencv.hpp>
#include <caffe/caffe.hpp>
#include <iostream>
#include <string>
#include <sstream>
#include <set>
#include <thread>
#include <chrono>
#include <functional>
#include <tuple>
#include "global.hpp"
#include "classifier.hpp"
#include "progress_bar.hpp"

using namespace std;
using namespace cv;

class CMotionAnalyser
{
  public:
    vector<tuple<int, int, int, int>> _blocks;

    int generate_blocks(tuple<int, int> size, int block_size);

    int detect_motion_optical_flow_in_block(const cv::Mat &img_mat,
                                            const cv::Mat &pre_img_mat,
                                            const int block_size);
};

int CMotionAnalyser::generate_blocks(tuple<int, int> size, int block_size)
{
    for (int y{}; y < std::get<1>(size) - block_size; y += block_size)
    {
        for (int x{}; x < std::get<0>(size); x += block_size)
        {
            _blocks.push_back(make_tuple(x, y, block_size, block_size));
        }
    }
    return 0;
}

int CMotionAnalyser::detect_motion_optical_flow_in_block(const cv::Mat &img_mat,
                                                         const cv::Mat &pre_img_mat,
                                                         const int block_size)
{
    Mat mgrey, mpre_grey;
    /* Convert image to grey */
    cvtColor(img_mat, mgrey, CV_BGR2GRAY);
    cvtColor(pre_img_mat, mpre_grey, CV_BGR2GRAY);

    
    vector<Point2f> curt_points, prev_points;

    goodFeaturesToTrack(mgrey, curt_points, 500, 0.001, 10);
    // calcOpticalFlowPyrLK(pre_img_mat,img_mat,prev_points,curt_points,)
    for (auto it{curt_points.begin()}; it != curt_points.end();it++)
    {
        circle(img_mat, *it, 2, Scalar(0, 0, 255));
    }
}

int main(int argc, char *argv[])
{
    ::google::InitGoogleLogging(argv[0]);

    std::string deploy_file{argv[1]};
    std::string caffemodel_file{argv[2]};
    std::string mean_file{argv[3]};
    std::string videoFilename{argv[4]};
    std::string outputFilename{argv[6]};
    std::string strVision{argv[7]};
    std::string strImageSeq{argv[8]};
    std::string ImageSeq_path{argv[9]};

    bool bVision{false};
    if (strVision == "yes")
        bVision = true;
    else
        bVision = false;

    bool bImageSeq{false};
    if (strImageSeq == "yes")
        bImageSeq = true;
    else
        bImageSeq = false;

    cout << "deploy_file:\t" << deploy_file << endl;
    cout << "caffemodel:\t" << caffemodel_file << endl;
    cout << "=======Process video=======" << endl;

    std::vector<std::string> labels{"fire", "normal", "smoke"};
    Cclassifier classifier(deploy_file, caffemodel_file, labels);

    classifier.SetMean(mean_file);

    int delayBtFrame{0};
    stringstream ss;
    ss << argv[5];
    ss >> delayBtFrame;

    VideoCapture captSource(videoFilename);
    if (!captSource.isOpened())
    {
        cout << "[Error!]: Open video failed";
        return 0;
    }

    CMotionAnalyser ma;
    
    // VideoWriter writer(outputFilename,
    //                    captSource.get(CAP_PROP_FOURCC),
    //                    captSource.get(CAP_PROP_FPS),
    //                    Size(captSource.get(CAP_PROP_FRAME_WIDTH), captSource.get(CAP_PROP_FRAME_HEIGHT)));
    // if (!writer.isOpened())
    // {
    //     cout << "[Error!]: Create output video file failed" << endl;
    //     return 0;
    // }

    int laststatus{-1};
    if (bVision)
        namedWindow("FireDetector", WINDOW_AUTOSIZE);
    // vector<vector<float>> result;
    int count{};
    const double frameCount{captSource.get(CAP_PROP_FRAME_COUNT)};

    CprogressBar pb("Calculating...");
    double start_time = cv::getTickCount();
    for (;;)
    {

        cout << flush << pb.update([&]() -> double { return count / frameCount; }, [&]() -> string { 
            stringstream ss;
            ss<<" FRAME:"<<count<<"/"<<frameCount;
            return ss.str(); });

        cv::Mat frame;
        captSource >> frame;
        if (frame.empty())
            break;

        ma.detect_motion_optical_flow_in_block(frame, frame, 24);

        // vector<float> res = classifier.Classify(frame); //classify
       
        //output
        count++;
/*
        string current_status{};
        Scalar text_color = Scalar(0, 0, 0);
        if (res[0] >= res[1] && res[0] >= res[2])
        {
            current_status = "[fire]";
            text_color = Scalar(0, 0, 255);
        }
        else if (res[1] >= res[0] && res[1] >= res[2])
        {
            current_status = "[normal]";
            text_color = Scalar(0, 255, 0);
        }
        else if (res[2] >= res[0] && res[2] >= res[1])
        {
            current_status = "[smoke]";
            text_color = Scalar(255, 0, 0);
        }

        // draw window
        rectangle(frame, Rect(20, 10, 300, 160), text_color, 1);
        stringstream ss;
        ss << "Frame:" << count << "/" << frameCount;
        putText(frame, ss.str(), Point(30, 30), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, text_color);
        ss.str("");
        putText(frame, current_status, Point(30, 60), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, text_color);
        ss << "Fire:" << res[0];
        putText(frame, ss.str(), Point(30, 90), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, text_color);
        ss.str("");
        ss << "Normal:" << res[1];
        putText(frame, ss.str(), Point(30, 120), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, text_color);
        ss.str("");
        ss << "Smoke:" << res[2];
        putText(frame, ss.str(), Point(30, 150), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, text_color);
        ss.str("");
*/
        if (bVision)
        {
            imshow("FireDetector", frame);
            waitKey(delayBtFrame);
        }
        // output file
        // if (!bImageSeq)
        //     writer << frame;
        // else
        // {
        //     stringstream oss;
        //     oss << ImageSeq_path << count << ".jpg";
        //     imwrite(oss.str(), frame);
        // }
    }

    cout << flush << pb.update([&]() -> double { return 1; }, [&]() -> string { 
            stringstream ss;
            ss<<" FRAME:"<<frameCount<<"/"<<frameCount;
            return ss.str(); });

    cout << endl
         << "Process Over" << endl;
    cout << "Result:" << outputFilename << endl;
    double end_time = getTickCount();
    cout << "Cost time:" << (end_time - start_time) * 1000 / (getTickFrequency())
         << "ms" << endl;
    return 1;
}

//
//                       _oo0oo_
//                      o8888888o
//                      88" . "88
//                      (| -_- |)
//                      0\  =  /0
//                    ___/`---'\___
//                  .' \\|     |// '.
//                 / \\|||  :  |||// \
//                / _||||| -:- |||||- \
//               |   | \\\  -  /// |   |
//               | \_|  ''\---/''  |_/ |
//               \  .-\__  '-'  ___/-. /
//             ___'. .'  /--.--\  `. .'___
//          ."" '<  `.___\_<|>_/___.' >' "".
//         | | :  `- \`.;`\ _ /`;.`/ - ` : | |
//         \  \ `_.   \_ __\ /__ _/   .-` /  /
//     =====`-.____`.___ \_____/___.-`___.-'=====
//                       `=---='
//
//
//     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//               佛祖保佑         永无BUG
//
//
