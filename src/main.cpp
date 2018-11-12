//main.cpp
#include "opencv2/opencv.hpp"
#include <caffe/caffe.hpp>
#include <iostream>
#include <string>
#include <sstream>
#include <set>
#include <thread>
#include <chrono>
#include <functional>
#include "classifier.hpp"
#include "progress_bar.hpp"

using namespace std;
using namespace cv;

int main(int argc, char *argv[])
{
    cout << "Hello" << endl;
    CprogressBar a(string("Loading..."));
    for (double i = 0; i <= 100; i += 1)
    {
        cout << a.update(
            [=]() -> double {
                return i / 100;
            },
            [&]() -> string {
                stringstream ss;
                ss << " ETA:" << 100 - i;
                return ss.str();
            });
        this_thread::sleep_for(chrono::milliseconds(100));
    }
    return 1;
    ::google::InitGoogleLogging(argv[0]);

    std::string deploy_file{argv[1]};
    std::string caffemodel_file{argv[2]};
    std::string mean_file{argv[3]};
    std::string videoFilename{argv[4]};

    cout << "deploy_file:\t" << deploy_file << endl;
    cout << "caffemodel:\t" << caffemodel_file << endl;

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

    VideoWriter writer("videoOutput.avi",
                       captSource.get(CAP_PROP_FOURCC),
                       captSource.get(CAP_PROP_FPS),
                       Size(captSource.get(CAP_PROP_FRAME_WIDTH), captSource.get(CAP_PROP_FRAME_HEIGHT)));
    if (!writer.isOpened())
    {
        cout << "[Error!]: Create output video file failed" << endl;
        return 0;
    }

    int laststatus{-1};

    namedWindow("FireDetector", WINDOW_AUTOSIZE);
    vector<vector<float>> result;
    int count{};
    const double frameCount{captSource.get(CAP_PROP_FRAME_COUNT)};

    for (;;)
    {
        cv::Mat frame;
        captSource >> frame;
        if (frame.empty())
            break;
        vector<float> res = classifier.Classify(frame); //classify
        result.push_back(res);

        //output
        count++;
        if (count % 100 == 0)
        {
            cout << "Process:[" << count << "]" << endl;
        }

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

        // imshow("FireDetector", frame);
        // waitKey(delayBtFrame);

        // output file
        writer << frame;
    }

    count = 0;
    for (auto i = result.begin(); i != result.end(); i++)
    {
        cout << "====frame:" << ++count << "====" << endl;
        for (auto i2 = i->begin(); i2 != i->end(); i2++)
        {
            cout << *i2 << "\t";
        }
        cout << endl;
    }

    cout << "ok" << endl;

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