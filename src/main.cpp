//main.cpp
#include "opencv2/opencv.hpp"
#include <caffe/caffe.hpp>
#include <iostream>
#include <string>
#include <sstream>
#include <set>
#include "classifier.hpp"

using namespace std;
using namespace cv;

int main(int argc, char *argv[])
{
    ::google::InitGoogleLogging(argv[0]);

    std::string deploy_file{argv[0]};
    std::string caffemodule_file{argv[1]};
    std::string mean_file{argv[2]};
    std::string videoFilename{argv[3]};
    
    std::vector<std::string> labels{"fire", "normal", "smoke"};
    Cclassifier classifier(deploy_file, caffemodule_file, labels);

    classifier.SetMean(mean_file);
    
    int delayBtFrame{0};
    stringstream ss;
    ss << argv[4];
    ss >> delayBtFrame;

    VideoCapture captSource(videoFilename);
    if (!captSource.isOpened())
    {
        cout << "[Error!]: Open video failed";
        return 0;
    }

    //captSource.set(CAP_PROP_POS_MSEC, 140000);

    int laststatus{-1};

    namedWindow("Here I am", WINDOW_AUTOSIZE);
    vector<vector<float>> result;
    int count{};
    for (;;)
    {
        cv::Mat frame;
        captSource >> frame;
        if (frame.empty())
            break;

        imshow("Here I am", frame);
        waitKey(delayBtFrame);
        vector<float> res = classifier.Classify(frame);

        result.push_back(res);
        count++;
        if (count % 100 == 0)
        {
            cout << "Process:[" << count << "]" << endl;
        }
        if (count % 20 == 0)
        {
            if (res[0] >= res[1] && res[0] >= res[2])
                cout << "[fire]:";
            else if (res[1] >= res[0] && res[1] >= res[2])
                cout << "[normal]:";
            else if (res[2] >= res[0] && res[2] >= res[1])
                cout << "[smoke]:";
            cout << "\tfire:" << res[0] << "\tnormal:" << res[1] << "\tsmoke:" << res[2] << endl;
        }
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
