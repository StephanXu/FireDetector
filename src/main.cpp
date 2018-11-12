//main.cpp
#include "opencv2/opencv.hpp"
#include <caffe/caffe.hpp>
#include <iostream>
#include <string>
#include <sstream>

#include "classifier.hpp"

using namespace std;
using namespace cv;

int main(int argc, char *argv[])
{
    std::string deploy_file{"/home/lorime/stephanxu/FireDetector/example/deploy.prototxt"};
    std::string caffemodule_file{"/home/lorime/stephanxu/FireDetector/example/final_model.caffemodel"};
    std::vector<std::string> labels{"fire", "normal", "smoke"};
    Cclassifier classifier(deploy_file, caffemodule_file, labels);

    cv::Mat img{cv::imread("./testimg.jpg")};
    std::vector<float> result = classifier.Classify(img);

    for (auto i{result.begin()}; i < result.end(); i++)
    {
        cout << *i << endl;
    }

    return 1;

    // Mat image = imread("./testimg.jpg");
    // cv::imshow("Here I am", image);
    // cv::waitKey(0);
    if (argc != 3)
    {
        cout << "Bad paramters!";
        return 0;
    }

    std::string videoFilename{argv[1]};
    int delayBtFrame{0};
    stringstream ss;
    ss << argv[2];
    ss >> delayBtFrame;

    VideoCapture captSource(videoFilename);
    if (!captSource.isOpened())
    {
        cout << "[Error!]: Open video failed";
        return 0;
    }

    namedWindow("Here I am", WINDOW_AUTOSIZE);
    Mat frame;
    for (captSource >> frame; !frame.empty(); captSource >> frame)
    {
        imshow("Here I am", frame);
        waitKey(delayBtFrame);
    }

    cout << "ok" << endl;

    return 1;
}
