//main.cpp
#include <opencv2/opencv.hpp>
#include <caffe/caffe.hpp>
#include <iostream>
#include <string>
#include <sstream>
#include <thread>
#include <chrono>
#include <functional>
#include <tuple>
#include <deque>
#include "global.hpp"
#include "classifier.hpp"
#include "progress_bar.hpp"
#include "MotionAnalyser.hpp"
#include "cmdline.h"

using namespace std;
using namespace cv;

int main(int argc, char *argv[])
{
    ::google::InitGoogleLogging(argv[0]);

    // Parse params
    global.parse_params(argc, argv);
    cout << "deploy_file:\t" << global.m_deploy_file << endl;
    cout << "caffemodel:\t" << global.m_model_file << endl;
    cout << "=======Process video=======" << endl;

    // Init network
    std::vector<std::string> labels{"fire", "normal", "smoke"};
    Cclassifier classifier(global.m_deploy_file, global.m_model_file, labels);
    classifier.SetMean(global.m_mean_file);

    // Init video reader
    VideoCapture captSource(global.m_video_file);
    if (!captSource.isOpened())
    {
        cout << "[Error!]: Open video failed";
        return 0;
    }

    // Init video writer
    VideoWriter writer;
    if (global.m_save_result)
    {
        writer.open(global.m_output_video,
                    captSource.get(CAP_PROP_FOURCC),
                    captSource.get(CAP_PROP_FPS),
                    Size(captSource.get(CAP_PROP_FRAME_WIDTH), captSource.get(CAP_PROP_FRAME_HEIGHT)));
        if (!writer.isOpened())
        {
            cout << "[Error!]: Create output video file failed" << endl;
            return 0;
        }
    }

    // Init preview window
    if (global.m_previous_wnd)
        namedWindow("Fire Detector - view", WINDOW_AUTOSIZE);

    CMotionAnalyser ma;

    int count{};
    const double frameCount{captSource.get(CAP_PROP_FRAME_COUNT)};
    CprogressBar pb("Calculating...");
    double start_time = cv::getTickCount();
    ma.initialize_detect_object(global.m_sample_capacity);
    ma.generate_blocks(make_tuple(captSource.get(CAP_PROP_FRAME_WIDTH), captSource.get(CAP_PROP_FRAME_HEIGHT)), global.m_block_size);

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

        /* Create a mask for drawing */
        cv::Mat mask;
        frame.copyTo(mask);

        vector<float> res = classifier.Classify(frame); //classify
        string current_status{};
        int current_status_index;
        current_status_index = std::get<0>(classifier.GetResult(res));
        current_status = labels[current_status_index];

        Mat motion_map;
        vector<tuple<int, int, int, int>> motion_blocks;
        ma.feed_img(frame);
        ma.detect_motion(motion_map);
        if (count > global.m_sample_capacity)
        {
            ma.get_motion_blocks(ma._blocks, motion_map, motion_blocks);
            for (auto it{motion_blocks.begin()}; it != motion_blocks.end(); it++)
            {
                int x{}, y{}, w{}, h{};
                tie(x, y, w, h) = *it;
                // x++;
                // y++;
                // w--;
                // h--;
                Mat block = frame(Rect(x, y, w, h));
                vector<float> block_res = classifier.Classify(block);
                int current_block_status_index;
                current_block_status_index = std::get<0>(classifier.GetResult(block_res));
                if (current_block_status_index != 1 && current_block_status_index == current_status_index)
                {
                    rectangle(mask, Rect(x, y, w, h), global.c_mask_colors[current_block_status_index], CV_FILLED);
                }
            }
        }

        //output
        count++;

        // draw window
        rectangle(frame, Rect(20, 10, 300, 155), global.c_mask_colors[current_status_index], 1);
        stringstream ss;
        ss << "Frame:" << count << "/" << frameCount;
        putText(frame, ss.str(), Point(30, 30), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, global.c_mask_colors[current_status_index]);
        ss.str("");
        putText(frame, current_status, Point(30, 60), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, global.c_mask_colors[current_status_index]);
        ss << "Fire:" << res[0];
        putText(frame, ss.str(), Point(30, 90), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, global.c_mask_colors[current_status_index]);
        ss.str("");
        ss << "Normal:" << res[1];
        putText(frame, ss.str(), Point(30, 120), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, global.c_mask_colors[current_status_index]);
        ss.str("");
        ss << "Smoke:" << res[2];
        putText(frame, ss.str(), Point(30, 150), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, global.c_mask_colors[current_status_index]);
        ss.str("");

        cv::addWeighted(mask, global.c_mask_alpha, frame, 1.0 - global.c_mask_alpha, 0, frame);

        if (global.m_previous_wnd)
        {
            imshow("Fire Detector - view", frame);
            waitKey(1);
        }
        // output file
        if (global.m_save_result)
            writer << frame;
    }

    cout << flush << pb.update([&]() -> double { return 1; }, [&]() -> string { 
            stringstream ss;
            ss<<" FRAME:"<<frameCount<<"/"<<frameCount;
            return ss.str(); });

    cout << endl
         << "Process Over" << endl;
    cout << "Result:" << global.m_output_video << endl;
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

/*
作者：知道创宇 云安全
链接：https://www.zhihu.com/question/29962541/answer/433926110
来源：知乎
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

MMMMMMM
                                                    .MMMMMMMMH
                                                      .MMMMMMMMMMM
                                                        MMMMMMMMMMMM'
                                                         'MMMMMMMMMMMMM
                                                           MMMMMMMMMMMMMMM
                                                            MMMMMMMMMMMMMMMM.
                                                             .MMMMMMMMMMMMMMMMH          H
                                                              MMMMMMMMMMMMMMMMMM          M
                                                               :MMMMMMMMMMMMMMMMMM         M
                                                               :MMMMMMMMMMMMMMMMMMMM        M
                                                                .MMMMMMMMMMMMMMMMMMMMM       M
                                                                 MMMMMMMMMMMMMMMMMMMMMM:      M
                                                                 MMMMMMMMMMMMMMMMMMMMMMMM      M
                                                                  HMMMMMMMMMMMMMMMMMMMMMMM     HM
                                                                  MMMMMMMMMMMMMMMMMMMMMMMMM     .I
                                                                   MMMMMMMMMMMMMMMMMMMMMMMMM     M
                                                                   'MMMMMMMMMMMMMMMMMMMMMMMMM'    M
                                                                    MMMMMMMMMMMMMMMMMMMMMMMMMM.    M
                                                                    MMMMMMMMMMMMMMMMMMMMMMMMMMMH   MM
                                                                    :MMMMMMMMMMMMMMMMMMMMMMMMMMMM   M
                                                                     MMMMMMMMMMMMMMMMMMMMMMMMMMMM.   M
                                                                     MMMMMMMMMMMMMMMMMMMMMMMMMMMMMM   M
                ..MMMMMMMMMMMMMMMMMMMMMMMMMMM:M.                     :MMMMMMMMMMMMMMMMMMMMMMMMMMMMM:  H
      MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMH                MMMMMMMMMMMMMMMMMMMMMMMMMMMMMM   M
   MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM            MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM   M
    MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM         MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM  M
       MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM'      MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM. '.
          MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM    'MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM' M
            HMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM.''MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM.M
               MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM'MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM'M
                IMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM:M
                  MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
                    .MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM'
                      MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM'
                        MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
                         MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
                           .MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM                                            H'                    :
                            HMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM                                       MH                    :MM
                             :MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM                                   MM.                  MMMMM
                               MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM                                M:                MMMMMMMM.
                                MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM                           :M             'MMMMMMMMMMMM
                                 MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM                        MM           MMMMMMMMMMMMMMM
                                  .MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM                     MM       MMMMMMMMMMMMMMMMMMM
                                   .MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM.                 MM    MMMMMMMMMMMMMMMMMMMMMM
                  MMMMMM             HMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM               MM  MMMMMMMMMMMMMMMMMMMMMMMM
                        MMMM.         MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM            MMMMMMMMMMMMMMMMMMMMMMMMMMMM
                            MMM'       MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM'        MMMMMMMMMMMMMMMMMMMMMMMMMMMMM
                              MMMM      MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM    MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
                                MMMM    MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM:
                                  MMMM   MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
                                    MMM  HMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMH
               MMMMMMMMM:            .MM  MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
                 MMMMMMMMMMMMMMMMMMH   MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
                     MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM'MMMMMMMMMMMMMMMMMM.H'MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
                        MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM MMMMMMMM'MMMMMMMMMMMMMMMMMMO''MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM'
                           MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM MMMMMMMM''MMMMMMMMMMMMMMMMM.''MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM'
                             MMMMMMMMMMMMMMMMMMMMMMMMMMMMMM: MMMMMM.'' MMMMMMMMMMMMMMMMM'''MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM.
                               MMMMMMMMMMMMMMMMMMMMMMMMMMMMM' MMMMMM''':MMMMMMMMMMMMMM'M'''MMMMMMMMMMMMMMMM'MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
                                 .MMMMMMMMMMMMMMMMMMMMMMMMMMMM MMMMM''''MMMMMMMMMMMMMM'''''MMMMMMMMMMMMMMM'''MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
                                   IMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM'''''MMMMMMMMMMMMM.'''''MMMMMMMMMMMMM''''MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM.
                                     MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM'''''MMMMMMMMMMMMM''''''MMMMMMMMMMMMM''M'MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
                                       MMMMMMMMMMMMMMMMMMMMMMMMMM'MMM'''''MMMMMMMMMMMM''M''''MMMMMMMMMMM'''M'MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM           .MMMMMMMMMMMMMMMM'
                                        MMMMMMMMMMMMMMMMMM'MMMMMMM'MMM'''''MMMMMMMMMMMH'M''''MMMMMMMMMMM'''M'MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM'    MMMMMM
                                         'MMMMMMMMMMM'''MM'''MMMMM''.M.'''''MMMMMMMMMMM'M'''''MMMMMMMMM'''':'MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMH''
                                           .MMMMMMMMM'MM''M''''MMMM ''M''''''MMMMMMMMMM'M'''''MMMMMMMMM'''''M.MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM'.:
                                             MMMMMMMM.'''MM''''''MMMM''M'''''''MMMMMMMM'' '''''MMMMMMMMM''''M'MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM'
                                              MMMMMMM.''''MM'''M'''MM''''''''''''MMMMMMM'M'''MMMMMMMMMM:''''M'MMMMMMMMMMMM'''''MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM:M'
                                               MMMMMM  '.M'M'''M''''MM''''''''''''MMMMMM'M'MMMMMMMMMM''''''''M'MMMMMMMMMM''''''''MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
                                                MMMMM'M'M''M'''M'''''MMM''''''''''''MMMM'MMMMMMMMMMM'''''''''M'M MM''MMM'''MM'::''MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
                                      .MMM'      MMMM:''M''''''M'''''MMMM''''''''''''MMMMMMMMMMMMMMMM''''''''''M'MM''MM'''M''''''''MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
                                    'MMMMMMMMMMMMMMMMM'''M.MM''M'''''MMMMM''''''''''''MMMMMMMMMMMM''M''''''''''''MM''''''MM'''''M''MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
                                          MMMMMMMMMMMM''HMM'M'''M''''MMMMMMM'''''''.MMMMMMMMMMMMM'''M.'''''''''''MM'''''MM''M''''.'MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
                                            'MMMMMMMMMM''MMMM''''M''''MMMMMMM''''MMMMMMM'''''M.''''''M'''''''''''MM''''M'M'''M'''M'MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
                                               MMMMMMMM 'MMMM'''' M''''MM'MM .'''MMMMH'''''''''''''''''''''''''''M'''''M MM:'''''M'MMMMMMMMMMMMMMMMMMMMMMMMMMM:'
                                                 MMMMMMM''H'M''''''M.''''''':''''M:M'''''''''''''''''''''''''''''''''''MM'''''M''M'MMMMMMMMMMMMMMMMMMMMMMMM
                                                    MMMMM''MM'''''''''''''''''''''''M''''''''''''''''''''''''''''''''''MMHM'''M':'MMMMMMMMMMMMMMMMMMMMMH
                                                      MMMMM'M'''''''''''''''''''''''HMMM'''''''''''''''''''''''''''''''M''M'''M'''MMMMMMMMMMMMMMMMM:
                                                        MMMMM''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''.''M'''MMMMMMMMMMMMMM
                                                           M'''''''''':MM''''''''''''''''''''''''''''''''''''''''''''''''''M''M''MMMMMMMMMMMM
                                                           M'''''''''''M'''''''''''''''''''''''''''''''''''''''''''''''''''''M''MMMMMMMMMM
                                                          M''''''''''''''''''''''''''''''''''''''''''''''''''''''''MMM'''M'''''MMMMMMMM
                                                           M'''''''''''''''''''''''''''''''''''''''''''''''''''''''HM'.M''''''MMMMMMMMMMMMMMMMMMMMMMMM
                                                            M''''''''''''''''''''''''''''''''''''''''''''''''''H'''MM:''''''MMMMMMMMMMMMMMMMMMMMMMMMMMMM
                                                             HM''''''''''''''''''''''''''''''''''''''''''''''OMM:'''''''''MMMMMMMMMMMMMMMMMMM'
                                                               M:'''''''''HMMMM''''''''''''''''''''''''''''''MMMM.'''''MMMMMMMMMMMMM M
                                                                .M''''''MMO::::M''''''''''''''''''''''''''''MMMMMMMMMMMMMMMMMM.
                                                                  MM''''MMMHIHMM''''''''''''''''''''''''''MMMMMMMMMMMMMMM
                                                                   MM'''''''''''''''''''''''''''''''''''MMMMMMMMMMMMMMMMMM:
                                                                     M''''''''''''''''''''''''''''''MMMMMMMMMMMMMMMMMMMMM
                                                                      M ''''''''''''''''''''''MMMMM'''MMMMM
                                                                       MM''''''''''''''.MMMMM'''''''''MMM
                                                                        MMM''''''MMMMMM''''''''''''''''MM
                                                                        M'MMMMMMM''''''''''''''''''''''MM
                                                                        H'''''''''''''''''''''''''''''''M
                                                                       :''''''''''''''''''''''''''''''''M
                                                                  '    M'''''''''''''''''''''''''''''''':M
                                                                 MM:::MM'''''''''''''''''''''''''''''''''MMM       MMMMH    H'MMMMMM:HM
                                                                MH::MMM''''''''''''''''''''''''''''''''''''''MMMM'MH::::::::::::::::::::M.
                                                         MMMMMHMH::M:MM'''''''''''''''''''''''''''''''''''''''''MM:::::::::::::::IMM':MM::M
                                                     MMM::::::M::HM':M''''''''''''''''''''''''''''''''''''''''.MH::::::::::::::MM'''''''':MM
                                                MMMH:::::::::M::MM'''''''''''''''''''''''''''''''''''''''''''MM::::::::::::::M''''''''''''''':M
                                            HMMM:::::::::::MM::::M'''''''''''''''''''''''''':MHMMMMMMMMMM''.MM:::::::::::::MI'''''''''''''''''''MM
                                           M::::::::::::::MM:::::MMM'''''''''''''''''''HMMM'''''''''''''''MM::::::::::::::M'''''''''''''''''''''''MM
                                            M:::::::::::::M:::::::MMMM:''''''''''''''MM'''''''''''''''''MM:::::::::::::::MM'''''''''''''''''''''''''MM
                                            M::::::::::::::::::::::M'''M'''''''M'''M''''''''''''''''''MM:::::::::::::::::M'''''''''''''''''''''''''''MM
                                           MH:::H::::::::::::::::::M''''''''''..'::'''''''''''''''':MM:::::::::::::::::::M''''''''''''''''''''''''''''M
                                          M:':::M:::::::::::::::::::M''''''MMM:''''''''''''''''''MMH:::::::::::::::::::::M''''''''''''''''''''''''''''M
                                         M'''M::M::::H:::::::::::::::O''''''''''''''''''''''MMMM:::::::::::::::::::::::::M''''''''''''''''''''''''''''M
                                        '.'''M:M:::::M:::::::::::::::H''''''''''''''''''''MM:::::::::::::::::::::::::::::M''''''''''''''''''''''''''''M
                                       'M''''.HM::::::M:::H:::::::::::H'''M'''''''''''MMM::::::::::::::::::::::::::::::::M'''''.''''''''''''''''''''''M
                                       M''''''M::::::::::::M:::::::::::M''M''''''''MMH:::::::::::::::::::::::::::::::::::MM'''''''''''''''''''''''''''M'
                                      M'''''''M::::::::M::::H::::::::::::'H''''''MM::::::::::::::::::::::::::::::::::::::MM'''H:''''''''''''''''''''''MHM
                                      ''''''''M:::::::::M:::::::::::::::MM ''''M:::::::::::::::::::::::::::H:::::::::::::OM'''M'''''''''''''''''''''''M'HM
                                     'M''''''H:::::::::::MM::::M:::::::::M''M::::::::::::::::::::::::::::.::::::::::::::::M''MM'''''''''''''''''''''''''''M
                                      M''''''M:::::::::::MMM:::::::::::::HM::::::::::::::::::::::::::::M::::::::::::::::::MM.MM'''''''''''''''''''''''''''MM
                                      M''''''M::::::::::::MMH::::::::::::::::::::::::::::::::::::::::MH:::::::::::::::::::M:MMM''''''''''''''''''''''''''''H'
                                      H:'''''H:::::::::::::MMM::::::::::::::::::::::::::::::::::::::M:::::::::::::::::::::::::M'''''''''''''''''''''''''''''M
                                      M'''''M:::::::::::::::HMM:::::::::::::::::::::::::::::::::::M::::::::::::::M::::::::::::MO'''''''''''''''''''''''''''''M
                                      M'''''M::::::::::::::::::M:::::::::::::::::::::::::::::::HMM:::::::::::::::::M::::::::::MM''''''''''''''''''''''''''''':M
                                      M'''''M::::::::::::::::::::::::::::::::::::::::::::HMH::::::::::::::::::::M::::M::::::::MM''''''''''''''''''''''''''''''.'
                                     M.''''.::::::::::::::::::::::::::::::::::::::::HM::::::::::::::::::::::::::::M:::MH:::::::M'''''''''''''''''''''''''''''''M
                                     M'''''M::::::::::::::::::::::::::::::::::::MM::::::::::::::::::::::::::::::::::M:::MM:::::M'''''''''''''''''''''''''''''''':
                                    .M'''''M::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::M::::MM:MM''''''''''''''''''''''''''''''''M
                                    M''''''M::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::M:::::HM'''''''''''''''''''''''''''''''''M
                                   'M'''''.H:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::MM::IM''''''''''''''''''''''''''''''''M
                                   M''''''M:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::HMMMM''''''''''''''''''''''''''''''''MM
                                   M''''''M:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::MMM M''''''''''''''''''''''''''''''''MM
                                   M'''''.H::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::MM   :M''''''''''''''''''''''''''''''''M
                                  M.'''''M::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::HM     M''''''''''''''''''''''''''''''''M
                                  M''''''M::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::M      'M'''''''''''''''''''''''''''''''.'
                                  M''''''M:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::M::      .M'''''''''''''''''''''''''''''''M
                                 M'''''''MMI::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::H::M         M'''''''''''''''''''''''''''''M.
                                 M''''''''''MM:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::M:::M          'M'M''''''''''''''''''''''''''M.
                                 M.'''''''''''MM::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::M::::M            M:M''''''''''''''''''''''''''MM
                                  M'''''''''''''MM:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::M             HM''''''''''''''''''''''''''''MM
                                  M.''''''''''''''M:::::::::::::::::MMMMM::MM::::::::::::::::::::::::::::::::::::::::::::MM               M.''''''''''''''''''''''''''''MM
                                  'M''''''''''''''''MM::::::::::::OMM '  M   MM:HMMMMMMMMMMI:::::::::::::::::::::::::::MM                 MM'''''''''''''''''''''''''''''OM
                                  M'''''''''''''''''MMMMMMMMM'     M MM M   M .M                   '::'H:HMMMMM:::::MM.                   ::''''''''''''''''''''''''''''''M
                                  :''''''''''''''''''MM           M  H  '  MM   M :'...H                        'MMH                      M:'''''''''''''''''''''''''''''''M
                                 M''''''''''''''''''MMM      HMM: M    M  MMMM     MMM.'                          M                       MM'''''''''''''''''''''''''''''''M:
                                'M''''''''''''''''''''MMMMO      MM  M   .M MMH   M                                M                      MM'''''''''''''''''''''''''''''''.M
                                M:'''''''''''''''''''MM:           HH    M:  MM   M'                              M                       .M''''''''''''''''''''''''''''''''M
                                M''''''''''''''''''''MM       MMM  M  M   :   M:  I:::::::::MOHMMMM'             M                         MM'''''''''''''''''''''''''''''''M'
                                M''''''''''''''''''''':MHMO:::::  ':HMM:H:M   .M   M:::::::::M::::::::::M::IMMM M                           MM''''''''''''''''''''''''''''''MM
                                M'''''''''''''''''''''M::::::::M  O  :::::M:   M   M::::::::::M::::::::::M:::::HM                            MM''''''''''''''''''''''''''''':M
                                M'''''''''''''''''''':M::::::::H  M H::::::M   M   M:::::::::::M:::::::::HM:::::MM                            MM'''''''''''''''''''''''''''''M
                                M''''''''''''''''''''M::::::::M   M M::::::M   MO  ':::::::::::MM:::::::::M:::::::M                            MM''''''''''''''''''''''''''''MM
                                M'''''''''''''''''''M:::::::::M  H  M::::::M    M   M:::::::::::M::::::::::H::::::MM                            MM'''''''''''''MMMMMMMMMMMMMMMMMM
                                HM'''''''''''''''''HM:::::::::   M  M::::::M    M   M::::::::::::M:::::::::M:::::::M                             M'''''''''' MMMMMMMMMMMMMMMMMMMM
                                 MMMMMMMMMMMMMMMMMMM:::::::::M   M  H::::::M    M   M::::::::::::H:::::::::M:::::::MM                             M''''''HMMMMMMMMMMMMMMMMMMMMMMM
                                HMMMMMMMMMMMMMMMMMMM:::::::::   M'  M:::::::         M::::::::::::M:::::::::::::::::M                              M''''MMMMMMMMMMMMMMMMMMMMMMMMM
                                 MMMMMMMMMMMMMMMMMM:::::::::M   M   ::::::::M    .   M:::::::::::::M::::::::::::::::.M                              M'MMMMMMMMMMMMMMMMMMMMMMMMMMM.
                                 MMMMMMMMMMMMMMMMMMH::::::::M   M  '::::::::M    M   M:::::::::::::H:::::::::::::::::MH                              MMMMMMMMMMMMMMMMMMMMMMMMMMMMM
                                'MMMMMMMMMMMMMMMMMM:::::::::M   :  :::::::::M    M    M:::::::::::::::::::::::::::::::M                             MMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
                                MMMMMMMMMMMMMMMMMMM:::::::::    .  M::::::::M    '    M::::::::::::::::::::::::::::::::M                            'MMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
                                MMMMMMMMMMMMMMMMMMM::::::::M   M   M::::::::M         M:::::::::::::::::::::::::::::::::'                            MMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
                                MMMMMMMMMMMMMMMMMMM::::::::M   M   M::::::::M         M:::::::::::::::::::::::::::::::::M                             MMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
                                MMMMMMMMMMMMMMMMMMM::::::::M   M   M::::::::M          ::::::::::::::::::::::::::::::::::M                            MMMMMMMMMMMMMMMMMMMMMMMMMMMMMH
                                MMMMMMMMMMMMMMMMMMM::::::::    H   M::::::::M          M:::::::::::::::::::::::::::::::::M                             MMMMMMMMMMMMMMMMMMMMMMMMMMMM
                               :MMMMMMMMMMMMMMMMMMM:::::::H    '   H::::::::M          M::::::::::::::::::::::::::::::::::M                             MMMMMMMMMMMMMM:''''''''''M
                                MMMMMMMMMMMMMMMMMMM:::::::M        :::::::::M          M:::::::::::::::::::::::::::::::::::M                             MMMMMM.''''''''''''''''''.
                                 MMM:''''''''MMMMMM:::::::M   M   .:::::::::M'          O:::::::::::::::::::::M:::::::::::::M                             M'''''''''''''''''''''''M
                                 M'''''''''''''''MM:::::::M       M:::::::::M           M:::::::::::::::::::::M:::::::::::::M                            M''''''''''''''''''''''''M
                                 M''M''''''''''''MM:::::::        M:::::::::M           H:::::::::::::::::::::H:::::::::::::M.                         MM'''''''''''''''''''''''''.M
                                 M''M''''''''''''M::::::::        ::::::::::M.          M:::::::::::::::::::::H::::::::::::H:::                      MM'''''''''''''''''M''''''''''M
                                M.''M''''''''''''M:::::::M        .:::::::::M.          H::::::::::::::::::::::::::::::::::H::M                      M'''''''''''''''''''M'''''''''M
                                M'''M'''''''''''M::::::::M         :::::::::M.           ::::::::::::::::::::M:::::::::::::M::MM                     :'''''''''''''''''''M''''''''':
                                M'''M''''''''''MH::::::::M        ':::::::::M'           M:::::::::::::::::::M:::::::::::::M::M:M                     M'''.M''''''''M'''''''''''''''M
                                :''M''''''''''MM:::::::::'        '::::::::::M           M:::::::::::::::::::M:::::::::::::I:::::M                    M''''M''''''''M'''''M'''''''''M
                               M'''M'''''''''MM::::::::::          ::::::::::M           'H:::::::::::::::::M::::::::::::::::::::HM                   'M''''M'''''''M'''''M''''':'''M
                               M''''''''''''MM''H:::::::M         :::::::::::M            M:::::::::::::::::M:::::::::::::::::::::HM                   M'MMM:'''''''M'''''M'''''''''M
                               M''O'''''''IM'' :M:::::::M         :::::::::::M            M:::::::::::::::::M::::::::::::::::::::::M:                  M''''''''''''M''''''''''''MMM
                              M'''' M''''' ''''MM:::::::M          M:::::::::M          MMH::::::::::::::::M:::::M::::::::::::::::::M                  MM'''M''''''''M'''''M''''M
                              MM''MM'M''''M  M'MM:::::::M          H:::::::::M       .M::::::::::::::::::::M::::M::::::::::::::::::::M                    M'M:'''''''M''''MMMMMM
                               MM'':''''M'.''M:'M:::::::M          ::::::::::M      ::::::::::::::::::::::H:::::H:::::::::::::::::::::M                    M'M''''''''''''M
                                .M'''''':M''H'''M:::::::M MM:::::MMM:::::::::M    M:::::::::::::::::::::::M::::M::::::::::::::::::::::OM                   :M'''''''MMMMMM
                                 HM'''MM'M''M'''M:::::::M::::::::::::::::::::M   M:::::::::::::::::::::::HM:::M::::::::::::::::::::::::MM                    :''''MM
                                   MMMMMM''M MMMM::::::::::::::::::::::::::::M MM::::::::::::::::::::::::M::::M:::::::::::::::::::::::::MM                   'MMM
                                                M::::::::::::::::::::::::::::MHH:::::::::::::::::::::::::M:::M:::::::::::::::::::::::::::M.
                                               'M::::::::::::::::::::::::::::MMMM:::::::::::::::::::::::M:::M:::::::::::::::::::::::::::::M
                                               HM::::::::::::::::::::::::::::::MM:::::::::::::::::::::::M::OH::::::::::::::::::::::::::::::M
                                               MH:::::::::::::::::::::::::::::::M::::::::::::::::::::::M:::M:::::::::::::::::::::::::::::::M
                                               M::::::::::::::::::::::::::::::::M:::::::::::::::::::::OM::M::::::::::::::::::::::::::::::::M
                                               M::::::::::::::::::::::::::::::::MM::::::::::::::::::::M::MH::::::::::::::::::::::::::::::::M
                                               M::::::::::::::::::::::::::::::::OM::::::::::::::::::::M::M:::::::::::::::::::::::::::::::::MH'
                                               M::::M::::::::::::::::::::::::::::MM:::::::::::::::::::::M::::::::::::::::::::::::::::::::::M::M
                                               M::::M::::::::::::::::::::::::::::MM::::::::::::::::::::M::::::::::::::::::::::::::::::::::MM:::M
                                               M::::MH::::::::::::::::::::::::::::MH:::::::::::::::::::M::::::::::::::::::::::::::::::::::M::::I
                                               M::::HM::::::::::::::::::::::::::::MM::::::::::::::::::::::::::::::::::::::::::::::::::::::M::::.
                                              'M:::::M::::::::::::::::::::::::::::M:::::::::::::::::::::::::::::::::::::::::::::::::::::::M:::H
                                              MM:::::M:::::::::::::::::::::::::::::M::::::::::::::::::::::::::::::::::::::::::::::::::::::M:::OM
                                              MM:::::MH::::::::::::::::::::::::::::HH::::::::::::::::::::::::::::::::::::::::::::::::::::MM:::MMM
                                              MH::::::M::::::::::::::::::::::::::::MM::::::::::::::::::::::::::::::::::::::::::::::::::::M::::M:MM
                                              MH::::::M::::::::::::::::::::::::::::IM::::::::::::::::::::::::::::::::::::::::::::::::::::M::::M::M
                                              MH:::::::M:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::M::::M:::M:
                                              MH:::::::M::::::::::::::::::::::::::::M:::::::::::::::::::::::::::::::::::::::::::::::::::MM:::MM:::IM'
                                             'M::::::::HH:::::::::::::::::::::::::::.M::::::::::::::::::::::::::::::::::::::::::::::::::MH:::MM::::HMM
                                             IM:::::::::M:::::::::::::::::::::::::::HM::::::::::::::::::::::::::::::::::::::::::::::::::M::::M::::::MM:
                                             MM:::::::::H:::::::::::::::::::::::::::HM::::::::::::::::::::::::::::::::::::::::::::::::::M::::M::::::::M'
                                             MM::::::::::M:::::::::::::::::::::::::::M:::::::::::::::::::::::::::::::::::::::::::::::::HM::::M::::::::MM
                                             MM::::::::::MM::::::::::::::::::::::::::M:::::::::::::::::::::::::::::::::::::::::::::::::MH::::M::::::::.MM
                                             M::::::::::::M::::::::::::::::::::::::::MM::::::::::::::::::::::::::::::::::::::::::::::::M::::IM:::M:::::HMM
                                             M:::::::::::::M::::::::::::::::::::::::::M::::::::::::::::::::::::::::::::::::::::::::::::M:::::M:::M:::::::M.
                                            'M:::::::::::::MM:::::::::::::::::::::::::M:::::::::::::::::::::::::::::::::::::::::::::::::::::M::::MI::::::MM
                                            MH::::::::::::::MO::::::::::::::::::::::::M:::::::::::::::::::::::::::::::::::::::::::::::::::::M:::::M:::::::M
                                            MM:::::::::::::::M::::::::::::::::::::::::H::::::::::::::::::::::::::::::::::::::::::::::::::::HM:::::M:::::::MM
                                            M::::::::::::::::MM:::::::::::::::::::::::.::::::::::::::::::::::::::::::::::::::::::::::::::::M::::::M::::::::M
                                            M:::::::::::::::::M:::::::::::::::::::::::HM::::::::::::::::::::::::::::::::::::::::::::::::::HM::::::M:::::::::M
                                            M::::::::::::::::::MH::::::::::::::::::::::M::::::::::::::::::::::::::::::::::::::::::::::::::M:::::::M:::::::::M.
                                           :M::::::::::::::::::::::::::::::::::::::::::M:::::::::::::::::::::::::::::::::::::::::::::::::MM:::::::M::::::::::M
                                           'M::::::::::::::::::::::::::::::::::::::::::M:::::::::::::::::::::::::::::::::::::::::::::::::M::::::::M::::::::::M
                                            MH:::::::::::::::::::::::::::::::::::::::::M::::::::::::::::::::::::::::::::::::::::::::::::MM:::::::::::::::::::M
                                             MM::::::::::::::::::::::::::::::::::::::::HM::::::::::::::::::::::::::::::::::::::::::::::HM:::::::::H:::::::::M
                                              MM::::::::::::::::::::::::::::::::::::::::MM:::::::::::::::::::::::::::::::::::::::::::::MM::::::::M::::::::::M
                                               HM::::::::::::::::::::::::::::::::::::::::HMH::::::::::::::::::::::::::::::::::::::::::MM:::::::::M::::::::::M
                                                .MH:::::::::::::::::::::::::::::::::::::::HMM::::::::::::::::::::::::::::::::::::::::HMM:::::::::M:::::::::M
                                                  MM::::::::::::::::::::::::::::::::::::::MMMMMM:::::::::::::::::::::::::::::::::::::MH::::::::::M:::::::::M
                                                   MM::::::::::::::::::::::::::::::::::::MM:::MMMH::::::::::::::::::::::::::::::::::::::::::::::MM::::::::::
                                                    MMH:::::::::::::::::::::::::::::::::MM:::::M: MM::::::::::::::::::::::::::::::::::::::::::::M:::::::::M
                                                    MMMM::::::::::::::::::::::::::::::MMI::::::M  MMMM:::::::::::::::::::::::::::::::::::::::::M::::::::::M
                                                   MM:MMM:::::::::::::::::::::::::::MMM::::::::M  'M::MH:::::::::::::::::::::::::::::::::::::::M::::::::::H
                                                   HM:::MMM:::::::::::::::::::::::IMM::::::::::M   M::::MM::::::::::::::::::::::::::::::::::::MM:::::::::M
                                                   IM::::MMM:::::::::::::::::::::MM::::::MM:::M     ::::::MM::::::::::::::::::::::::::::::::::MO:::::::::M
                                                   MM::::::MMM:::::::::::::::::MM:::::::M:M:::M     M:::::::MM:::::::::::::::::::::::::::::::MM:::::::::M
                                                   MM::::::MOMM::::::::::::::MM::::::::M::::::M     M:::::::::HM:::::::::::::::::::::::::::::MM:::::::::M
                                                   MM::::::M::IMM:::::::::MMM::::::::HM:::::::M     .:::::::::::MM::::M:::::::::::::::::::::HM:::::::::M
                                                   MM::::::M::::MM::::::M:::::::::::MM::::::::M     I:::::::::::::MM::::M:::::::::::::::::::M::::::::::M
                                                   MH::::::M:::::MM::::::::::::::::MM:::::::::M      O::::::::::::::M::::MM::::::::::::::::HM:::::::::M
                                                   MM::::::MH:::::M:::::::::::::::MI::::::::OMH      'M::::::::::::::MH::::MM::::::::::::::M:::::::::HM
                                                   MM:::::::M:::::H::::::::::::::M::::::::HMH           MM::::::::::::H::::::M:::::::::::::::::::::::M'
                                                    MM::::::M::::::::::::::::::HM::::::::M'               MM:::::::::::H::::::MM:::::::::::::::::::::M
                                                     MM:::::H::::::::::::::::::M::::::MM                     MI:::::::::::::::::MM::::::::::::::::::MM
                                                      MM:::::M::::::::::::::::M:::::MM                         MM::::::::::::::::MMM::::::::::::::::M
                                                        MH::::M::::::::::::::M::HMMMM                            MMM:::::::::::::::MMM:::::::::::::MM'
                                                         MMMMMMMMMMMMMMMMMMMMMMMMMMMM                              'MM::::::::::::::MMMMM:::::::::MMM
                                                          MMMMMMMMMMMMMMMMMMMMMMMMMMM                                 MM:::::::::::::MMMM:::MMMMMMMMM
                                                           MMMMMMMMMMMMMMMMMMMMMMM.M                                   :MMMMMMMMMMMMMMMMMMMMMMMMMMMM
                                                            M''''''''''''''''''''''MM                                   MMMMMMMMMMMMMMMMMMMMMMMMMMM'M
                                                            M'''''''''''''''''''''''M                                    MMMMMMMMMMMMMMMMMMMMMMM:'''''
                                                          O''''''''''''''''''''''''''                                     MMMMMMMMMM.I.''''''''''''''M
                                                          H''M'''''''''''''''''''''''M                                     M''''''''''''''''''''''''''M
                                                         MM'''' :MM'''''''''''''''''''M                                   'M''.MH:''''''''''''''''''''M
                                                      MM'''''''''''''''''''''''''''''HM                                   .M.''''''''''''''''''''''''''M
                                                   MM:''''''''''''''''''''''''''''''MMMM                                  MM'''''''''''''''''''''''''''M
                                                MMMMMMM'''''''''''''''''''''''''' MMMMMM                                 'M'''''''IHMMMMMMMM '''''''''''M
                                             MMMMMMMMMMM:'''''''''''''''''''''HMMMMMMMMM                                 MH'':MMMMMMMMMMMMMMMMM'''''''''M
                                          'MMMMMMMMMMMMMMMH''''''''''HMMMMMMMMMMMMMMMMMMM                               OM'MMMMMMMMMMMMMMMMMMMMMH'''''''MM
                                        MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM                               M'MMMMMMMMMMMMMMMMMMMMMMMM '''''MM
                                     'MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM                               MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM:
                                     MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM                              MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
                                    MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM                               MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
                                    MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM.                                 MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
                                    MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM:                                     'MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
                                     .MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM                                               HMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMH
                                            'MMMMMMMMMM:                                                             MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
                                                                                                                      MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
                                                                                                                      MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
                                                                                                                      MMMMMMMMMMMMMMMMMMMMMMMMMMMM.
*/