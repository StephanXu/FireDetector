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
#include "taskControl.hpp"

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
  Cclassifier classifier(global.m_deploy_file, global.m_model_file);
  classifier.SetMean(global.m_mean_file);

  CAsyncAnalyseTaskControl async_classifier;
  if (global.m_enable_multithread)
    async_classifier.Initialize(global.m_deploy_file, global.m_model_file, global.m_mean_file, global.m_multithread_num);

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

    if (!global.m_disable_motion_block)
    {
      Mat motion_map;
      vector<tuple<int, int, int, int>> motion_blocks;
      ma.feed_img(frame);
      ma.detect_motion(motion_map);

      if (count > global.m_sample_capacity)
      {
        ma.get_motion_blocks(ma._blocks, motion_map, motion_blocks);
        if (!global.m_enable_multithread)
        {
          for (auto it{motion_blocks.begin()}; it != motion_blocks.end(); it++)
          {
            int x{}, y{}, w{}, h{};
            tie(x, y, w, h) = *it;
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
        else
        {
          vector<vector<float>> block_results = async_classifier.BatchClassify(motion_blocks, frame);
          for (unsigned int i{}; i < block_results.size(); i++)
          {
            int x{}, y{}, w{}, h{};
            tie(x, y, w, h) = motion_blocks[i];
            int current_block_status_index;
            current_block_status_index = std::get<0>(classifier.GetResult(block_results[i]));
            if (current_block_status_index != 1 && current_block_status_index == current_status_index)
            {
              rectangle(mask, Rect(x, y, w, h), global.c_mask_colors[current_block_status_index], CV_FILLED);
            }
          }
        }
      }
    }

    //output
    count++;

    // draw window
    cv::addWeighted(mask, global.c_mask_alpha, frame, 1.0 - global.c_mask_alpha, 0, frame);

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

    /* Rend window */
    if (global.m_previous_wnd)
    {
      imshow("Fire Detector - view", frame);
      waitKey(1);
    }
    /* output file */
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