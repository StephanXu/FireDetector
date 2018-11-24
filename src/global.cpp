#include "global.hpp"
#include "cmdline.h"
#include <iostream>
#include <string>

using namespace std;

Cglobal global;

Cglobal::Cglobal()
    : m_deploy_file(""),
      m_model_file(""),
      m_mean_file(""),
      m_video_file(""),
      m_save_result(false),
      m_output_video(""),
      m_previous_wnd(false),
      m_multithread_num(1),
      m_enable_multithread(false),
      m_disable_motion_block(false),
      m_sample_capacity(8),
      m_block_size(48)
{
    ;
}

int Cglobal::parse_params(int argc, char *argv[])
{
    cmdline::parser argv_parser;
    /* Basic configurations */
    argv_parser.add<string>("deploy", 'd', "The deploy file of model.", false, "../example/example_deploy.prototxt");
    argv_parser.add<string>("model", 'm', "The '.caffemodel' file of model.", false, "../example/example_model.caffemodel");
    argv_parser.add<string>("mean", '\0', "Mean file", false, "../example/example_mean.binaryproto");
    argv_parser.add<string>("video", 'v', "The video to be detect.", true, "../example_testvideo.mp4");
    argv_parser.add("save", '\0', "Save the results.");
    argv_parser.add<string>("output", 'o', "Path to save the video processed.(need 'save' to be enabled)", false, "");
    argv_parser.add("view", '\0', "Show the process while processing video.");
    argv_parser.add("parallel", 'p', "Parallel processing.");
    argv_parser.add<int>("multi", 't', "The number of multithread", false, 1, cmdline::range(1, 100));
    argv_parser.add("no_MB", '\0', "Disable the motion blocks");

    /* Advance configurations */
    argv_parser.add<int>("sample", 's', "Capacity of sample frames for motion detecting.", false, 8, cmdline::range(2, 100));
    argv_parser.add<int>("block_size", 'b', "Size of motion blocks.", false, 48, cmdline::range(24, 1000));

    argv_parser.parse_check(argc, argv);

    m_deploy_file = argv_parser.get<string>("deploy");
    m_model_file = argv_parser.get<string>("model");
    m_mean_file = argv_parser.get<string>("mean");
    m_video_file = argv_parser.get<string>("video");
    m_save_result = argv_parser.exist("save");
    m_output_video = argv_parser.get<string>("output");
    m_previous_wnd = argv_parser.exist("view");
    m_enable_multithread = argv_parser.exist("parallel");
    m_multithread_num = argv_parser.get<int>("multi");
    m_disable_motion_block = argv_parser.exist("no_MB");

    m_sample_capacity = argv_parser.get<int>("sample");
    m_block_size = argv_parser.get<int>("block_size");
    return 1;
}
