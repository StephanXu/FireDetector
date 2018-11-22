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
      m_previous_wnd(false)
{
    ;
}

int Cglobal::parse_params(int argc, char *argv[])
{
    cmdline::parser argv_parser;
    argv_parser.add<string>("deploy", 'd', "The deploy file of model.", false, "../example/example_deploy.prototxt");
    argv_parser.add<string>("model", 'm', "The '.caffemodel' file of model.", false, "../example/example_model.caffemodel");
    argv_parser.add<string>("mean", '\0', "Mean file", false, "../example/example_mean.binaryproto");
    argv_parser.add<string>("video", 'v', "The video to be detect.", true, "../example_testvideo.mp4");
    argv_parser.add("save", '\0', "Save the results.");
    argv_parser.add<string>("output", 'o', "Path to save the video processed.(need 'save' to be enabled)", false, "");
    argv_parser.add("view", '\0', "Show the process while processing video.");

    argv_parser.parse_check(argc, argv);

    global.m_deploy_file = argv_parser.get<string>("deploy");
    global.m_model_file = argv_parser.get<string>("model");
    global.m_mean_file = argv_parser.get<string>("mean");
    global.m_video_file = argv_parser.get<string>("video");
    global.m_save_result = argv_parser.exist("save");
    global.m_output_video = argv_parser.get<string>("output");
    global.m_previous_wnd = argv_parser.exist("view");
    return 1;
}

// void Cglobal::parse_command(int argc, char *argv[])
// {
//     for (int i{1}; i < argc; ++i)
//     {
//         string cmd(argv[i]);
//         for each (auto item in _parse_list)
//         {
//             if (item.first==cmd)
//             {
//                 if (item.second)
//                 {
//                     if (++i>=argc)
//                     {
//                         throw "argument parse fail";
//                     }

//                 }
//             }
//         }
//     }
// }