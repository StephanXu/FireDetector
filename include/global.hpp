#pragma once

#ifndef MAIN_HPP
#define MAIN_HPP

#include <string>
#include <vector>

using namespace std;

class Cglobal
{
  public:
    explicit Cglobal();

    int parse_params(int argc, char *argv[]);

    /* Configurations */
    string m_deploy_file;
    string m_model_file;
    string m_mean_file;
    string m_video_file;
    bool m_save_result;
    string m_output_video;
    bool m_previous_wnd;
};

extern Cglobal global;

#endif