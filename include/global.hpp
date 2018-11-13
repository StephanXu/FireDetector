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

    void parse_command(int argc, char *argv[]);

    /* Configurations */
    string deploy_file;
    string model_file;
    string mean_file;
    string video_file;

    vector<pair<string, string>> _parse_list;
    // vector<pair<
    bool bPreviousWnd;
};

#endif