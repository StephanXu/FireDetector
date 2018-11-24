#pragma once

#ifndef TASKCONTROL_HPP
#define TASKCONTROL_HPP

#include "classifier.hpp"
#include <queue>
#include <string>
#include <thread>
#include <chrono>

using namespace std;

class CtaskControl
{
public:
};

class CAsyncAnalyseTaskControl : public CtaskControl
{
private:
  /* Here are flags */
public:
  explicit CAsyncAnalyseTaskControl();

  ~CAsyncAnalyseTaskControl();

  int getClassifier();

  void Initialize(const std::string &deploy_file,
                  const std::string &caffemodel,
                  const std::string &mean_file,
                  int async_num);

  vector<vector<float>> BatchClassify(const vector<tuple<int, int, int, int>> &block_list,
                                      const cv::Mat &img);

  vector<vector<float>> BatchClassifyDispatch(const vector<tuple<int, int, int, int>> &block_list,
                                              const cv::Mat &img);

private:
  vector<pair<bool, Cclassifier *>> _objects;
};

#endif