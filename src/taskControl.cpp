#include "taskControl.hpp"
#include <queue>
#include <string>
#include <thread>
#include <chrono>
#include <future>

using namespace std;

CAsyncAnalyseTaskControl::CAsyncAnalyseTaskControl()
{
    return;
}

CAsyncAnalyseTaskControl::~CAsyncAnalyseTaskControl()
{
    for (auto it{_objects.begin()}; it != _objects.end(); it++)
    {
        delete it->second;
    }
}

int CAsyncAnalyseTaskControl::getClassifier()
{
    int classifier_index{};
    auto it{_objects.begin()};
    for (; it != _objects.end(); it++)
    {
        if (it->first)
        {
            it->first = false;
            classifier_index = it - _objects.begin();
            break;
        }
    }
    if (it == _objects.end())
    {
        return -1;
    }
    return classifier_index;
}

void CAsyncAnalyseTaskControl::Initialize(const std::string &deploy_file,
                                          const std::string &caffemodel,
                                          const std::string &mean_file,
                                          int async_num)
{
    for (int i{}; i < async_num; i++)
    {
        Cclassifier *pclassifier{new Cclassifier(deploy_file, caffemodel)};
        pclassifier->SetMean(mean_file);
        _objects.push_back(make_pair(true, pclassifier));
    }
    return;
}

vector<vector<float>> CAsyncAnalyseTaskControl::BatchClassifyDispatch(const vector<tuple<int, int, int, int>> &block_list,
                                                                      const cv::Mat &img)
{
    vector<vector<float>> results;
    vector<future<vector<float>>> futures(block_list.size());

    unsigned int count{};
    for (auto it{block_list.begin()}; it != block_list.end(); it++)
    {
        int classifier_index{-1};
        cv::Mat block_img;
        do
        {
            classifier_index = getClassifier();
        } while (classifier_index == -1);

        int x{}, y{}, w{}, h{};
        tie(x, y, w, h) = *it;
        block_img = img(cv::Rect(x, y, w, h));

        auto process_proc = [this](int classifier_index, const cv::Mat &img) -> vector<float> {
            vector<float> res = this->_objects[classifier_index].second->Classify(img);
            this->_objects[classifier_index].first = true;
            return res;
        };

        futures[count] = async(std::launch::async, process_proc, classifier_index, block_img);
        count++;
    }

    for (count = 0; count < block_list.size(); count++)
    {
        results[count] = futures[count].get();
    }

    return results;
}

vector<vector<float>> CAsyncAnalyseTaskControl::BatchClassify(const vector<tuple<int, int, int, int>> &block_list,
                                                              const cv::Mat &img)
{
    vector<vector<float>> results;

    int enable_thread{};
    if (block_list.size() >= _objects.size())
    {
        enable_thread = _objects.size();
    }
    else
    {
        enable_thread = block_list.size();
    }

    vector<future<vector<vector<float>>>> futures(enable_thread);

    int task_size{static_cast<int>(block_list.size()) / enable_thread};

    for (int i{}; i < enable_thread; i++)
    {
        auto batch_process_proc = [&](int classifier_index, int min, int size, const cv::Mat &img) -> vector<vector<float>> {
            vector<vector<float>> block_result;
            for (int j{min}; j < min + size; j++)
            {
                int x{}, y{}, w{}, h{};
                tie(x, y, w, h) = block_list[j];
                cv::Mat block_img = img(cv::Rect(x, y, w, h));
                vector<float> res = this->_objects[classifier_index].second->Classify(block_img);
                block_result.push_back(res);
            }
            return block_result;
        };
        futures[i] = async(std::launch::async, batch_process_proc, i, i * task_size, task_size, img);
    }

    for (auto it{futures.begin()}; it != futures.end(); it++)
    {
        vector<vector<float>> block_result = it->get();
        results.insert(results.end(), block_result.begin(), block_result.end());
    }

    return results;
}
