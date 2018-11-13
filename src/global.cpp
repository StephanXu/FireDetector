#include "global.hpp"
#include <string>

using namespace std;

Cglobal::Cglobal()
    : deploy_file(""), model_file(""), mean_file(""), video_file(""), bPreviousWnd(false)
{
    ;
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