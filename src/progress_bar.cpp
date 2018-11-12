#include "progress_bar.hpp"
#include <string>
#include <vector>
#include <sstream>
#include <functional>

CprogressBar::CprogressBar()
{
    ;
}

CprogressBar::CprogressBar(std::string message)
{
    set_message(message);
}

void CprogressBar::set_message(std::string str)
{
    _msg = str;
}

std::string CprogressBar::generate_bar(double _value)
{
    std::stringstream ss;
    ss << _msg << bar_prefix;

    //calculate params
    int full_len{}, unfull_char_index{}, empty_len{};
    double fill_len{};
    fill_len = (_value)*bar_width;
    full_len = static_cast<int>((_value)*bar_width);                           // full width
    unfull_char_index = static_cast<int>((fill_len - full_len) * fill.size()); // which last char
    empty_len = bar_width - full_len + 1;                                      // empty length

    //generate string
    for (int i{}; i < full_len; i++)
        ss << fill.back();
    if (unfull_char_index > 0)
        ss << fill[unfull_char_index];
    if (0 < empty_len - 1)
        for (int i{}; i < empty_len; i++)
            ss << empty_fill;
    ss << bar_suffix;
    ss << static_cast<int>((_value)*100) << "%";
    return ss.str();
}
