#pragma once

#ifndef PROGRESS_BAR_HPP
#define PROGRESS_BAR_HPP

#include <string>
#include <vector>
#include <functional>
class CprogressBar
{
  public:
    explicit CprogressBar();

    // constuctor with a message
    explicit CprogressBar(std::string message);

    /*
	Here we provide 2 functional params to refresh your data.
	You can simply call this function to generate progress bar.
	Arguments:
		refresh_proc: process to refresh percentage
		suffix_proc: process to generate suffix where after the progress bar
	Example:
		Here is an example:
			CprogressBar a(string("Loading..."));
			for (double i = 0; i <= 100; i += 1)
			{
				cout << a.update(
					[=]()->double {
					return i / 100;
				},
					[&]()->string {
					stringstream ss;
					ss << " ETA:" << 100 - i;
					return ss.str();
				});
				this_thread::sleep_for(100ms);
			}
			cout<<endl;
		It looks like:
			Loading... |████████████████████████████████| 100% ETA:0
	*/
    template <typename refreshT, typename sufT2>
    std::string update(refreshT const &refresh_proc, sufT2 const &suffix_proc);

    void set_message(std::string str);

  private:
    const int bar_width = 32;
    const std::string bar_prefix{" |"};
    const std::string bar_suffix{"| "};
    const std::string empty_fill{" "};
    const std::vector<std::string> fill{" ", "▏", "▎", "▍", "▌", "▋", "▊", "▉", "█"};
    //const std::vector<std::string> fill{ " ","▌","█" };
  private:
    std::string _msg;

    std::string generate_bar(double _value);
};

#endif