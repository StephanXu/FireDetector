#pragma once

#ifndef PROGRESS_BAR_HPP
#define PROGRESS_BAR_HPP

#include <string>
#include <vector>
#include <sstream>
#include <functional>
#include <thread>
#include <chrono>
#include <iostream>

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
	template <typename refreshT, typename sufT>
	std::string update(refreshT const &refresh_proc, sufT const &suffix_proc)
	{
		std::stringstream ss;
		ss << "\r" << generate_bar(refresh_proc()) << suffix_proc() << "";
		return ss.str();
	}

	/*
	Here is a function to create a thread auto refresh the view of progress bar
	Just provide a refresh process, suffix process and a callback function could be called when
		'value' reach to 1(100%)
	Argument:
		refresh_proc and suffix_proc please see function 'update''s notes.
		callback: process could be called when 'value' reach to 1(100%)
	Example:
		CprogressBar pb;
		double i{};
		auto refresher = [&]() {
			return i / 100;
		};
		auto suffix = [&]() {
			return " ETA";
		};
		auto callbacks = [&]() {
			cout << "we've done" << endl;
		};
		pb.auto_refresh_begin(refresher, suffix, callbacks);
		for (;i<=100;i+=1)
		{
			this_thread::sleep_for(chrono::milliseconds(100));
		}
		cout << "we've really done" << endl;
		return 1;
	*/
	template <typename refreshT, typename sufT, typename callbackT>
	void auto_refresh_begin(refreshT const &refresh_proc, sufT const &suffix_proc, callbackT const &callback, int refresh_delay = 1)
	{
		std::thread t([&]() {
			for (; refresh_proc() < 1; std::this_thread::sleep_for(std::chrono::milliseconds(refresh_delay)))
			{
				std::cout << std::flush << this->update(refresh_proc, suffix_proc);
			}
			std::cout << std::flush << this->update(refresh_proc, suffix_proc) << std::endl;
			callback();
		});
		t.detach();
	}

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

	std::thread *_pthread;
};

#endif
