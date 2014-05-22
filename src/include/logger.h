#ifndef LOGGER_HPP
#define LOGGER_HPP

#include <iostream>
#include <string>

using namespace std;


class logger
{
public:
	bool enableXml;
	string xmlFile;

	logger(bool _enableXml=false, string _xmlFile="");
	~logger();

	// Overloaded function to print on stdout/android activity
	int print(string str);
	int print(double val);
	int print(float val);
	int print(int val);
	int print(unsigned int val);

	// Overloaded function to record metrics into xml file
	int record(string key, string value);
	int record(string key, double value);
	int record(string key, float value);
	int record(string key, int value);
	int record(string key, unsigned int value);

};

#endif  // LOGGER_HPP