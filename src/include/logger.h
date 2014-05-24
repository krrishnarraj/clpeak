#ifndef LOGGER_HPP
#define LOGGER_HPP

#include <iostream>
#include <string>
#include <fstream>
#include <xml_writer.h>

using namespace std;


class logger
{
public:
	bool enableXml;
	string xmlFileName;
	ofstream xmlFile;
	xmlWriter *xw;

	logger(bool _enableXml=false, string _xmlFileName="");
	~logger();

	// Overloaded function to print on stdout/android activity
	void print(string str);
	void print(double val);
	void print(float val);
	void print(int val);
	void print(unsigned int val);

	// Functions to record metrics into xml file
	void xmlOpenTag(string tag);
	void xmlAppendAttribs(string key, string value);
	void xmlCloseTag();

	void xmlRecord(string tag, string value);
	void xmlRecord(string tag, double value);
	void xmlRecord(string tag, float value);
};

#endif  // LOGGER_HPP