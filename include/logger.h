#ifndef LOGGER_HPP
#define LOGGER_HPP

/*
 * ANDROID_LOGGER -- defined only incase of android ndk build
 */

#include <iostream>
#include <string>
#include <fstream>
#include <xml_writer.h>
#include "common.h"

#ifdef ANDROID_LOGGER
#include <jni.h>
#endif

using namespace std;

class logger
{
public:
  bool enableXml;
  ofstream xmlFile;
  xmlWriter *xw;

#ifdef ANDROID_LOGGER
  JNIEnv *jEnv;
  jobject *jObj;
  jmethodID printCallback;
#endif

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
  void xmlAppendAttribs(string key, uint value);
  void xmlSetContent(string value);
  void xmlSetContent(float value);
  void xmlCloseTag();

  void xmlRecord(string tag, string value);
  void xmlRecord(string tag, float value);
};

#endif  // LOGGER_HPP
