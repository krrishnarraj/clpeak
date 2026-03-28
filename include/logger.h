#ifndef LOGGER_HPP
#define LOGGER_HPP

/*
 * ANDROID_LOGGER -- defined only incase of android ndk build
 */

#include <iostream>
#include <string>
#include <fstream>
#include <map>
#include <vector>
#include <xml_writer.h>
#include <result_store.h>
#include "common.h"

#ifdef ANDROID_LOGGER
#include <jni.h>
#endif

using namespace std;

// One frame on the XML context stack.  Tracks the tag name and all key-value
// attributes set via xmlAppendAttribs so that xmlRecord can build a fully
// qualified ResultEntry without needing changes in the benchmark code.
struct ContextFrame {
    std::string tag;
    std::map<std::string, std::string> attribs;
};

class logger
{
public:
  // ---- XML output ---------------------------------------------------------
  bool enableXml;
  ofstream xmlFile;
  xmlWriter *xw;

  // ---- JSON / CSV output --------------------------------------------------
  bool enableJson;
  std::string jsonFileName;
  bool enableCsv;
  std::string csvFileName;

  // ---- Baseline compare ---------------------------------------------------
  bool compareEnabled;
  BaselineMap baseline;  // key -> value loaded from the compare file

  // ---- Internal state -----------------------------------------------------
  ResultStore results;                    // flat metric accumulator
  std::vector<ContextFrame> contextStack; // mirrors the live XML element stack

#ifdef ANDROID_LOGGER
  JNIEnv *jEnv;
  jobject *jObj;
  jmethodID printCallback;
#endif

  logger(bool _enableXml     = false, string _xmlFileName     = "",
         bool _enableJson     = false, string _jsonFileName    = "",
         bool _enableCsv      = false, string _csvFileName     = "",
         string _compareFileName = "");
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

private:
  // Extract context and push a ResultEntry; optionally print compare delta.
  // Call whenever a numeric metric result is available (contextStack size == 4).
  void recordMetric(const std::string &metric, float value);
};

#endif  // LOGGER_HPP
