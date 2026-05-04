#ifndef LOGGER_HPP
#define LOGGER_HPP

/*
 * ANDROID_LOGGER -- defined only in case of an android ndk build.  On
 * Android there is no file output; the print() and metric-emission paths
 * dispatch directly to Kotlin via JNI callbacks.
 */

#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <result_store.h>
#include "common.h"

#ifdef ANDROID_LOGGER
#include <jni.h>
#endif

class logger
{
public:
  // ---- Output sinks (file paths) -----------------------------------------
  bool        enableXml;
  std::string xmlFileName;
  bool        enableJson;
  std::string jsonFileName;
  bool        enableCsv;
  std::string csvFileName;

  // ---- Baseline compare ---------------------------------------------------
  bool        compareEnabled;
  BaselineMap baseline;

  // ---- Accumulated metrics ------------------------------------------------
  // Single source of truth: all formats serialize from this store at exit.
  ResultStore results;

#ifdef ANDROID_LOGGER
  JNIEnv  *jEnv;
  jobject *jObj;
  jmethodID printCallback;
  jmethodID recordMetricCallback;
#endif

  logger(bool _enableXml      = false, std::string _xmlFileName     = "",
         bool _enableJson     = false, std::string _jsonFileName    = "",
         bool _enableCsv      = false, std::string _csvFileName     = "",
         std::string _compareFileName = "");
  ~logger();

  // ---- stdout / Android UI ------------------------------------------------
  void print(std::string str);
  void print(double val);
  void print(float val);
  void print(int val);
  void print(unsigned int val);

  // ---- High-level recording API ------------------------------------------
  // Backends call these in nested scopes:
  //   deviceBegin -> categoryBegin -> testBegin -> record* -> testEnd ->
  //                  categoryEnd  -> deviceEnd
  // Each backend's runAll iterates the four categories in fixed order
  // (FpCompute, IntCompute, Bandwidth, Latency).  Within a test, every
  // measured variant gets one record() or recordSkip() call.
  void deviceBegin(const std::string &backend,
                   const std::string &platform,
                   const std::string &device,
                   const std::string &driver);
  void deviceEnd();

  void categoryBegin(Category c);
  void categoryEnd();

  void testBegin(const std::string &test, const std::string &unit);
  void testEnd();

  void record    (const std::string &metric, float value);
  void recordSkip(const std::string &metric, ResultStatus status,
                  const std::string &reason);

  // ---- Result-scope recording API ----------------------------------------
  // Backends use these for the historical nested result shape:
  //   clpeak -> platform -> device -> test -> metric
  // The logger translates that shape into ResultEntry rows; result_store then
  // serializes those rows to every enabled dump format.
  void resultScopeBegin(std::string name);
  void resultScopeAttribute(std::string key, std::string value);
  void resultScopeAttribute(std::string key, unsigned int value);
  void resultSetContent(std::string value);
  void resultSetContent(float value);
  void resultScopeEnd();
  void resultRecord(std::string metric, std::string value);
  void resultRecord(std::string metric, float value);

private:
  // Current run / category / test scope.  Updated by both APIs and read by
  // emit() to qualify each ResultEntry.
  std::string curBackend, curPlatform, curDevice, curDriver;
  Category    curCategory;
  std::string curTest, curUnit;

  // Result-scope cursor: depth in the implicit
  //   clpeak (1) > platform (2) > device (3) > test (4)
  // stack as driven by resultScopeBegin / resultScopeEnd.
  int shimDepth;

  // Tracks whether we are inside a depth-4 (test-group) scope.
  // resultScopeEnd() becomes a no-op when called before the matching
  // resultScopeBegin() was reached, preventing shimDepth corruption.
  bool inTestScope;

  // Build a ResultEntry from the current scope plus the supplied metric
  // and append to `results`.  Also prints a baseline-delta line to stdout
  // when compare mode is on and the metric matches a baseline key.
  void emit(const std::string &metric, ResultStatus status,
            float value, const std::string &reason);
};

#endif  // LOGGER_HPP
