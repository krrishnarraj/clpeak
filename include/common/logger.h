#ifndef LOGGER_HPP
#define LOGGER_HPP

/*
 * ANDROID_LOGGER -- defined only in case of an android ndk build.  On
 * Android there is no file output; the print() and metric-emission paths
 * dispatch directly to Kotlin via JNI callbacks.
 */

#include <iostream>
#include <string>
#include <vector>
#include <result_store.h>
#include "common.h"

#ifdef ANDROID_LOGGER
#include <jni.h>
#endif

class logger
{
public:
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

  // Construct with optional baseline-compare file path.
  // File output is centralized in the CLI entry point; per-backend loggers
  // handle stdout + baseline deltas only.
  explicit logger(std::string compareFileName = "");
  ~logger();

  // ---- stdout / Android UI ------------------------------------------------
  void print(std::string str);
  void print(double val);
  void print(float val);
  void print(int val);
  void print(unsigned int val);

  // ---- Result-scope recording API ----------------------------------------
  // Backends use these for the historical nested result shape:
  //   clpeak -> platform -> device -> test -> metric
  // The logger translates that shape into ResultEntry rows; result_store then
  // serializes those rows to every enabled dump format.
  void resultScopeBegin(std::string name);
  void resultScopeAttribute(std::string key, std::string value);
  void resultScopeAttribute(std::string key, unsigned int value);
  void resultSetContent(float value);
  void resultScopeEnd();
  void resultRecord(std::string metric, float value);

  // Record a skipped/unsupported/error metric.
  void recordSkip(const std::string &metric, ResultStatus status,
                  const std::string &reason);

  // ---- RAII scope guard --------------------------------------------------
  // Use resultScope(name) to get a guard that calls resultScopeEnd() on
  // destruction, eliminating scope leaks on early returns and exceptions.
  class ResultScope {
  public:
    ResultScope(logger *log, std::string name) : log(log) { log->resultScopeBegin(std::move(name)); }
    ~ResultScope() { if (log) log->resultScopeEnd(); }
    ResultScope(const ResultScope &) = delete;
    ResultScope &operator=(const ResultScope &) = delete;
    ResultScope(ResultScope &&other) noexcept : log(other.log) { other.log = nullptr; }
    ResultScope &operator=(ResultScope &&) = delete;
  private:
    logger *log;
  };

  ResultScope resultScope(std::string name) { return ResultScope(this, std::move(name)); }

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
