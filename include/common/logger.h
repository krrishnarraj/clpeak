#ifndef LOGGER_HPP
#define LOGGER_HPP

#include <iostream>
#include <string>
#include <vector>
#include <common/result_store.h>
#include "common.h"

class logger
{
public:
  // ---- Baseline compare ---------------------------------------------------
  bool        compareEnabled;
  BaselineMap baseline;

  // ---- Accumulated metrics ------------------------------------------------
  ResultStore results;

  explicit logger(std::string compareFileName = "");
  virtual ~logger() = default;

  // ---- Client-specific output — derived classes implement -----------------
  virtual void print(std::string str) = 0;
  virtual void print(double val) = 0;
  virtual void print(float val) = 0;
  virtual void print(int val) = 0;
  virtual void print(unsigned int val) = 0;

  // ---- Result-scope recording API — common implementation -----------------
  void resultScopeBegin(std::string name);
  void resultScopeAttribute(std::string key, std::string value);
  void resultScopeAttribute(std::string key, unsigned int value);
  void resultSetContent(float value);
  void resultScopeEnd();
  void resultRecord(std::string metric, float value);
  void recordSkip(const std::string &metric, ResultStatus status,
                  const std::string &reason);

  // ---- RAII scope guard --------------------------------------------------
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

protected:
  // Override to get notified on every emitted metric (baseline deltas, JNI, etc.).
  virtual void onMetricEmitted(const ResultEntry &e, float value);

  // Current scope context — populated by resultScope* methods, read by emit().
  std::string curBackend, curPlatform, curDevice, curDriver;
  Category    curCategory = Category::Unknown;
  std::string curTest, curUnit;
  int  shimDepth    = 0;
  bool inTestScope  = false;

private:
  void emit(const std::string &metric, ResultStatus status,
            float value, const std::string &reason);
};

#endif  // LOGGER_HPP
