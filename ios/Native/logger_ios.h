#ifndef LOGGER_IOS_H
#define LOGGER_IOS_H

#include "clpeak_ios_bridge.h"
#include <common/logger.h>
#include <string>

class LoggerIOS : public logger
{
public:
  LoggerIOS(ClpeakIOSCallbacks callbacks, void *context);

  void note(const std::string &msg) override;

protected:
  void onBackendBegin(const std::string &name) override { (void)name; }
  void onDeviceBegin(const std::string &name,
                     const std::string &platform,
                     const std::string &driverVersion,
                     const std::vector<Prop> &props,
                     bool showPlatformLine,
                     int platformIndex,
                     int deviceIndex) override;
  void onTestBegin(const std::string &tag,
                   const std::string &display,
                   const std::string &unit) override;
  void onMetricEmitted(const ResultEntry &e,
                       float value,
                       bool subMetric) override;
  void onMetricSkipped(const ResultEntry &e) override;
  void onTestSkippedAll(ResultStatus status,
                        const std::string &reason) override { (void)status; (void)reason; }
  void onTestEnd() override {}
  void onDeviceEnd() override {}
  void onBackendEnd() override {}

private:
  ClpeakIOSCallbacks callbacks;
  void *context;
  std::string curDisplay;
};

#endif // LOGGER_IOS_H
