#include "logger_ios.h"

#include <common/common.h>
#include <common/result_store.h>
#include <sstream>

namespace
{
const char *iosStatusString(ResultStatus status)
{
  switch (status)
  {
  case ResultStatus::Ok:          return "ok";
  case ResultStatus::Unsupported: return "unsupported";
  case ResultStatus::Skipped:     return "skipped";
  case ResultStatus::Error:       return "error";
  }
  return "error";
}
}

LoggerIOS::LoggerIOS(ClpeakIOSCallbacks callbacks, void *context)
  : callbacks(callbacks), context(context)
{
}

void LoggerIOS::note(const std::string &msg)
{
  (void)msg;
}

void LoggerIOS::onDeviceBegin(const std::string &name,
                              const std::string &platform,
                              const std::string &driverVersion,
                              const std::vector<Prop> &props,
                              bool showPlatformLine,
                              int platformIndex,
                              int deviceIndex)
{
  (void)showPlatformLine;
  if (!callbacks.device) return;

  std::stringstream json;
  json << "[";
  for (size_t i = 0; i < props.size(); i++)
  {
    if (i > 0) json << ",";
    json << "{\"k\":\"" << jsonEscape(props[i].key)
         << "\",\"v\":\"" << jsonEscape(props[i].value) << "\"}";
  }
  json << "]";

  callbacks.device(context,
                   curBackend.c_str(),
                   platform.c_str(),
                   name.c_str(),
                   driverVersion.c_str(),
                   json.str().c_str(),
                   platformIndex,
                   deviceIndex);
}

void LoggerIOS::onTestBegin(const std::string &tag,
                            const std::string &display,
                            const std::string &unit)
{
  (void)tag;
  (void)unit;
  curDisplay = display;
}

void LoggerIOS::onMetricEmitted(const ResultEntry &e, float value, bool subMetric)
{
  (void)subMetric;
  if (!callbacks.metric) return;

  callbacks.metric(context,
                   e.backend.c_str(),
                   e.platform.c_str(),
                   e.device.c_str(),
                   e.driver.c_str(),
                   e.category.c_str(),
                   e.test.c_str(),
                   curDisplay.c_str(),
                   e.metric.c_str(),
                   e.unit.c_str(),
                   value,
                   "ok",
                   "");
}

void LoggerIOS::onMetricSkipped(const ResultEntry &e)
{
  if (!callbacks.metric) return;

  callbacks.metric(context,
                   e.backend.c_str(),
                   e.platform.c_str(),
                   e.device.c_str(),
                   e.driver.c_str(),
                   e.category.c_str(),
                   e.test.c_str(),
                   curDisplay.c_str(),
                   e.metric.c_str(),
                   e.unit.c_str(),
                   0.0f,
                   iosStatusString(e.status),
                   e.reason.c_str());
}
