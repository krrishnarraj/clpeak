#include "logger_ffi.h"

#include <common/common.h>
#include <common/result_store.h>

#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <locale>
#include <sstream>

namespace
{

const char *kindTag(LogEvent::Kind k)
{
    switch (k)
    {
    case LogEvent::Kind::BackendBegin:   return "backend_begin";
    case LogEvent::Kind::DeviceBegin:    return "device";
    case LogEvent::Kind::TestBegin:      return "test_begin";
    case LogEvent::Kind::Metric:         return "metric";
    case LogEvent::Kind::TestSkippedAll: return "test_skipped";
    case LogEvent::Kind::TestEnd:        return "test_end";
    case LogEvent::Kind::DeviceEnd:      return "device_end";
    case LogEvent::Kind::BackendEnd:     return "backend_end";
    case LogEvent::Kind::Note:           return "note";
    }
    return "unknown";
}

void appendStr(std::ostringstream &ss, const char *key, const std::string &value)
{
    ss << ",\"" << key << "\":\"" << jsonEscape(value) << "\"";
}

// JSON numbers are '.'-separated by definition, so this must not follow the
// process locale.  It matters here more than anywhere else in clpeak: the GUI
// hosts this library inside a toolkit that sets the locale for us (GTK's
// gtk_init calls setlocale(LC_ALL, "")), so on a comma-decimal desktop a
// printf("%.4f") emitted `"value":1234,5678` -- malformed JSON that the Dart
// side dropped, leaving the live results view empty while the XML dump (which
// never went through the C locale) stayed correct.
std::string fmtFloat(float v)
{
    std::ostringstream ss;
    ss.imbue(std::locale::classic());
    ss << std::fixed << std::setprecision(4) << v;
    return ss.str();
}

} // namespace

std::string ffiEventToJson(const LogEvent &e)
{
    std::ostringstream ss;
    ss.imbue(std::locale::classic());  // see fmtFloat -- the indices too
    ss << "{\"t\":\"" << kindTag(e.kind) << "\"";

    // Scope context — present on every scoped event.
    appendStr(ss, "backend",  e.backend);
    appendStr(ss, "platform", e.platform);
    appendStr(ss, "device",   e.device);
    appendStr(ss, "driver",   e.driver);

    switch (e.kind)
    {
    case LogEvent::Kind::DeviceBegin:
        ss << ",\"platform_index\":" << e.platformIndex
           << ",\"device_index\":"   << e.deviceIndex
           << ",\"props\":[";
        for (size_t i = 0; i < e.props.size(); i++)
        {
            if (i) ss << ",";
            ss << "{\"k\":\"" << jsonEscape(e.props[i].key)
               << "\",\"v\":\"" << jsonEscape(e.props[i].value) << "\"}";
        }
        ss << "]";
        break;

    case LogEvent::Kind::TestBegin:
        appendStr(ss, "test",     e.testTag);
        appendStr(ss, "display",  e.testDisplay);
        appendStr(ss, "unit",     e.unit);
        appendStr(ss, "category", categoryString(e.category));
        appendStr(ss, "desc",     e.testDescription);
        break;

    case LogEvent::Kind::Metric:
        appendStr(ss, "category", e.entry.category);
        appendStr(ss, "test",     e.entry.test);
        appendStr(ss, "display",  e.testDisplay);
        appendStr(ss, "metric",   e.entry.metric);
        appendStr(ss, "unit",     e.entry.unit);
        ss << ",\"value\":" << fmtFloat(e.entry.value);
        appendStr(ss, "status", statusString(e.entry.status));
        appendStr(ss, "reason", e.entry.reason);
        ss << ",\"sub\":" << (e.subMetric ? "true" : "false");
        appendStr(ss, "desc",  e.entry.description);
        appendStr(ss, "minfo", e.entry.metricDescription);
        break;

    case LogEvent::Kind::TestSkippedAll:
        appendStr(ss, "test",     e.testTag);
        appendStr(ss, "display",  e.testDisplay);
        appendStr(ss, "unit",     e.unit);
        appendStr(ss, "category", categoryString(e.category));
        appendStr(ss, "desc",     e.testDescription);
        ss << ",\"metrics\":[";
        for (size_t i = 0; i < e.metricNames.size(); i++)
        {
            if (i) ss << ",";
            ss << "\"" << jsonEscape(e.metricNames[i]) << "\"";
        }
        ss << "]";
        appendStr(ss, "status", statusString(e.status));
        appendStr(ss, "reason", e.reason);
        break;

    case LogEvent::Kind::Note:
        appendStr(ss, "message", e.message);
        break;

    default:
        break;  // begin/end markers carry only the context
    }

    ss << "}";
    return ss.str();
}

void ffiEmitJson(ClpeakEventCallback cb, void *userData, const std::string &json)
{
    if (!cb)
        return;
    char *out = static_cast<char *>(std::malloc(json.size() + 1));
    if (!out)
        return;
    std::memcpy(out, json.c_str(), json.size() + 1);
    cb(userData, out);  // ownership transfers to the callee
}

void LoggerFfi::onEvent(const LogEvent &e)
{
    ffiEmitJson(onEventCb, userData, ffiEventToJson(e));
}
