#include "logger_ffi.h"

#include <common/common.h>
#include <common/run_document.h>

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
// side dropped, leaving the live results view empty while the file dump
// (which never went through the C locale) stayed correct.
//
// Seven significant digits, matching the file writer, so a reading looks the
// same live as it does after a reload.
std::string fmtNum(double v)
{
    std::ostringstream ss;
    ss.imbue(std::locale::classic());
    ss << std::setprecision(7) << v;
    return ss.str();
}

// The resolved test header, identical on test_begin and test_skipped: the GUI
// builds its test node from either, and both must describe the same thing.
void appendTestHeader(std::ostringstream &ss, const LogEvent &e)
{
    appendStr(ss, "test",    e.testId);
    appendStr(ss, "title",   e.testTitle);
    appendStr(ss, "variant", e.testVariant);
    appendStr(ss, "axis",    e.testAxis);
    appendStr(ss, "category", categoryString(e.category));
    appendStr(ss, "shape",     shapeString(e.shape));
    appendStr(ss, "direction", directionString(e.direction));
    appendStr(ss, "quantity",  quantityString(e.quantity));
    appendStr(ss, "unit",      e.unit);
    appendStr(ss, "desc", e.testDescription);
}

} // namespace

std::string ffiEventToJson(const LogEvent &e)
{
    std::ostringstream ss;
    ss.imbue(std::locale::classic());  // see fmtNum -- the indices too
    ss << "{\"t\":\"" << kindTag(e.kind) << "\"";

    // Scope context — present on every scoped event.
    appendStr(ss, "backend",  e.backend);
    appendStr(ss, "platform", e.platform);
    appendStr(ss, "device",   e.device);
    appendStr(ss, "driver",   e.driver);
    // Scope, not a DeviceBegin detail: a device name is not unique (MoltenVK
    // exposes one GPU twice), so every event has to say which one it is on.
    ss << ",\"device_index\":" << e.deviceIndex;

    switch (e.kind)
    {
    case LogEvent::Kind::DeviceBegin:
        appendStr(ss, "type", deviceTypeString(e.type));
        ss << ",\"platform_index\":" << e.platformIndex
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
        appendTestHeader(ss, e);
        // A reopened test already has a node on the consumer's side; saying so
        // keeps it from being rebuilt and losing the readings already in it.
        ss << ",\"reopened\":" << (e.reopened ? "true" : "false");
        break;

    case LogEvent::Kind::Metric:
        // Enough to find the open test's node, and no more: its header
        // arrived on test_begin and does not change per reading.
        appendStr(ss, "test",    e.testId);
        appendStr(ss, "variant", e.testVariant);
        appendStr(ss, "metric",  e.metric.id);
        appendStr(ss, "label",   e.metric.label);
        if (e.metric.status == ResultStatus::Ok)
        {
            ss << ",\"value\":" << fmtNum(e.metric.value);
        }
        else
        {
            appendStr(ss, "status", statusString(e.metric.status));
            appendStr(ss, "reason", e.metric.reason);
        }
        // Unit fields ride along only when this reading overrides its test's,
        // which is what lets one test carry both FLOPS and OPS readings.
        if (e.metric.hasUnit)
        {
            appendStr(ss, "unit",     e.metric.unit);
            appendStr(ss, "quantity", quantityString(e.metric.quantity));
        }
        if (e.metric.direction != Direction::FromUnit)
            appendStr(ss, "direction", directionString(e.metric.direction));
        appendStr(ss, "minfo", e.metric.description);
        break;

    case LogEvent::Kind::TestSkippedAll:
        appendTestHeader(ss, e);
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
