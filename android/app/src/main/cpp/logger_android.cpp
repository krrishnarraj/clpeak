#include <logger.h>
#include <iomanip>
#include <sstream>

// Android logger: print() methods dispatch to the Java UI via JNI.
// XML / JSON / CSV file output is not used on Android, but the context stack
// (xmlOpenTag / xmlAppendAttribs / xmlCloseTag) is maintained in memory so
// that recordMetric() can extract fully-qualified result entries and dispatch
// them to the Kotlin layer via a second JNI callback.

logger::logger(bool _enableXml,     std::string _xmlFileName,
               bool _enableJson,    std::string _jsonFileName,
               bool _enableCsv,     std::string _csvFileName,
               std::string _compareFileName)
  : enableXml(false),
    xw(nullptr),
    enableJson(false),
    enableCsv(false),
    compareEnabled(false)
{
  (void)_enableXml; (void)_xmlFileName;
  (void)_enableJson; (void)_jsonFileName;
  (void)_enableCsv;  (void)_csvFileName;
  (void)_compareFileName;
}

logger::~logger()
{
}

// ---- stdout -> JNI callbacks ----------------------------------------------

void logger::print(std::string str)
{
  jEnv->CallVoidMethod((*jObj), printCallback, jEnv->NewStringUTF(str.c_str()));
}

void logger::print(double val)
{
  std::stringstream ss;
  ss << std::setprecision(2) << std::fixed << val;
  jEnv->CallVoidMethod((*jObj), printCallback, jEnv->NewStringUTF(ss.str().c_str()));
}

void logger::print(float val)
{
  std::stringstream ss;
  ss << std::setprecision(2) << std::fixed << val;
  jEnv->CallVoidMethod((*jObj), printCallback, jEnv->NewStringUTF(ss.str().c_str()));
}

void logger::print(int val)
{
  std::stringstream ss;
  ss << val;
  jEnv->CallVoidMethod((*jObj), printCallback, jEnv->NewStringUTF(ss.str().c_str()));
}

void logger::print(unsigned int val)
{
  std::stringstream ss;
  ss << val;
  jEnv->CallVoidMethod((*jObj), printCallback, jEnv->NewStringUTF(ss.str().c_str()));
}

// ---- Context stack -- maintained in memory for recordMetric() -------------

void logger::xmlOpenTag(std::string tag)
{
  contextStack.push_back({tag, {}});
}

void logger::xmlAppendAttribs(std::string key, std::string value)
{
  if (!contextStack.empty())
    contextStack.back().attribs[key] = value;
}

void logger::xmlAppendAttribs(std::string key, unsigned int value)
{
  if (!contextStack.empty())
    contextStack.back().attribs[key] = std::to_string(value);
}

void logger::xmlSetContent(std::string value)
{
  (void)value;
}

void logger::xmlSetContent(float value)
{
  // kernel_latency uses xmlSetContent instead of xmlRecord
  recordMetric("latency", value);
}

void logger::xmlCloseTag()
{
  if (!contextStack.empty())
    contextStack.pop_back();
}

void logger::xmlRecord(std::string tag, std::string value)
{
  (void)tag; (void)value;
}

void logger::xmlRecord(std::string tag, float value)
{
  // Called at context depth 4 (clpeak > platform > device > test_group)
  recordMetric(tag, value);
}

// ---- Structured metric callback -> Kotlin ---------------------------------

void logger::recordMetric(const std::string &metric, float value)
{
  if (contextStack.size() < 4 || !recordMetricCallback)
    return;

  auto getAttrib = [&](int idx, const char *key) -> std::string {
    auto it = contextStack[idx].attribs.find(key);
    return (it != contextStack[idx].attribs.end()) ? it->second : "";
  };

  // Outer frames (clpeak > platform > device) are at fixed positions; the
  // test frame is always the innermost open tag.  Using back() instead of
  // [3] keeps metrics attributed correctly even if a prior test leaked its
  // xmlCloseTag -- otherwise all subsequent siblings would nest under the
  // leaked parent and collapse into a single Android result card.
  const size_t testIdx = contextStack.size() - 1;
  const std::string backend  = getAttrib(1, "backend");
  const std::string platform = getAttrib(1, "name");
  const std::string device   = getAttrib(2, "name");
  const std::string driver   = getAttrib(2, "driver_version");
  const std::string test     = contextStack[testIdx].tag;
  const std::string unit     = getAttrib(testIdx, "unit");

  jEnv->CallVoidMethod(
      (*jObj),
      recordMetricCallback,
      jEnv->NewStringUTF(backend.c_str()),
      jEnv->NewStringUTF(platform.c_str()),
      jEnv->NewStringUTF(device.c_str()),
      jEnv->NewStringUTF(driver.c_str()),
      jEnv->NewStringUTF(test.c_str()),
      jEnv->NewStringUTF(metric.c_str()),
      jEnv->NewStringUTF(unit.c_str()),
      static_cast<jfloat>(value));
}
