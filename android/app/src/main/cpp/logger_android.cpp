#include <logger.h>
#include <iomanip>
#include <sstream>

// Android logger: print() routes to the Kotlin UI via JNI; metric emission
// fires record_metric_callback_from_c() on every Ok measurement.  File
// outputs (XML / JSON / CSV) are unused on Android, but the result store
// is still populated for parity with the desktop build.
//
// Stage 1 of the v2 schema refactor keeps the JNI callback signature as it
// was (backend, platform, device, driver, test, metric, unit, value).
// Stage 4 extends it with category / status / reason and updates Kotlin to
// match.

logger::logger(bool, std::string,
               bool, std::string,
               bool, std::string,
               std::string)
  : enableXml(false),
    enableJson(false),
    enableCsv(false),
    compareEnabled(false),
    curCategory(Category::Unknown),
    shimDepth(0)
{
}

logger::~logger()
{
}

// ---- print -> JNI ---------------------------------------------------------

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

// ---- High-level recording API --------------------------------------------

void logger::deviceBegin(const std::string &backend,
                         const std::string &platform,
                         const std::string &device,
                         const std::string &driver)
{
  curBackend  = backend;
  curPlatform = platform;
  curDevice   = device;
  curDriver   = driver;
}

void logger::deviceEnd()
{
  curBackend.clear();
  curPlatform.clear();
  curDevice.clear();
  curDriver.clear();
  curCategory = Category::Unknown;
  curTest.clear();
  curUnit.clear();
}

void logger::categoryBegin(Category c)
{
  curCategory = c;
}

void logger::categoryEnd()
{
  curCategory = Category::Unknown;
  curTest.clear();
  curUnit.clear();
}

void logger::testBegin(const std::string &test, const std::string &unit)
{
  curTest = test;
  curUnit = unit;
  if (curCategory == Category::Unknown)
    curCategory = categoryFromUnit(unit);
}

void logger::testEnd()
{
  curTest.clear();
  curUnit.clear();
}

void logger::record(const std::string &metric, float value)
{
  emit(metric, ResultStatus::Ok, value, "");
}

void logger::recordSkip(const std::string &metric, ResultStatus status,
                        const std::string &reason)
{
  emit(metric, status, 0.0f, reason);
}

// ---- Legacy XML shim ------------------------------------------------------

void logger::xmlOpenTag(std::string tag)
{
  shimDepth++;
  if (shimDepth == 4)
  {
    curTest = tag;
    curUnit.clear();
    curCategory = Category::Unknown;
  }
}

void logger::xmlAppendAttribs(std::string key, std::string value)
{
  switch (shimDepth)
  {
  case 2:
    if      (key == "name")    curPlatform = value;
    else if (key == "backend") curBackend  = value;
    break;
  case 3:
    if      (key == "name")           curDevice = value;
    else if (key == "driver_version") curDriver = value;
    break;
  case 4:
    if (key == "unit")
    {
      curUnit     = value;
      curCategory = categoryFromUnit(value);
    }
    break;
  default:
    break;
  }
}

void logger::xmlAppendAttribs(std::string key, unsigned int value)
{
  std::stringstream ss;
  ss << value;
  xmlAppendAttribs(key, ss.str());
}

void logger::xmlSetContent(std::string)
{
}

void logger::xmlSetContent(float value)
{
  if (shimDepth == 4 && !curTest.empty())
    emit(curTest, ResultStatus::Ok, value, "");
}

void logger::xmlCloseTag()
{
  if (shimDepth == 4)
  {
    curTest.clear();
    curUnit.clear();
    curCategory = Category::Unknown;
  }
  if (shimDepth > 0)
    shimDepth--;
}

void logger::xmlRecord(std::string, std::string)
{
}

void logger::xmlRecord(std::string tag, float value)
{
  if (shimDepth == 4)
    emit(tag, ResultStatus::Ok, value, "");
}

// ---- emit -> JNI ----------------------------------------------------------

void logger::emit(const std::string &metric, ResultStatus status,
                  float value, const std::string &reason)
{
  ResultEntry e;
  e.backend  = curBackend;
  e.platform = curPlatform;
  e.device   = curDevice;
  e.driver   = curDriver;
  e.category = categoryString(curCategory);
  e.test     = curTest;
  e.metric   = metric;
  e.unit     = curUnit;
  e.status   = status;
  e.value    = (status == ResultStatus::Ok) ? value : 0.0f;
  e.reason   = reason;
  results.push_back(e);

  // v2 JNI signature carries `category` after `backend`.  Skip /
  // unsupported / error rows are still recorded in `results` but don't
  // fire the callback (Kotlin renders only Ok values).  A follow-up
  // change can extend the signature to include status+reason and
  // surface skipped rows in the UI.
  if (status != ResultStatus::Ok || !recordMetricCallback)
    return;

  jEnv->CallVoidMethod(
      (*jObj),
      recordMetricCallback,
      jEnv->NewStringUTF(e.backend.c_str()),
      jEnv->NewStringUTF(e.platform.c_str()),
      jEnv->NewStringUTF(e.device.c_str()),
      jEnv->NewStringUTF(e.driver.c_str()),
      jEnv->NewStringUTF(e.category.c_str()),
      jEnv->NewStringUTF(e.test.c_str()),
      jEnv->NewStringUTF(e.metric.c_str()),
      jEnv->NewStringUTF(e.unit.c_str()),
      static_cast<jfloat>(value));
}
