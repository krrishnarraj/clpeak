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

logger::logger(std::string compareFileName)
    : compareEnabled(!compareFileName.empty()),
      curCategory(Category::Unknown),
      shimDepth(0),
      inTestScope(false)
{
    if (compareEnabled)
        baseline = buildBaselineMap(loadResultFile(compareFileName));
}

logger::~logger()
{
}

// ---- print -> JNI ---------------------------------------------------------

void logger::print(std::string str)
{
  jstring jstr = jEnv->NewStringUTF(str.c_str());
  jEnv->CallVoidMethod((*jObj), printCallback, jstr);
  jEnv->DeleteLocalRef(jstr);
}

void logger::print(double val)
{
  std::stringstream ss;
  ss << std::setprecision(2) << std::fixed << val;
  jstring jstr = jEnv->NewStringUTF(ss.str().c_str());
  jEnv->CallVoidMethod((*jObj), printCallback, jstr);
  jEnv->DeleteLocalRef(jstr);
}

void logger::print(float val)
{
  std::stringstream ss;
  ss << std::setprecision(2) << std::fixed << val;
  jstring jstr = jEnv->NewStringUTF(ss.str().c_str());
  jEnv->CallVoidMethod((*jObj), printCallback, jstr);
  jEnv->DeleteLocalRef(jstr);
}

void logger::print(int val)
{
  std::stringstream ss;
  ss << val;
  jstring jstr = jEnv->NewStringUTF(ss.str().c_str());
  jEnv->CallVoidMethod((*jObj), printCallback, jstr);
  jEnv->DeleteLocalRef(jstr);
}

void logger::print(unsigned int val)
{
  std::stringstream ss;
  ss << val;
  jstring jstr = jEnv->NewStringUTF(ss.str().c_str());
  jEnv->CallVoidMethod((*jObj), printCallback, jstr);
  jEnv->DeleteLocalRef(jstr);
}

void logger::recordSkip(const std::string &metric, ResultStatus status,
                        const std::string &reason)
{
  emit(metric, status, 0.0f, reason);
}

// ---- Result-scope recording ----------------------------------------------

void logger::resultScopeBegin(std::string name)
{
  shimDepth++;

  if (shimDepth == 4)
  {
    inTestScope = true;
    curTest = name;
    curUnit.clear();
    if (curCategory == Category::Unknown ||
        curCategory == Category::Latency ||
        curCategory == Category::Bandwidth ||
        curCategory == Category::FpCompute ||
        curCategory == Category::IntCompute)
    {
      curCategory = Category::Unknown;
    }
  }
}

void logger::resultScopeAttribute(std::string key, std::string value)
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

void logger::resultScopeAttribute(std::string key, unsigned int value)
{
  std::stringstream ss;
  ss << value;
  resultScopeAttribute(key, ss.str());
}

void logger::resultSetContent(float value)
{
  if (shimDepth == 4 && !curTest.empty())
    emit(curTest, ResultStatus::Ok, value, "");
}

void logger::resultScopeEnd()
{
  if (shimDepth == 4)
  {
    curTest.clear();
    curUnit.clear();
    curCategory = Category::Unknown;
    inTestScope = false;
  }
  if (shimDepth > 0)
    shimDepth--;
}

void logger::resultRecord(std::string metric, float value)
{
  if (shimDepth == 4)
    emit(metric, ResultStatus::Ok, value, "");
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

  jstring jBackend  = jEnv->NewStringUTF(e.backend.c_str());
  jstring jPlatform = jEnv->NewStringUTF(e.platform.c_str());
  jstring jDevice   = jEnv->NewStringUTF(e.device.c_str());
  jstring jDriver   = jEnv->NewStringUTF(e.driver.c_str());
  jstring jCategory = jEnv->NewStringUTF(e.category.c_str());
  jstring jTest     = jEnv->NewStringUTF(e.test.c_str());
  jstring jMetric   = jEnv->NewStringUTF(e.metric.c_str());
  jstring jUnit     = jEnv->NewStringUTF(e.unit.c_str());

  jEnv->CallVoidMethod(
      (*jObj),
      recordMetricCallback,
      jBackend,
      jPlatform,
      jDevice,
      jDriver,
      jCategory,
      jTest,
      jMetric,
      jUnit,
      static_cast<jfloat>(value));

  jEnv->DeleteLocalRef(jBackend);
  jEnv->DeleteLocalRef(jPlatform);
  jEnv->DeleteLocalRef(jDevice);
  jEnv->DeleteLocalRef(jDriver);
  jEnv->DeleteLocalRef(jCategory);
  jEnv->DeleteLocalRef(jTest);
  jEnv->DeleteLocalRef(jMetric);
  jEnv->DeleteLocalRef(jUnit);
}
