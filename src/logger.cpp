#include <logger.h>
#include <iomanip>
#include <sstream>

// All output formats (JSON, CSV, XML) are produced from the in-memory
// `results` store at program exit -- this guarantees identical ordering and
// row counts across the three files and lets every entry be tagged with the
// new backend / category / status fields uniformly.  Live stdout still
// streams during execution via print().

logger::logger(bool _enableXml,     std::string _xmlFileName,
               bool _enableJson,    std::string _jsonFileName,
               bool _enableCsv,     std::string _csvFileName,
               std::string _compareFileName)
  : enableXml(_enableXml),
    xmlFileName(_xmlFileName),
    enableJson(_enableJson),
    jsonFileName(_jsonFileName),
    enableCsv(_enableCsv),
    csvFileName(_csvFileName),
    compareEnabled(!_compareFileName.empty()),
    curCategory(Category::Unknown),
    shimDepth(0),
    inTestScope(false)
{
  if (compareEnabled)
  {
    ResultStore base = loadResultFile(_compareFileName);
    baseline = buildBaselineMap(base);
    if (!baseline.empty())
    {
      std::cout << "clpeak: comparing against baseline: " << _compareFileName
                << " (" << baseline.size() << " entries)" << "\n";
      std::cout.flush();
    }
  }
}

logger::~logger()
{
  if (enableJson) saveJson(results, jsonFileName);
  if (enableCsv)  saveCsv (results, csvFileName);
  if (enableXml)  saveXml (results, xmlFileName);
}

// ---- stdout output --------------------------------------------------------

void logger::print(std::string str)
{
  std::cout << str;
  std::cout.flush();
}

void logger::print(double val)
{
  std::cout << std::setprecision(2) << std::fixed << val;
  std::cout.flush();
}

void logger::print(float val)
{
  std::cout << std::setprecision(2) << std::fixed << val;
  std::cout.flush();
}

void logger::print(int val)
{
  std::cout << val;
  std::cout.flush();
}

void logger::print(unsigned int val)
{
  std::cout << val;
  std::cout.flush();
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
  // Derive category from unit if the caller hasn't already set one via
  // categoryBegin.  Stage 3 backends will always call categoryBegin, but
  // the legacy shim path relies on this fallback.
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

// ---- Result-scope recording ----------------------------------------------
// Translates resultScopeBegin / resultScopeAttribute / resultRecord /
// resultScopeEnd into updates of the current run / category / test scope so
// emit() can produce fully-qualified ResultEntry rows without backend-side
// changes.
//
// Implicit scope depth:
//   1 = <clpeak>
//   2 = <platform>
//   3 = <device>
//   4 = <test_group>
// Attribute side-effects fire on whichever frame is currently topmost.

void logger::resultScopeBegin(std::string name)
{
  shimDepth++;

  // Frame 4 = a test group -- remember the scope name as the current test.
  // The unit attribute (if any) follows in subsequent resultScopeAttribute
  // calls and refines curUnit / curCategory.
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
      // Defer category derivation until the unit attribute arrives so
      // tests with the same name across categories (e.g. wmma fp vs int)
      // are categorised by their declared unit, not by a sticky leftover.
      curCategory = Category::Unknown;
    }
  }
}

void logger::resultScopeAttribute(std::string key, std::string value)
{
  switch (shimDepth)
  {
  case 2:  // <platform>
    if      (key == "name")    curPlatform = value;
    else if (key == "backend") curBackend  = value;
    break;
  case 3:  // <device>
    if      (key == "name")           curDevice = value;
    else if (key == "driver_version") curDriver = value;
    break;
  case 4:  // <test_group>
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

void logger::resultSetContent(std::string)
{
  // Dead path historically: kept as a no-op for ABI compatibility.
}

void logger::resultSetContent(float value)
{
  // Historical: a single-value test called resultSetContent with the
  // measurement.  Translate into a record() using the test name as the metric
  // so the row still carries a non-empty metric field.
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
  if (inTestScope || shimDepth > 3)
    shimDepth--;
}

void logger::resultRecord(std::string, std::string)
{
  // String-valued metrics are not represented in the result store.
}

void logger::resultRecord(std::string metric, float value)
{
  if (shimDepth == 4)
    emit(metric, ResultStatus::Ok, value, "");
}

// ---- Common emission ------------------------------------------------------

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

  if (!compareEnabled || status != ResultStatus::Ok)
    return;

  auto it = baseline.find(e.key());
  if (it == baseline.end())
    return;

  float base  = it->second;
  float delta = (base != 0.0f) ? 100.0f * (value - base) / base : 0.0f;

  char  sign     = (delta >= 0.0f) ? '+' : '-';
  float absDelta = (delta < 0.0f)  ? -delta : delta;

  std::cout << "        "
            << "(was " << std::fixed << std::setprecision(2) << base
            << ",  " << sign << std::setprecision(1) << absDelta << "%)"
            << "\n";
  std::cout.flush();
}
