#include <logger.h>
#include <iomanip>
#include <sstream>

logger::logger(bool _enableXml,     std::string _xmlFileName,
               bool _enableJson,    std::string _jsonFileName,
               bool _enableCsv,     std::string _csvFileName,
               std::string _compareFileName)
  : enableXml(_enableXml),
    xw(nullptr),
    enableJson(_enableJson),
    jsonFileName(_jsonFileName),
    enableCsv(_enableCsv),
    csvFileName(_csvFileName),
    compareEnabled(!_compareFileName.empty())
{
  if (enableXml)
  {
    xmlFile.open(_xmlFileName);
    xw = new xmlWriter(xmlFile);
    xmlFile.flush();
  }

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
  if (enableXml)
  {
    xw->closeAll();
    delete xw;
    xmlFile.close();
  }

  if (enableJson)
    saveJson(results, jsonFileName);

  if (enableCsv)
    saveCsv(results, csvFileName);
}

// ---- stdout output --------------------------------------------------------

void logger::print(std::string str)
{
  std::cout << str;
  std::cout.flush();
}

void logger::print(double val)
{
  std::cout << std::setprecision(2) << std::fixed;
  std::cout << val;
  std::cout.flush();
}

void logger::print(float val)
{
  std::cout << std::setprecision(2) << std::fixed;
  std::cout << val;
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

// ---- XML / context-tracking -----------------------------------------------

void logger::xmlOpenTag(std::string tag)
{
  contextStack.push_back({tag, {}});

  if (enableXml)
  {
    xw->openElt(tag.c_str());
    xmlFile.flush();
  }
}

void logger::xmlAppendAttribs(std::string key, std::string value)
{
  if (!contextStack.empty())
    contextStack.back().attribs[key] = value;

  if (enableXml)
  {
    xw->attr(key.c_str(), value.c_str());
    xmlFile.flush();
  }
}

void logger::xmlAppendAttribs(std::string key, unsigned int value)
{
  std::stringstream ss;
  ss << value;
  // Delegate to string overload so the context frame is also updated
  xmlAppendAttribs(key, ss.str());
}

void logger::xmlSetContent(std::string value)
{
  if (enableXml)
  {
    xw->content(value.c_str());
    xmlFile.flush();
  }
}

void logger::xmlSetContent(float value)
{
  // kernel_latency.cpp calls this instead of xmlRecord for its single metric.
  // The test group tag doubles as the metric name here.
  if (contextStack.size() == 4)
    recordMetric("latency", value);

  if (enableXml)
  {
    std::stringstream ss;
    ss << value;
    xw->content(ss.str().c_str());
    xmlFile.flush();
  }
}

void logger::xmlCloseTag()
{
  if (!contextStack.empty())
    contextStack.pop_back();

  if (enableXml)
  {
    xw->closeElt();
    xmlFile.flush();
  }
}

void logger::xmlRecord(std::string tag, std::string value)
{
  if (enableXml)
  {
    xw->openElt(tag.c_str());
    xw->content(value.c_str());
    xw->closeElt();
    xmlFile.flush();
  }
}

void logger::xmlRecord(std::string tag, float value)
{
  // xmlRecord is always called at context depth 4 (clpeak > platform > device
  // > test_group).  Collect the result and optionally print the compare delta.
  if (contextStack.size() == 4)
    recordMetric(tag, value);

  if (enableXml)
  {
    std::stringstream ss;
    ss << value;
    xw->openElt(tag.c_str());
    xw->content(ss.str().c_str());
    xw->closeElt();
    xmlFile.flush();
  }
}

// ---- private helper -------------------------------------------------------

void logger::recordMetric(const std::string &metric, float value)
{
  if (contextStack.size() < 4)
    return;

  ResultEntry e;
  e.platform = contextStack[1].attribs.count("name")           ? contextStack[1].attribs.at("name")           : "";
  e.device   = contextStack[2].attribs.count("name")           ? contextStack[2].attribs.at("name")           : "";
  e.driver   = contextStack[2].attribs.count("driver_version") ? contextStack[2].attribs.at("driver_version") : "";
  e.test     = contextStack[3].tag;
  e.unit     = contextStack[3].attribs.count("unit")           ? contextStack[3].attribs.at("unit")           : "";
  e.metric   = metric;
  e.value    = value;
  results.push_back(e);

  if (!compareEnabled)
    return;

  auto it = baseline.find(e.key());
  if (it == baseline.end())
    return;

  float base  = it->second;
  float delta = (base != 0.0f) ? 100.0f * (value - base) / base : 0.0f;

  // Sign and magnitude string, e.g. "+2.6%" or "-1.5%" or "~0.0%"
  char sign = (delta >= 0.0f) ? '+' : '-';
  float absDelta = (delta < 0.0f) ? -delta : delta;

  std::cout << "        "
            << "(was " << std::fixed << std::setprecision(2) << base
            << ",  " << sign << std::setprecision(1) << absDelta << "%)"
            << "\n";
  std::cout.flush();
}
