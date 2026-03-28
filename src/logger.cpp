#include <logger.h>
#include <iomanip>
#include <sstream>

logger::logger(bool _enableXml,     string _xmlFileName,
               bool _enableJson,    string _jsonFileName,
               bool _enableCsv,     string _csvFileName,
               string _compareFileName)
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
      cout << "clpeak: comparing against baseline: " << _compareFileName
           << " (" << baseline.size() << " entries)" << "\n";
      cout.flush();
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

void logger::print(string str)
{
  cout << str;
  cout.flush();
}

void logger::print(double val)
{
  cout << setprecision(2) << fixed;
  cout << val;
  cout.flush();
}

void logger::print(float val)
{
  cout << setprecision(2) << fixed;
  cout << val;
  cout.flush();
}

void logger::print(int val)
{
  cout << val;
  cout.flush();
}

void logger::print(unsigned int val)
{
  cout << val;
  cout.flush();
}

// ---- XML / context-tracking -----------------------------------------------

void logger::xmlOpenTag(string tag)
{
  contextStack.push_back({tag, {}});

  if (enableXml)
  {
    xw->openElt(tag.c_str());
    xmlFile.flush();
  }
}

void logger::xmlAppendAttribs(string key, string value)
{
  if (!contextStack.empty())
    contextStack.back().attribs[key] = value;

  if (enableXml)
  {
    xw->attr(key.c_str(), value.c_str());
    xmlFile.flush();
  }
}

void logger::xmlAppendAttribs(string key, uint value)
{
  stringstream ss;
  ss << value;
  // Delegate to string overload so the context frame is also updated
  xmlAppendAttribs(key, ss.str());
}

void logger::xmlSetContent(string value)
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
    stringstream ss;
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

void logger::xmlRecord(string tag, string value)
{
  if (enableXml)
  {
    xw->openElt(tag.c_str());
    xw->content(value.c_str());
    xw->closeElt();
    xmlFile.flush();
  }
}

void logger::xmlRecord(string tag, float value)
{
  // xmlRecord is always called at context depth 4 (clpeak > platform > device
  // > test_group).  Collect the result and optionally print the compare delta.
  if (contextStack.size() == 4)
    recordMetric(tag, value);

  if (enableXml)
  {
    stringstream ss;
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

  // Print on its own line, indented to nest visually under the metric value.
  // TAB TAB TAB TAB = 8 spaces; "(was X.XX, +/-d.d%)" gives full context.
  cout << "        "
       << "(was " << fixed << setprecision(2) << base
       << ",  " << sign << setprecision(1) << absDelta << "%)"
       << "\n";
  cout.flush();
}
