#include <logger.h>
#include <iomanip>
#include <sstream>

// Android logger: print() methods dispatch to the Java UI via JNI.
// All XML / JSON / CSV / compare functionality is stubbed out — output
// formats that write to the filesystem are not useful in the Android app.

logger::logger(bool _enableXml,     string _xmlFileName,
               bool _enableJson,    string _jsonFileName,
               bool _enableCsv,     string _csvFileName,
               string _compareFileName)
  : enableXml(false),
    xw(nullptr),
    enableJson(false),
    enableCsv(false),
    compareEnabled(false)
{
  // Suppress unused-parameter warnings
  (void)_enableXml; (void)_xmlFileName;
  (void)_enableJson; (void)_jsonFileName;
  (void)_enableCsv;  (void)_csvFileName;
  (void)_compareFileName;
}

logger::~logger()
{
}

// ---- stdout → JNI callbacks ----------------------------------------------

void logger::print(string str)
{
  jEnv->CallVoidMethod((*jObj), printCallback, jEnv->NewStringUTF(str.c_str()));
}

void logger::print(double val)
{
  stringstream ss;
  ss << setprecision(2) << fixed << val;
  jEnv->CallVoidMethod((*jObj), printCallback, jEnv->NewStringUTF(ss.str().c_str()));
}

void logger::print(float val)
{
  stringstream ss;
  ss << setprecision(2) << fixed << val;
  jEnv->CallVoidMethod((*jObj), printCallback, jEnv->NewStringUTF(ss.str().c_str()));
}

void logger::print(int val)
{
  stringstream ss;
  ss << val;
  jEnv->CallVoidMethod((*jObj), printCallback, jEnv->NewStringUTF(ss.str().c_str()));
}

void logger::print(unsigned int val)
{
  stringstream ss;
  ss << val;
  jEnv->CallVoidMethod((*jObj), printCallback, jEnv->NewStringUTF(ss.str().c_str()));
}

// ---- XML / result recording — all no-ops on Android ---------------------

void logger::xmlOpenTag(string tag)          { (void)tag; }
void logger::xmlAppendAttribs(string key, string value) { (void)key; (void)value; }
void logger::xmlAppendAttribs(string key, uint value)   { (void)key; (void)value; }
void logger::xmlSetContent(string value)     { (void)value; }
void logger::xmlSetContent(float value)      { (void)value; }
void logger::xmlCloseTag()                   { }
void logger::xmlRecord(string tag, string value) { (void)tag; (void)value; }
void logger::xmlRecord(string tag, float value)  { (void)tag; (void)value; }
void logger::recordMetric(const std::string &metric, float value) { (void)metric; (void)value; }
