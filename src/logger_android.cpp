#include <logger.h>
#include <iomanip>
#include <sstream>

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

logger::logger(bool _enableXml, string _xmlFileName) : enableXml(false)
{
}

logger::~logger()
{
}

// xml dump disabled
void logger::xmlOpenTag(string tag)
{
}

void logger::xmlAppendAttribs(string key, string value)
{
}

void logger::xmlAppendAttribs(string key, uint value)
{
}

void logger::xmlSetContent(string value)
{
}

void logger::xmlSetContent(float value)
{
}

void logger::xmlCloseTag()
{
}

void logger::xmlRecord(string tag, string value)
{
}

void logger::xmlRecord(string tag, float value)
{
}
