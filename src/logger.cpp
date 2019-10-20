#include <logger.h>
#include <iomanip>
#include <sstream>

logger::logger(bool _enableXml, string _xmlFileName) : enableXml(_enableXml)
{
  if (enableXml)
  {
    xmlFile.open(_xmlFileName);
    xw = new xmlWriter(xmlFile);
    xmlFile.flush();
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
}

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

void logger::xmlOpenTag(string tag)
{
  if (enableXml)
  {
    xw->openElt(tag.c_str());
    xmlFile.flush();
  }
}

void logger::xmlAppendAttribs(string key, string value)
{
  if (enableXml)
  {
    xw->attr(key.c_str(), value.c_str());
    xmlFile.flush();
  }
}

void logger::xmlAppendAttribs(string key, uint value)
{
  if (enableXml)
  {
    stringstream ss;
    ss << value;

    xw->attr(key.c_str(), ss.str().c_str());
    xmlFile.flush();
  }
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
    stringstream ss;
    ss << value;

    xw->openElt(tag.c_str());
    xw->content(ss.str().c_str());
    xw->closeElt();
    xmlFile.flush();
  }
}

void logger::xmlRecord(string tag, float value)
{
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
