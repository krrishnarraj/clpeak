#include <logger.h>
#include <iomanip>

logger::logger(bool _enableXml, string _xmlFile):
			enableXml(_enableXml), xmlFile(_xmlFile)
{
}


logger::~logger()
{
}

int logger::print(string str)
{
	cout << str;
	cout.flush();
}

int logger::print(double val)
{
	cout << setprecision(2) << fixed;
	cout << val;
	cout.flush();
}

int logger::print(float val)
{
	cout << setprecision(2) << fixed;
	cout << val;
	cout.flush();
}

int logger::print(int val)
{
	cout << val;
	cout.flush();
}

int logger::print(unsigned int val)
{
	cout << val;
	cout.flush();
}


// FIXME xml dump TO BE IMPLEMEMTED
int logger::record(string key, string value)
{
}

int logger::record(string key, double value)
{
}

int logger::record(string key, float value)
{
}

int logger::record(string key, int value)
{
}

int logger::record(string key, unsigned int value)
{
}

