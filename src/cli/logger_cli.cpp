#include <cli/logger_cli.h>
#include <iomanip>
#include <sstream>

// ---- stdout output --------------------------------------------------------

void LoggerCli::print(std::string str)
{
    std::cout << str;
    std::cout.flush();
}

void LoggerCli::print(double val)
{
    std::cout << std::setprecision(2) << std::fixed << val;
    std::cout.flush();
}

void LoggerCli::print(float val)
{
    std::cout << std::setprecision(2) << std::fixed << val;
    std::cout.flush();
}

void LoggerCli::print(int val)
{
    std::cout << val;
    std::cout.flush();
}

void LoggerCli::print(unsigned int val)
{
    std::cout << val;
    std::cout.flush();
}

// ---- Baseline delta on stdout --------------------------------------------

void LoggerCli::onMetricEmitted(const ResultEntry &e, float value)
{
    if (!compareEnabled)
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
