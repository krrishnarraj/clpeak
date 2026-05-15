#ifndef LOGGER_CLI_HPP
#define LOGGER_CLI_HPP

#include <common/logger.h>

// Desktop CLI logger: print() → stdout, onMetricEmitted() prints baseline delta.
class LoggerCli : public logger
{
public:
    using logger::logger;  // inherit constructor

    void print(std::string str) override;
    void print(double val) override;
    void print(float val) override;
    void print(int val) override;
    void print(unsigned int val) override;

protected:
    void onMetricEmitted(const ResultEntry &e, float value) override;
};

#endif // LOGGER_CLI_HPP
