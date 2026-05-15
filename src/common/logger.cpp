#include <common/logger.h>
#include <sstream>

// ---- Constructor ----------------------------------------------------------

logger::logger(std::string compareFileName)
    : compareEnabled(!compareFileName.empty())
{
    if (compareEnabled)
        baseline = buildBaselineMap(loadResultFile(compareFileName));
}

// ---- recordSkip -----------------------------------------------------------

void logger::recordSkip(const std::string &metric, ResultStatus status,
                        const std::string &reason)
{
    emit(metric, status, 0.0f, reason);
}

// ---- Result-scope recording -----------------------------------------------

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

// ---- emit (common: builds ResultEntry, pushes to results, calls hook) -----

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

    onMetricEmitted(e, value);
}

// Default hook: no-op
void logger::onMetricEmitted(const ResultEntry &, float) {}
