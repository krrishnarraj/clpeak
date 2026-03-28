#include <result_store.h>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <iostream>

BaselineMap buildBaselineMap(const ResultStore &store)
{
    BaselineMap m;
    for (const ResultEntry &e : store)
        m[e.key()] = e.value;
    return m;
}

// ---- JSON helpers ---------------------------------------------------------

static std::string jsonEscape(const std::string &s)
{
    std::string out;
    out.reserve(s.size() + 4);
    for (char c : s)
    {
        switch (c)
        {
        case '"':  out += "\\\""; break;
        case '\\': out += "\\\\"; break;
        case '\n': out += "\\n";  break;
        case '\r': out += "\\r";  break;
        default:   out += c;      break;
        }
    }
    return out;
}

void saveJson(const ResultStore &store, const std::string &filename)
{
    std::ofstream f(filename);
    if (!f.is_open())
    {
        std::cerr << "clpeak: cannot open JSON output file: " << filename << "\n";
        return;
    }
    f << "[\n";
    for (size_t i = 0; i < store.size(); i++)
    {
        const ResultEntry &e = store[i];
        f << "{"
          << "\"platform\":\"" << jsonEscape(e.platform) << "\","
          << "\"device\":\""   << jsonEscape(e.device)   << "\","
          << "\"driver\":\""   << jsonEscape(e.driver)   << "\","
          << "\"test\":\""     << jsonEscape(e.test)     << "\","
          << "\"metric\":\""   << jsonEscape(e.metric)   << "\","
          << "\"unit\":\""     << jsonEscape(e.unit)     << "\","
          << "\"value\":"
          << std::fixed << std::setprecision(4) << e.value
          << "}";
        if (i + 1 < store.size())
            f << ",";
        f << "\n";
    }
    f << "]\n";
}

// ---- CSV helpers ----------------------------------------------------------

static std::string csvField(const std::string &s)
{
    // Wrap in double-quotes, escaping embedded quotes as ""
    std::string out = "\"";
    for (char c : s)
    {
        if (c == '"') out += "\"\"";
        else          out += c;
    }
    out += "\"";
    return out;
}

void saveCsv(const ResultStore &store, const std::string &filename)
{
    std::ofstream f(filename);
    if (!f.is_open())
    {
        std::cerr << "clpeak: cannot open CSV output file: " << filename << "\n";
        return;
    }
    f << "platform,device,driver,test,metric,unit,value\n";
    for (const ResultEntry &e : store)
    {
        f << csvField(e.platform) << ","
          << csvField(e.device)   << ","
          << csvField(e.driver)   << ","
          << csvField(e.test)     << ","
          << csvField(e.metric)   << ","
          << csvField(e.unit)     << ","
          << std::fixed << std::setprecision(4) << e.value
          << "\n";
    }
}

// ---- JSON parser ----------------------------------------------------------
// Handles the one-object-per-line format produced by saveJson.  Only the
// fields needed for ResultEntry are extracted; everything else is ignored.

static std::string extractStr(const std::string &line, const std::string &key)
{
    std::string needle = "\"" + key + "\":\"";
    size_t pos = line.find(needle);
    if (pos == std::string::npos)
        return "";
    pos += needle.size();
    std::string out;
    for (; pos < line.size() && line[pos] != '"'; pos++)
    {
        if (line[pos] == '\\' && pos + 1 < line.size())
        {
            char next = line[++pos];
            if      (next == '"')  out += '"';
            else if (next == '\\') out += '\\';
            else if (next == 'n')  out += '\n';
            else if (next == 'r')  out += '\r';
            else                   out += next;
        }
        else
        {
            out += line[pos];
        }
    }
    return out;
}

static float extractFloat(const std::string &line, const std::string &key)
{
    std::string needle = "\"" + key + "\":";
    size_t pos = line.find(needle);
    if (pos == std::string::npos)
        return 0.0f;
    pos += needle.size();
    while (pos < line.size() && line[pos] == ' ')
        pos++;
    // Must not be a quoted string
    if (pos >= line.size() || line[pos] == '"')
        return 0.0f;
    try
    {
        size_t consumed = 0;
        float v = std::stof(line.substr(pos), &consumed);
        return (consumed > 0) ? v : 0.0f;
    }
    catch (...)
    {
        return 0.0f;
    }
}

ResultStore loadJson(const std::string &filename)
{
    ResultStore store;
    std::ifstream f(filename);
    if (!f.is_open())
    {
        std::cerr << "clpeak: cannot open compare file: " << filename << "\n";
        return store;
    }
    std::string line;
    while (std::getline(f, line))
    {
        // Object lines contain a '{' — skip array brackets and blank lines
        size_t brace = line.find('{');
        if (brace == std::string::npos)
            continue;
        const std::string obj = line.substr(brace);

        ResultEntry e;
        e.platform = extractStr  (obj, "platform");
        e.device   = extractStr  (obj, "device");
        e.driver   = extractStr  (obj, "driver");
        e.test     = extractStr  (obj, "test");
        e.metric   = extractStr  (obj, "metric");
        e.unit     = extractStr  (obj, "unit");
        e.value    = extractFloat(obj, "value");

        if (!e.platform.empty() && !e.test.empty() && !e.metric.empty())
            store.push_back(e);
    }
    if (store.empty())
        std::cerr << "clpeak: warning: no valid entries found in: " << filename << "\n";
    return store;
}
