#include <result_store.h>
#include <version.h>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <cstring>

// ---- OS name --------------------------------------------------------------
// Mirrors the OS_NAME macro in common.h.  Duplicated here so result_store.cpp
// stays free of the OpenCL include chain in common.h.

static const char *osName()
{
#if defined(__APPLE__) || defined(__MACOSX)
    return "Macintosh";
#elif defined(__ANDROID__)
    return "Android";
#elif defined(_WIN32)
  #if defined(_WIN64)
    return "Win64";
  #else
    return "Win32";
  #endif
#elif defined(__linux__)
  #if defined(__x86_64__)
    return "Linux x64";
  #elif defined(__i386__)
    return "Linux x86";
  #elif defined(__aarch64__)
    return "Linux ARM64";
  #elif defined(__arm__)
    return "Linux ARM";
  #else
    return "Linux unknown";
  #endif
#elif defined(__FreeBSD__)
    return "FreeBSD";
#else
    return "Unknown";
#endif
}

// ---- Enum <-> string ------------------------------------------------------

const char *categoryString(Category c)
{
    switch (c)
    {
    case Category::FpCompute:  return "fp_compute";
    case Category::IntCompute: return "int_compute";
    case Category::Bandwidth:  return "bandwidth";
    case Category::Latency:    return "latency";
    case Category::Unknown:    return "";
    }
    return "";
}

const char *statusString(ResultStatus s)
{
    switch (s)
    {
    case ResultStatus::Ok:          return "ok";
    case ResultStatus::Unsupported: return "unsupported";
    case ResultStatus::Skipped:     return "skipped";
    case ResultStatus::Error:       return "error";
    }
    return "ok";
}

Category categoryFromString(const std::string &s)
{
    if (s == "fp_compute")  return Category::FpCompute;
    if (s == "int_compute") return Category::IntCompute;
    if (s == "bandwidth")   return Category::Bandwidth;
    if (s == "latency")     return Category::Latency;
    return Category::Unknown;
}

ResultStatus statusFromString(const std::string &s)
{
    if (s == "unsupported") return ResultStatus::Unsupported;
    if (s == "skipped")     return ResultStatus::Skipped;
    if (s == "error")       return ResultStatus::Error;
    return ResultStatus::Ok;
}

Category categoryFromUnit(const std::string &unit)
{
    if (unit == "gflops" || unit == "tflops") return Category::FpCompute;
    if (unit == "gops"   || unit == "tops")   return Category::IntCompute;
    if (unit == "gbps")                       return Category::Bandwidth;
    if (unit == "us")                         return Category::Latency;
    return Category::Unknown;
}

// ---- Baseline -------------------------------------------------------------

BaselineMap buildBaselineMap(const ResultStore &store)
{
    BaselineMap m;
    for (const ResultEntry &e : store)
        if (e.status == ResultStatus::Ok)
            m[e.key()] = e.value;
    return m;
}

// ---- Shared escapers ------------------------------------------------------

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

static std::string csvField(const std::string &s)
{
    std::string out = "\"";
    for (char c : s)
    {
        if (c == '"') out += "\"\"";
        else          out += c;
    }
    out += "\"";
    return out;
}

static std::string xmlEscape(const std::string &s)
{
    std::string out;
    out.reserve(s.size());
    for (char c : s)
    {
        switch (c)
        {
        case '&':  out += "&amp;";  break;
        case '<':  out += "&lt;";   break;
        case '>':  out += "&gt;";   break;
        case '\'': out += "&apos;"; break;
        case '"':  out += "&quot;"; break;
        default:   out += c;        break;
        }
    }
    return out;
}

static std::string fmtValue(float v)
{
    std::stringstream ss;
    ss << std::fixed << std::setprecision(4) << v;
    return ss.str();
}

// ---- JSON save ------------------------------------------------------------
// Self-describing wrapper:
//   {"format_version":2,"clpeak_version":"...","os":"...","entries":[ … ]}
// Each entry on its own line for easy line-by-line parsing.

void saveJson(const ResultStore &store, const std::string &filename)
{
    std::ofstream f(filename);
    if (!f.is_open())
    {
        std::cerr << "clpeak: cannot open JSON output file: " << filename << "\n";
        return;
    }
    f << "{\"format_version\":" << CLPEAK_FORMAT_VERSION
      << ",\"clpeak_version\":\"" << jsonEscape(CLPEAK_VERSION_STR) << "\""
      << ",\"os\":\"" << jsonEscape(osName()) << "\""
      << ",\"entries\":[\n";
    for (size_t i = 0; i < store.size(); i++)
    {
        const ResultEntry &e = store[i];
        f << "{"
          << "\"backend\":\""  << jsonEscape(e.backend)  << "\","
          << "\"platform\":\"" << jsonEscape(e.platform) << "\","
          << "\"device\":\""   << jsonEscape(e.device)   << "\","
          << "\"driver\":\""   << jsonEscape(e.driver)   << "\","
          << "\"category\":\"" << jsonEscape(e.category) << "\","
          << "\"test\":\""     << jsonEscape(e.test)     << "\","
          << "\"metric\":\""   << jsonEscape(e.metric)   << "\","
          << "\"unit\":\""     << jsonEscape(e.unit)     << "\"";

        if (e.status == ResultStatus::Ok)
        {
            f << ",\"value\":" << fmtValue(e.value);
        }
        else
        {
            f << ",\"status\":\"" << statusString(e.status) << "\""
              << ",\"reason\":\"" << jsonEscape(e.reason) << "\"";
        }
        f << "}";
        if (i + 1 < store.size())
            f << ",";
        f << "\n";
    }
    f << "]}\n";
}

// ---- CSV save -------------------------------------------------------------
// Header:
//   format_version,backend,platform,device,driver,category,test,metric,unit,status,value,reason
// `format_version` repeats per row to keep CSV self-describing without a
// preamble (some tools strip the first row).  Ok rows leave `reason` empty
// and populate `value`; non-Ok rows leave `value` empty and populate
// `reason`.

void saveCsv(const ResultStore &store, const std::string &filename)
{
    std::ofstream f(filename);
    if (!f.is_open())
    {
        std::cerr << "clpeak: cannot open CSV output file: " << filename << "\n";
        return;
    }
    f << "format_version,backend,platform,device,driver,category,test,metric,unit,status,value,reason\n";
    for (const ResultEntry &e : store)
    {
        f << CLPEAK_FORMAT_VERSION   << ","
          << csvField(e.backend)     << ","
          << csvField(e.platform)    << ","
          << csvField(e.device)      << ","
          << csvField(e.driver)      << ","
          << csvField(e.category)    << ","
          << csvField(e.test)        << ","
          << csvField(e.metric)      << ","
          << csvField(e.unit)        << ","
          << csvField(statusString(e.status)) << ",";
        if (e.status == ResultStatus::Ok)
            f << fmtValue(e.value);
        f << "," << csvField(e.reason) << "\n";
    }
}

// ---- XML save -------------------------------------------------------------
// Tree:
//   <clpeak format_version="2" clpeak_version="..." os="...">
//     <run backend="..." platform="..." device="..." driver="...">
//       <category name="fp_compute">
//         <test name="single_precision_compute" unit="gflops">
//           <metric name="float">12500.5</metric>
//           <metric name="double" status="unsupported" reason="..."/>
//         </test>
//       </category>
//     </run>
//   </clpeak>
//
// Streamed in a single linear pass; relies on entries already being grouped
// by (backend, platform, device, driver) -> category -> test order, which
// the logger guarantees because backends emit them in run order.

namespace {
struct XmlPos {
    std::string backend, platform, device, driver;
    std::string category;
    std::string test, unit;
    bool inRun  = false;
    bool inCat  = false;
    bool inTest = false;
};

bool sameRun(const XmlPos &p, const ResultEntry &e)
{
    return p.inRun &&
        p.backend  == e.backend  &&
        p.platform == e.platform &&
        p.device   == e.device   &&
        p.driver   == e.driver;
}

void closeTest(std::ofstream &f, XmlPos &p)
{
    if (!p.inTest) return;
    f << "      </test>\n";
    p.inTest = false;
    p.test.clear();
    p.unit.clear();
}

void closeCategory(std::ofstream &f, XmlPos &p)
{
    closeTest(f, p);
    if (!p.inCat) return;
    f << "    </category>\n";
    p.inCat = false;
    p.category.clear();
}

void closeRun(std::ofstream &f, XmlPos &p)
{
    closeCategory(f, p);
    if (!p.inRun) return;
    f << "  </run>\n";
    p.inRun = false;
}
} // namespace

void saveXml(const ResultStore &store, const std::string &filename)
{
    std::ofstream f(filename);
    if (!f.is_open())
    {
        std::cerr << "clpeak: cannot open XML output file: " << filename << "\n";
        return;
    }

    f << "<?xml version=\"1.0\" encoding=\"utf-8\"?>\n"
      << "<clpeak format_version=\"" << CLPEAK_FORMAT_VERSION << "\""
      << " clpeak_version=\""        << xmlEscape(CLPEAK_VERSION_STR) << "\""
      << " os=\""                    << xmlEscape(osName()) << "\">\n";

    XmlPos p;

    for (const ResultEntry &e : store)
    {
        if (!sameRun(p, e))
        {
            closeRun(f, p);
            f << "  <run"
              << " backend=\""  << xmlEscape(e.backend)  << "\""
              << " platform=\"" << xmlEscape(e.platform) << "\""
              << " device=\""   << xmlEscape(e.device)   << "\""
              << " driver=\""   << xmlEscape(e.driver)   << "\">\n";
            p = XmlPos();
            p.inRun    = true;
            p.backend  = e.backend;
            p.platform = e.platform;
            p.device   = e.device;
            p.driver   = e.driver;
        }

        if (!p.inCat || p.category != e.category)
        {
            closeCategory(f, p);
            f << "    <category name=\"" << xmlEscape(e.category) << "\">\n";
            p.inCat    = true;
            p.category = e.category;
        }

        if (!p.inTest || p.test != e.test || p.unit != e.unit)
        {
            closeTest(f, p);
            f << "      <test name=\"" << xmlEscape(e.test) << "\""
              << " unit=\"" << xmlEscape(e.unit) << "\">\n";
            p.inTest = true;
            p.test   = e.test;
            p.unit   = e.unit;
        }

        f << "        <metric name=\"" << xmlEscape(e.metric) << "\"";
        if (e.status == ResultStatus::Ok)
        {
            f << ">" << fmtValue(e.value) << "</metric>\n";
        }
        else
        {
            f << " status=\"" << statusString(e.status) << "\""
              << " reason=\"" << xmlEscape(e.reason)    << "\"/>\n";
        }
    }

    closeRun(f, p);
    f << "</clpeak>\n";
}

// ---- Loaders --------------------------------------------------------------
// All three loaders reject v1 (or unversioned) files with a stderr warning
// and return an empty store -- callers (--compare baselines) then have no
// matches and report deltas accordingly.

static bool checkVersion(int got, const std::string &filename)
{
    if (got == CLPEAK_FORMAT_VERSION) return true;
    std::cerr << "clpeak: " << filename
              << " is format_version=" << got
              << "; this build expects v" << CLPEAK_FORMAT_VERSION
              << ". Regenerate the file with this version of clpeak.\n";
    return false;
}

// ---- JSON loader ----------------------------------------------------------
// Accepts the new wrapper form and the bare array form (latter is
// unversioned -> rejected).  Field extraction is intentionally permissive:
// any line with both '{' and '"backend":' is parsed.

static std::string jsonExtractStr(const std::string &line, const std::string &key)
{
    std::string needle = "\"" + key + "\":\"";
    size_t pos = line.find(needle);
    if (pos == std::string::npos) return "";
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

static bool jsonHasKey(const std::string &line, const std::string &key)
{
    return line.find("\"" + key + "\":") != std::string::npos;
}

static float jsonExtractFloat(const std::string &line, const std::string &key)
{
    std::string needle = "\"" + key + "\":";
    size_t pos = line.find(needle);
    if (pos == std::string::npos) return 0.0f;
    pos += needle.size();
    while (pos < line.size() && line[pos] == ' ') pos++;
    if (pos >= line.size() || line[pos] == '"') return 0.0f;
    try
    {
        size_t consumed = 0;
        float v = std::stof(line.substr(pos), &consumed);
        return (consumed > 0) ? v : 0.0f;
    }
    catch (...) { return 0.0f; }
}

static int jsonExtractInt(const std::string &line, const std::string &key)
{
    std::string needle = "\"" + key + "\":";
    size_t pos = line.find(needle);
    if (pos == std::string::npos) return 0;
    pos += needle.size();
    while (pos < line.size() && line[pos] == ' ') pos++;
    try
    {
        size_t consumed = 0;
        int v = std::stoi(line.substr(pos), &consumed);
        return (consumed > 0) ? v : 0;
    }
    catch (...) { return 0; }
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

    bool versionChecked = false;
    bool versionOk      = false;
    std::string line;

    while (std::getline(f, line))
    {
        // Header line carries format_version.  Find it once before parsing
        // entries.
        if (!versionChecked && line.find("\"format_version\":") != std::string::npos)
        {
            int v = jsonExtractInt(line, "format_version");
            versionOk      = checkVersion(v, filename);
            versionChecked = true;
            if (!versionOk) return store;
            continue;
        }

        // Entry lines have a '{'.  Skip anything else (array brackets,
        // the wrapper-object braces, blank lines).
        size_t brace = line.find('{');
        if (brace == std::string::npos) continue;

        if (!versionChecked)
        {
            // Reached an object line before seeing format_version.  Either
            // a v1 array-of-objects file or a hand-edited file missing the
            // wrapper.  Reject and stop.
            checkVersion(1, filename);
            return store;
        }

        if (!jsonHasKey(line, "backend")) continue;

        const std::string obj = line.substr(brace);

        ResultEntry e;
        e.backend  = jsonExtractStr(obj, "backend");
        e.platform = jsonExtractStr(obj, "platform");
        e.device   = jsonExtractStr(obj, "device");
        e.driver   = jsonExtractStr(obj, "driver");
        e.category = jsonExtractStr(obj, "category");
        e.test     = jsonExtractStr(obj, "test");
        e.metric   = jsonExtractStr(obj, "metric");
        e.unit     = jsonExtractStr(obj, "unit");

        if (jsonHasKey(obj, "status"))
        {
            e.status = statusFromString(jsonExtractStr(obj, "status"));
            e.reason = jsonExtractStr(obj, "reason");
        }
        else
        {
            e.status = ResultStatus::Ok;
            e.value  = jsonExtractFloat(obj, "value");
        }

        if (!e.backend.empty() && !e.test.empty() && !e.metric.empty())
            store.push_back(e);
    }

    if (store.empty())
        std::cerr << "clpeak: warning: no valid entries found in: " << filename << "\n";
    return store;
}

// ---- CSV loader -----------------------------------------------------------

static std::vector<std::string> parseCsvLine(const std::string &line)
{
    std::vector<std::string> fields;
    std::string field;
    bool inQuotes = false;
    for (size_t i = 0; i < line.size(); i++)
    {
        char c = line[i];
        if (inQuotes)
        {
            if (c == '"')
            {
                if (i + 1 < line.size() && line[i + 1] == '"')
                {
                    field += '"';
                    i++;
                }
                else
                {
                    inQuotes = false;
                }
            }
            else
            {
                field += c;
            }
        }
        else
        {
            if      (c == '"')  inQuotes = true;
            else if (c == ',')  { fields.push_back(field); field.clear(); }
            else if (c == '\r') { /* skip CR in CRLF */ }
            else                field += c;
        }
    }
    fields.push_back(field);
    return fields;
}

ResultStore loadCsv(const std::string &filename)
{
    ResultStore store;
    std::ifstream f(filename);
    if (!f.is_open())
    {
        std::cerr << "clpeak: cannot open compare file: " << filename << "\n";
        return store;
    }
    std::string line;
    bool header = true;
    bool versionChecked = false;

    while (std::getline(f, line))
    {
        if (header) { header = false; continue; }
        if (line.empty() || line[0] == '\r') continue;

        std::vector<std::string> fields = parseCsvLine(line);
        // v2 has 12 fields; reject anything shorter (v1 had 7).
        if (fields.size() < 12)
        {
            if (!versionChecked)
            {
                checkVersion(1, filename);
                return store;
            }
            continue;
        }

        if (!versionChecked)
        {
            int v = 0;
            try { v = std::stoi(fields[0]); } catch (...) { v = 0; }
            if (!checkVersion(v, filename)) return store;
            versionChecked = true;
        }

        ResultEntry e;
        e.backend  = fields[1];
        e.platform = fields[2];
        e.device   = fields[3];
        e.driver   = fields[4];
        e.category = fields[5];
        e.test     = fields[6];
        e.metric   = fields[7];
        e.unit     = fields[8];
        e.status   = statusFromString(fields[9]);
        if (e.status == ResultStatus::Ok)
        {
            try
            {
                size_t consumed = 0;
                e.value = std::stof(fields[10], &consumed);
                if (consumed == 0) continue;
            }
            catch (...) { continue; }
        }
        e.reason = fields[11];

        if (!e.backend.empty() && !e.test.empty() && !e.metric.empty())
            store.push_back(e);
    }
    if (store.empty())
        std::cerr << "clpeak: warning: no valid entries found in: " << filename << "\n";
    return store;
}

// ---- XML loader -----------------------------------------------------------
// Line-by-line state machine.  Tracks current run/category/test from
// opening tags; emits an entry on each <metric> leaf.

static std::string xmlUnescape(const std::string &s)
{
    std::string out;
    out.reserve(s.size());
    for (size_t i = 0; i < s.size(); i++)
    {
        if (s[i] != '&') { out += s[i]; continue; }
        if      (s.compare(i, 5, "&amp;")  == 0) { out += '&';  i += 4; }
        else if (s.compare(i, 4, "&lt;")   == 0) { out += '<';  i += 3; }
        else if (s.compare(i, 4, "&gt;")   == 0) { out += '>';  i += 3; }
        else if (s.compare(i, 6, "&apos;") == 0) { out += '\''; i += 5; }
        else if (s.compare(i, 6, "&quot;") == 0) { out += '"';  i += 5; }
        else                                      { out += s[i]; }
    }
    return out;
}

static std::string xmlAttr(const std::string &t, const std::string &attr)
{
    std::string needle = " " + attr + "=\"";
    size_t pos = t.find(needle);
    if (pos == std::string::npos) return "";
    pos += needle.size();
    std::string raw;
    for (; pos < t.size() && t[pos] != '"'; pos++) raw += t[pos];
    return xmlUnescape(raw);
}

static int xmlAttrInt(const std::string &t, const std::string &attr)
{
    std::string s = xmlAttr(t, attr);
    if (s.empty()) return 0;
    try { return std::stoi(s); } catch (...) { return 0; }
}

ResultStore loadXml(const std::string &filename)
{
    ResultStore store;
    std::ifstream f(filename);
    if (!f.is_open())
    {
        std::cerr << "clpeak: cannot open compare file: " << filename << "\n";
        return store;
    }

    std::string backend, platform, device, driver, category, test, unit;
    std::string line;
    bool versionChecked = false;

    while (std::getline(f, line))
    {
        size_t first = line.find_first_not_of(" \t\r");
        if (first == std::string::npos) continue;
        const std::string t = line.substr(first);
        if (t.empty() || t[0] != '<') continue;
        if (t.rfind("<?", 0) == 0 || t.rfind("</", 0) == 0) continue;

        if (t.rfind("<clpeak", 0) == 0)
        {
            int v = xmlAttrInt(t, "format_version");
            if (!checkVersion(v, filename)) return store;
            versionChecked = true;
            continue;
        }

        if (!versionChecked)
        {
            checkVersion(0, filename);
            return store;
        }

        if (t.rfind("<run", 0) == 0)
        {
            backend  = xmlAttr(t, "backend");
            platform = xmlAttr(t, "platform");
            device   = xmlAttr(t, "device");
            driver   = xmlAttr(t, "driver");
            category.clear(); test.clear(); unit.clear();
            continue;
        }

        if (t.rfind("<category", 0) == 0)
        {
            category = xmlAttr(t, "name");
            test.clear(); unit.clear();
            continue;
        }

        if (t.rfind("<test", 0) == 0)
        {
            test = xmlAttr(t, "name");
            unit = xmlAttr(t, "unit");
            continue;
        }

        if (t.rfind("<metric", 0) == 0)
        {
            ResultEntry e;
            e.backend  = backend;
            e.platform = platform;
            e.device   = device;
            e.driver   = driver;
            e.category = category;
            e.test     = test;
            e.metric   = xmlAttr(t, "name");
            e.unit     = unit;

            // Self-closing form is unsupported/error/skipped:
            //   <metric name="x" status="..." reason="..."/>
            // Open form is Ok:
            //   <metric name="x">12.34</metric>
            std::string status = xmlAttr(t, "status");
            if (!status.empty())
            {
                e.status = statusFromString(status);
                e.reason = xmlAttr(t, "reason");
            }
            else
            {
                size_t openEnd = t.find('>');
                size_t closePos = (openEnd != std::string::npos)
                    ? t.find("</metric>", openEnd + 1) : std::string::npos;
                if (openEnd == std::string::npos || closePos == std::string::npos)
                    continue;
                std::string content = xmlUnescape(
                    t.substr(openEnd + 1, closePos - openEnd - 1));
                try { e.value = std::stof(content); }
                catch (...) { continue; }
            }

            if (!e.backend.empty() && !e.test.empty() && !e.metric.empty())
                store.push_back(e);
            continue;
        }
    }

    if (store.empty())
        std::cerr << "clpeak: warning: no valid entries found in: " << filename << "\n";
    return store;
}

// ---- Dispatch -------------------------------------------------------------

ResultStore loadResultFile(const std::string &filename)
{
    std::string ext;
    size_t dot = filename.rfind('.');
    if (dot != std::string::npos)
    {
        ext = filename.substr(dot);
        for (char &c : ext) c = static_cast<char>(tolower(static_cast<unsigned char>(c)));
    }
    if (ext == ".csv") return loadCsv(filename);
    if (ext == ".xml") return loadXml(filename);
    return loadJson(filename);
}
