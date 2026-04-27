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

// ---- CSV parser -----------------------------------------------------------
// Parses the RFC-4180-style quoted format produced by saveCsv.
// Fields 0-5 are double-quoted strings; field 6 is an unquoted float.

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
                // "" inside quotes is an escaped quote character
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
            if      (c == '"')  { inQuotes = true; }
            else if (c == ',')  { fields.push_back(field); field.clear(); }
            else if (c == '\r') { /* skip CR in CRLF line endings */ }
            else                { field += c; }
        }
    }
    fields.push_back(field); // last field (unquoted value)
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
    while (std::getline(f, line))
    {
        if (header) { header = false; continue; } // skip header row
        if (line.empty() || line[0] == '\r') continue;

        std::vector<std::string> fields = parseCsvLine(line);
        if (fields.size() < 7) continue;

        ResultEntry e;
        e.platform = fields[0];
        e.device   = fields[1];
        e.driver   = fields[2];
        e.test     = fields[3];
        e.metric   = fields[4];
        e.unit     = fields[5];
        try
        {
            size_t consumed = 0;
            e.value = std::stof(fields[6], &consumed);
            if (consumed == 0) continue;
        }
        catch (...) { continue; }

        if (!e.platform.empty() && !e.test.empty() && !e.metric.empty())
            store.push_back(e);
    }
    if (store.empty())
        std::cerr << "clpeak: warning: no valid entries found in: " << filename << "\n";
    return store;
}

// Dispatch to the right parser based on the file extension (.csv, .xml, or
// JSON for anything else).
ResultStore loadResultFile(const std::string &filename)
{
    // Extract and lowercase the extension
    std::string ext;
    size_t dot = filename.rfind('.');
    if (dot != std::string::npos)
    {
        ext = filename.substr(dot);
        for (char &c : ext) c = static_cast<char>(tolower(static_cast<unsigned char>(c)));
    }
    if (ext == ".csv")  return loadCsv(filename);
    if (ext == ".xml")  return loadXml(filename);
    return loadJson(filename);
}

// ---- XML parser -----------------------------------------------------------
// Parses the format produced by the xmlWriter / --xml-file path.
// No external XML library required: the format is regular enough for a
// line-by-line state-machine that tracks the four nesting levels
// (clpeak > platform > device > test_group > metric_leaf).

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

// Extract the tag name from an opening-tag line (already left-trimmed).
// Returns "" for closing tags, processing instructions, or malformed input.
static std::string xmlTagName(const std::string &t)
{
    if (t.size() < 2 || t[0] != '<' || t[1] == '/' || t[1] == '?')
        return "";
    std::string name;
    for (size_t i = 1; i < t.size(); i++)
    {
        char c = t[i];
        if (c == ' ' || c == '>' || c == '/') break;
        name += c;
    }
    return name;
}

// Extract the value of an XML attribute ( attr="value" ).
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

// Try to read a complete leaf element from a single trimmed line:
//   <tagname>content</tagname>   (no attributes — metrics never have any)
// Fills tag and content; returns false if the pattern does not match.
static bool xmlLeaf(const std::string &t,
                    std::string &tag, std::string &content)
{
    tag = xmlTagName(t);
    if (tag.empty()) return false;

    // Opening tag must be exactly "<tag>" with no attributes
    std::string open  = "<"  + tag + ">";
    std::string close = "</" + tag + ">";
    if (t.compare(0, open.size(), open) != 0) return false;

    size_t closePos = t.find(close, open.size());
    if (closePos == std::string::npos) return false;

    content = xmlUnescape(t.substr(open.size(), closePos - open.size()));
    return true;
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

    std::string platform, device, driver, test, unit;
    std::string line;

    while (std::getline(f, line))
    {
        // Left-trim
        size_t first = line.find_first_not_of(" \t\r");
        if (first == std::string::npos) continue;
        const std::string t = line.substr(first);
        if (t.empty()) continue;

        // Skip processing instructions and closing tags
        if (t.rfind("<?", 0) == 0 || t.rfind("</", 0) == 0) continue;

        const std::string tag = xmlTagName(t);
        if (tag.empty()) continue;

        // ---- Structural tags ---------------------------------------------

        if (tag == "clpeak")   { continue; } // root: nothing to extract here

        if (tag == "platform")
        {
            platform = xmlAttr(t, "name");
            device.clear(); driver.clear(); test.clear(); unit.clear();
            continue;
        }

        if (tag == "device")
        {
            device = xmlAttr(t, "name");
            driver = xmlAttr(t, "driver_version");
            test.clear(); unit.clear();
            continue;
        }

        // ---- Test group opener (always carries unit= attribute) ----------

        if (t.find(" unit=\"") != std::string::npos)
        {
            test = tag;
            unit = xmlAttr(t, "unit");

            // Single-value tests pack open + content + close on one line:
            //   <kernel_launch_latency unit="us">2.35</kernel_launch_latency>
            std::string openTag = "<" + test;
            size_t openEnd = t.find('>', openTag.size());
            if (openEnd != std::string::npos)
            {
                std::string closeTag = "</" + test + ">";
                size_t closePos = t.find(closeTag, openEnd + 1);
                if (closePos != std::string::npos)
                {
                    std::string content = xmlUnescape(
                        t.substr(openEnd + 1, closePos - openEnd - 1));
                    try
                    {
                        ResultEntry e;
                        e.platform = platform;
                        e.device   = device;
                        e.driver   = driver;
                        e.test     = test;
                        e.metric   = "latency";
                        e.unit     = unit;
                        e.value    = std::stof(content);
                        if (!e.platform.empty() && !e.test.empty())
                            store.push_back(e);
                    }
                    catch (...) {}
                }
            }
            continue;
        }

        // ---- Metric leaf: <tag>value</tag> -------------------------------

        if (platform.empty() || device.empty() || test.empty()) continue;

        std::string metricTag, content;
        if (xmlLeaf(t, metricTag, content))
        {
            try
            {
                ResultEntry e;
                e.platform = platform;
                e.device   = device;
                e.driver   = driver;
                e.test     = test;
                e.metric   = metricTag;
                e.unit     = unit;
                e.value    = std::stof(content);
                store.push_back(e);
            }
            catch (...) {}
        }
    }

    if (store.empty())
        std::cerr << "clpeak: warning: no valid entries found in: " << filename << "\n";
    return store;
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
