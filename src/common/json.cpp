#include <common/json.h>

#include <cstdio>
#include <locale>
#include <sstream>

// ── Accessors ──────────────────────────────────────────────────────────────

const JsonValue *JsonValue::find(const std::string &key) const
{
    if (t != Type::Object) return nullptr;
    auto it = obj.find(key);
    return (it == obj.end()) ? nullptr : &it->second;
}

std::string JsonValue::str(const std::string &key, const std::string &def) const
{
    const JsonValue *v = find(key);
    return (v && v->t == Type::String) ? v->s : def;
}

double JsonValue::num(const std::string &key, double def) const
{
    const JsonValue *v = find(key);
    return (v && v->t == Type::Number) ? v->n : def;
}

bool JsonValue::flag(const std::string &key, bool def) const
{
    const JsonValue *v = find(key);
    return (v && v->t == Type::Bool) ? v->b : def;
}

const std::vector<JsonValue> &JsonValue::items() const
{
    static const std::vector<JsonValue> empty;
    return (t == Type::Array) ? arr : empty;
}

// ── Parser ─────────────────────────────────────────────────────────────────

namespace {

class Parser
{
public:
    Parser(const std::string &text) : s(text) {}

    bool parse(JsonValue &out)
    {
        skipWs();
        if (!parseValue(out)) return false;
        skipWs();
        if (i != s.size()) return fail("trailing text after document");
        return true;
    }

    std::string error;

private:
    const std::string &s;
    size_t             i = 0;

    bool fail(const char *what)
    {
        std::ostringstream ss;
        ss << what << " at offset " << i;
        error = ss.str();
        return false;
    }

    void skipWs()
    {
        while (i < s.size() &&
               (s[i] == ' ' || s[i] == '\t' || s[i] == '\n' || s[i] == '\r'))
            i++;
    }

    bool lit(const char *word)
    {
        const size_t n = std::char_traits<char>::length(word);
        if (s.compare(i, n, word) != 0) return false;
        i += n;
        return true;
    }

    bool parseValue(JsonValue &out)
    {
        if (i >= s.size()) return fail("unexpected end of document");
        switch (s[i])
        {
        case '{': return parseObject(out);
        case '[': return parseArray(out);
        case '"': out.t = JsonValue::Type::String; return parseString(out.s);
        case 't':
            if (!lit("true")) return fail("bad literal");
            out.t = JsonValue::Type::Bool; out.b = true;  return true;
        case 'f':
            if (!lit("false")) return fail("bad literal");
            out.t = JsonValue::Type::Bool; out.b = false; return true;
        case 'n':
            if (!lit("null")) return fail("bad literal");
            out.t = JsonValue::Type::Null; return true;
        default:  return parseNumber(out);
        }
    }

    bool parseObject(JsonValue &out)
    {
        out.t = JsonValue::Type::Object;
        i++;                        // '{'
        skipWs();
        if (i < s.size() && s[i] == '}') { i++; return true; }

        for (;;)
        {
            skipWs();
            if (i >= s.size() || s[i] != '"') return fail("expected object key");
            std::string key;
            if (!parseString(key)) return false;
            skipWs();
            if (i >= s.size() || s[i] != ':') return fail("expected ':'");
            i++;
            skipWs();
            JsonValue v;
            if (!parseValue(v)) return false;
            out.obj[key] = std::move(v);
            skipWs();
            if (i < s.size() && s[i] == ',') { i++; continue; }
            if (i < s.size() && s[i] == '}') { i++; return true; }
            return fail("expected ',' or '}'");
        }
    }

    bool parseArray(JsonValue &out)
    {
        out.t = JsonValue::Type::Array;
        i++;                        // '['
        skipWs();
        if (i < s.size() && s[i] == ']') { i++; return true; }

        for (;;)
        {
            skipWs();
            JsonValue v;
            if (!parseValue(v)) return false;
            out.arr.push_back(std::move(v));
            skipWs();
            if (i < s.size() && s[i] == ',') { i++; continue; }
            if (i < s.size() && s[i] == ']') { i++; return true; }
            return fail("expected ',' or ']'");
        }
    }

    // Decode one code point to UTF-8.  Surrogate pairs are joined so the
    // description strings survive a round trip through a writer that escapes
    // non-ASCII; a lone surrogate is passed through as U+FFFD rather than
    // failing the parse.
    static void appendUtf8(unsigned cp, std::string &out)
    {
        if (cp < 0x80) { out += (char)cp; return; }
        if (cp < 0x800)
        {
            out += (char)(0xC0 | (cp >> 6));
            out += (char)(0x80 | (cp & 0x3F));
            return;
        }
        if (cp < 0x10000)
        {
            out += (char)(0xE0 | (cp >> 12));
            out += (char)(0x80 | ((cp >> 6) & 0x3F));
            out += (char)(0x80 | (cp & 0x3F));
            return;
        }
        out += (char)(0xF0 | (cp >> 18));
        out += (char)(0x80 | ((cp >> 12) & 0x3F));
        out += (char)(0x80 | ((cp >> 6) & 0x3F));
        out += (char)(0x80 | (cp & 0x3F));
    }

    bool hex4(unsigned &cp)
    {
        if (i + 4 > s.size()) return false;
        cp = 0;
        for (int k = 0; k < 4; k++)
        {
            const char c = s[i + k];
            cp <<= 4;
            if (c >= '0' && c <= '9')      cp |= (unsigned)(c - '0');
            else if (c >= 'a' && c <= 'f') cp |= (unsigned)(c - 'a' + 10);
            else if (c >= 'A' && c <= 'F') cp |= (unsigned)(c - 'A' + 10);
            else return false;
        }
        i += 4;
        return true;
    }

    bool parseString(std::string &out)
    {
        i++;                        // opening quote
        out.clear();
        while (i < s.size())
        {
            const char c = s[i++];
            if (c == '"') return true;
            if (c != '\\') { out += c; continue; }
            if (i >= s.size()) break;
            const char e = s[i++];
            switch (e)
            {
            case '"':  out += '"';  break;
            case '\\': out += '\\'; break;
            case '/':  out += '/';  break;
            case 'b':  out += '\b'; break;
            case 'f':  out += '\f'; break;
            case 'n':  out += '\n'; break;
            case 'r':  out += '\r'; break;
            case 't':  out += '\t'; break;
            case 'u':
            {
                unsigned cp = 0;
                if (!hex4(cp)) return fail("bad \\u escape");
                if (cp >= 0xD800 && cp <= 0xDBFF &&
                    i + 1 < s.size() && s[i] == '\\' && s[i + 1] == 'u')
                {
                    const size_t save = i;
                    i += 2;
                    unsigned lo = 0;
                    if (hex4(lo) && lo >= 0xDC00 && lo <= 0xDFFF)
                        cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                    else
                        i = save;
                }
                if (cp >= 0xD800 && cp <= 0xDFFF) cp = 0xFFFD;
                appendUtf8(cp, out);
                break;
            }
            default: return fail("bad escape");
            }
        }
        return fail("unterminated string");
    }

    // Numbers go through a classic-locale stream, never strtod/stof: those
    // read LC_NUMERIC, which the GUI's toolkit has already changed by the time
    // a result file is opened.  See the same guard on the writing side.
    bool parseNumber(JsonValue &out)
    {
        const size_t start = i;
        if (i < s.size() && (s[i] == '-' || s[i] == '+')) i++;
        while (i < s.size() &&
               ((s[i] >= '0' && s[i] <= '9') || s[i] == '.' ||
                s[i] == 'e' || s[i] == 'E' || s[i] == '+' || s[i] == '-'))
            i++;
        if (i == start) return fail("expected a value");

        std::istringstream is(s.substr(start, i - start));
        is.imbue(std::locale::classic());
        double v = 0.0;
        if (!(is >> v)) { i = start; return fail("malformed number"); }

        out.t = JsonValue::Type::Number;
        out.n = v;
        return true;
    }
};

} // namespace

JsonValue jsonParse(const std::string &text, std::string *error)
{
    JsonValue root;
    Parser    p(text);
    if (p.parse(root)) return root;
    if (error) *error = p.error;
    return JsonValue();
}
