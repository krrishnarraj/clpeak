#ifndef CLPEAK_JSON_WRITER_H
#define CLPEAK_JSON_WRITER_H

#include <cstdint>
#include <iomanip>
#include <locale>
#include <ostream>
#include <sstream>
#include <string>

#include <common/common.h>  // jsonEscape

// ── Streaming JSON emitter ─────────────────────────────────────────────────
//
// Indent-tracking writer shared by the result document (run_document.cpp),
// the device inventory (inventory.cpp) and the run-log sidecar (run_log.cpp),
// so one shape has one serializer.  Nothing is built in memory: the caller
// walks its own structures and the writer punctuates.
//
// Three layouts, all valid JSON:
//
//   Pretty   -- one key per line, two-space indent.  The document is read,
//               diffed and pasted into bug reports, so it is laid out for
//               people.
//   Inline   -- one line, ", " and ": " separators.  A record that reads as
//               one line of a transcript (a log entry) starts one with
//               beginObjectInline(); everything up to its endObject() stays
//               on that line.
//   Compact  -- no whitespace at all: interchange payloads (the GUI catalog).
//
// Every number goes through the classic locale.  The GUI hosts this code
// inside toolkits that call setlocale(LC_ALL, "") -- GTK's gtk_init does --
// and a comma decimal separator produces a file that is not JSON at all.
class JsonWriter
{
public:
    explicit JsonWriter(std::ostream &o, bool compact = false)
        : out(o), compact(compact)
    {
        out.imbue(std::locale::classic());
    }

    void beginObject()               { punctuate(); out << "{"; open(); }
    void beginObject(const char *k)  { key(k);      out << "{"; open(); }
    void endObject()                 { close();     out << "}"; }
    void beginArray()                { punctuate(); out << "["; open(); }
    void beginArray(const char *k)   { key(k);      out << "["; open(); }
    void endArray()                  { close();     out << "]"; }

    // A record kept on one line.  Ends at its matching endObject().
    void beginObjectInline()
    {
        punctuate();
        if (inlineFrom < 0) inlineFrom = depth;
        out << "{";
        open();
    }

    void num(const char *k, double v)         { key(k); out << fmtNum(v); }
    void uint(const char *k, std::uint64_t v) { key(k); out << v; }
    void integer(const char *k, long long v)  { key(k); out << v; }
    void boolean(const char *k, bool v)       { key(k); out << (v ? "true" : "false"); }
    void str(const char *k, const std::string &v)
    {
        key(k);
        out << "\"" << jsonEscape(v) << "\"";
    }
    // Optional string: absent rather than empty, so a file carries only facts.
    void strIf(const char *k, const std::string &v) { if (!v.empty()) str(k, v); }

    // A bare string element of an array.
    void rawString(const std::string &v)
    {
        punctuate();
        out << "\"" << jsonEscape(v) << "\"";
    }

    // Seven significant digits: enough to round-trip a float's worth of
    // precision (measurements are floats), enough to keep a six-digit GFLOPS
    // reading whole, and -- unlike the fixed four decimals this replaced --
    // it does not flatten the ONNX numeric-error readings, which are parts
    // per million and used to be written as 0.0000.
    static std::string fmtNum(double v)
    {
        std::ostringstream ss;
        ss.imbue(std::locale::classic());
        ss << std::setprecision(7) << v;
        return ss.str();
    }

private:
    bool oneLine() const { return compact || inlineFrom >= 0; }

    void open()  { depth++; fresh = true; }
    void close()
    {
        depth--;
        if (!oneLine()) newline();
        if (inlineFrom == depth) inlineFrom = -1;
        fresh = false;
    }
    void newline()
    {
        out << "\n" << std::string(static_cast<size_t>(depth) * 2, ' ');
    }
    void punctuate()
    {
        if (!fresh) out << (compact ? "," : oneLine() ? ", " : ",");
        // The top-level value starts the text; only nested values start a
        // line of their own.
        if (!oneLine() && depth > 0) newline();
        fresh = false;
    }
    void key(const char *k)
    {
        punctuate();
        out << "\"" << k << (compact ? "\":" : "\": ");
    }

    std::ostream &out;
    bool compact;
    int  depth = 0;
    bool fresh = true;     // nothing written at this depth yet -> no leading comma
    int  inlineFrom = -1;  // depth at which an inline record began, or -1
};

#endif  // CLPEAK_JSON_WRITER_H
