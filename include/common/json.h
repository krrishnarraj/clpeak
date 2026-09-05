#ifndef CLPEAK_JSON_H
#define CLPEAK_JSON_H

#include <map>
#include <string>
#include <vector>

// ── Minimal JSON DOM ───────────────────────────────────────────────────────
//
// Just enough to read a clpeak result document back (run_document.cpp) and
// nothing more.  Hand-rolled on purpose: it is the only parser clpeak needs,
// it is smaller than the XML/CSV line scanners it replaces, and it lets the
// classic-locale rule for numbers live in one place -- the GUI hosts this
// code inside toolkits that call setlocale(LC_ALL, "") (GTK's gtk_init does),
// and a comma decimal separator would silently mis-read every value.
//
// Writing is not here: the writers stream JSON text directly, so nothing has
// to build a document in memory just to serialize it.

class JsonValue
{
public:
    enum class Type { Null, Bool, Number, String, Array, Object };

    JsonValue() = default;

    Type type() const { return t; }
    bool isNull()   const { return t == Type::Null; }
    bool isObject() const { return t == Type::Object; }
    bool isArray()  const { return t == Type::Array; }

    // Scalar access.  Reading a value as the wrong type yields the default
    // rather than throwing: a malformed field should cost that one field, not
    // the whole file.
    const std::string &asString() const { return s; }
    double             asNumber() const { return n; }
    bool               asBool()   const { return b; }

    // Object access.  `find` returns nullptr when this is not an object or
    // the key is absent, which is also how an absent optional field reads --
    // the writers omit empty ones.
    const JsonValue *find(const std::string &key) const;
    bool has(const std::string &key) const { return find(key) != nullptr; }

    std::string str (const std::string &key, const std::string &def = "") const;
    double      num (const std::string &key, double def = 0.0) const;
    bool        flag(const std::string &key, bool def = false) const;

    // Array access.  Empty for a non-array, so `for (const auto &v : x.items())`
    // is safe on anything.
    const std::vector<JsonValue> &items() const;

    // Populated by the parser.
    Type                              t = Type::Null;
    bool                              b = false;
    double                            n = 0.0;
    std::string                       s;
    std::vector<JsonValue>            arr;
    std::map<std::string, JsonValue>  obj;
};

// Parse a whole document.  On failure returns a Null value and, when `error`
// is non-null, fills it with a human-readable message including the offset.
JsonValue jsonParse(const std::string &text, std::string *error = nullptr);

#endif  // CLPEAK_JSON_H
