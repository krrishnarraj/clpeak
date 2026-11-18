#include <common/path_scrub.h>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <system_error>

namespace xpupeak {
namespace {

// A byte that continues a path segment's name, so a match ends only where
// the name does: "/Users/kr" is not a prefix of "/Users/kristen/x".  Bytes
// past ASCII are a UTF-8 name's.
bool nameChar(char c)
{
    const unsigned char u = static_cast<unsigned char>(c);
    return std::isalnum(u) || c == '.' || c == '_' || c == '-' || u >= 0x80;
}

bool isSep(char c) { return c == '/' || c == '\\'; }

char lower(char c) { return static_cast<char>(std::tolower(static_cast<unsigned char>(c))); }

// Whether `text` at `at` reads `want`.  Windows paths come with either
// separator and in any case; elsewhere a path is compared as written.
bool matchPath(const std::string &text, size_t at, const std::string &want)
{
    if (at + want.size() > text.size()) return false;
    for (size_t k = 0; k < want.size(); k++)
    {
        const char a = text[at + k], b = want[k];
#ifdef _WIN32
        if (isSep(a) && isSep(b)) continue;
        if (lower(a) != lower(b)) return false;
#else
        if (a != b) return false;
#endif
    }
    return true;
}

// A marker ("/Users/") in any case and with either separator: a well-known
// directory is spelled the same everywhere a path can come from.
bool matchMarker(const std::string &text, size_t at, const char *marker)
{
    size_t k = 0;
    for (; marker[k]; k++)
    {
        if (at + k >= text.size()) return false;
        const char a = text[at + k], b = marker[k];
        if (b == '/' ? !isSep(a) : lower(a) != lower(b)) return false;
    }
    return true;
}

// The end of the path segment starting at `at`.
size_t segmentEnd(const std::string &text, size_t at)
{
    while (at < text.size() && !isSep(text[at]) && text[at] != '"' && text[at] != '\'' &&
           text[at] != ' ' && text[at] != ':' && text[at] != ',' && text[at] != ')' &&
           text[at] != ']' && text[at] != '\n')
        at++;
    return at;
}

bool isUuid(const std::string &s)
{
    if (s.size() != 36) return false;
    for (size_t k = 0; k < s.size(); k++)
    {
        const bool dash = k == 8 || k == 13 || k == 18 || k == 23;
        if (dash ? s[k] != '-' : !std::isxdigit(static_cast<unsigned char>(s[k]))) return false;
    }
    return true;
}

struct Prefix {
    std::string path;
    const char *word;
};

// A temporary directory every user shares names nobody.
bool sharedTemp(const std::string &dir)
{
    return dir == "/tmp" || dir == "/var/tmp" || dir == "/private/tmp" ||
           dir == "/private/var/tmp" || dir == "/data/local/tmp";
}

std::vector<Prefix> makePrefixes()
{
    std::vector<Prefix> out;
    auto add = [&out](std::string path, const char *word) {
        while (path.size() > 1 && isSep(path.back())) path.pop_back();
        // A root ("/", "C:") would turn every path into the word.
        if (path.size() < 4) return;
        for (const Prefix &p : out)
            if (p.path == path) return;
        out.push_back({path, word});
    };
    // As given, and as the OS resolves it: macOS hands out /var/folders/...
    // and libraries print /private/var/folders/....
    auto addBoth = [&add](const std::string &path, const char *word) {
        if (path.empty()) return;
        add(path, word);
        std::error_code ec;
        const std::filesystem::path real = std::filesystem::weakly_canonical(path, ec);
        if (!ec) add(real.string(), word);
    };

    std::error_code ec;
    std::string temp = std::filesystem::temp_directory_path(ec).string();
    while (temp.size() > 1 && isSep(temp.back())) temp.pop_back();
    if (!ec && !sharedTemp(temp)) addBoth(temp, "$TMPDIR");
#ifdef _WIN32
    if (const char *home = std::getenv("USERPROFILE")) addBoth(home, "~");
#endif
    if (const char *home = std::getenv("HOME")) addBoth(home, "~");

    // Longest first: a temporary directory inside the home is the temporary
    // directory, and /private/var/... goes before /var/....
    std::stable_sort(out.begin(), out.end(), [](const Prefix &a, const Prefix &b) {
        return a.path.size() > b.path.size();
    });
    return out;
}

const std::vector<Prefix> &prefixes()
{
    static const std::vector<Prefix> p = makePrefixes();
    return p;
}

// A home directory's name that names nobody: the shared and template ones,
// and one this pass already wrote.
bool sharedHome(const std::string &name)
{
    std::string n;
    for (char c : name) n += lower(c);
    return n.empty() || n == "shared" || n == "public" || n == "default" ||
           n == "default user" || n == "all users" || name[0] == '<' || name[0] == '$';
}

} // namespace

std::string scrubPaths(const std::string &text)
{
    // Most lines hold no path at all: leave them as they are, uncopied.
    if (text.find('/') == std::string::npos && text.find('\\') == std::string::npos)
        return text;

    const std::vector<Prefix> &own = prefixes();
    std::string out;
    out.reserve(text.size());
    size_t i = 0;
    while (i < text.size())
    {
        bool done = false;
        for (const Prefix &p : own)
            if (matchPath(text, i, p.path) &&
                (i + p.path.size() == text.size() || !nameChar(text[i + p.path.size()])))
            {
                out += p.word;
                i += p.path.size();
                done = true;
                break;
            }
        if (done) continue;

        if (isSep(text[i]))
        {
            // Another user's home: its name, and nothing after it.
            for (const char *marker : {"/Users/", "/home/"})
            {
                const size_t len = std::char_traits<char>::length(marker);
                if (!matchMarker(text, i, marker)) continue;
                const size_t from = i + len, to = segmentEnd(text, from);
                const std::string name = text.substr(from, to - from);
                if (sharedHome(name)) break;
                out += text.substr(i, len) + "<user>";
                i = to;
                done = true;
                break;
            }
            if (done) continue;

            // macOS's per-user folders: /var/folders/<two letters>/<id>.
            if (matchMarker(text, i, "/var/folders/"))
            {
                const size_t a = i + 13, aEnd = segmentEnd(text, a);
                if (aEnd < text.size() && isSep(text[aEnd]) && aEnd > a)
                {
                    const size_t bEnd = segmentEnd(text, aEnd + 1);
                    if (bEnd > aEnd + 1)
                    {
                        out += text.substr(i, 13) + "<user>";
                        i = bEnd;
                        continue;
                    }
                }
            }

            // Android's install directory: /data/app/~~<id>/<package>-<id>.
            if (matchMarker(text, i, "/data/app/~~"))
            {
                size_t to = segmentEnd(text, i + 10);
                if (to < text.size() && isSep(text[to]))
                {
                    const size_t next = segmentEnd(text, to + 1);
                    if (text.substr(to + 1, next - to - 1).find("==") != std::string::npos)
                        to = next;
                }
                out += text.substr(i, 10) + "<install>";
                i = to;
                continue;
            }

            // iOS's containers: .../Application/<UUID>, .../AppGroup/<UUID>.
            for (const char *marker : {"/Application/", "/AppGroup/"})
            {
                const size_t len = std::char_traits<char>::length(marker);
                if (!matchMarker(text, i, marker)) continue;
                const size_t from = i + len, to = segmentEnd(text, from);
                if (!isUuid(text.substr(from, to - from))) break;
                out += text.substr(i, len) + "<install>";
                i = to;
                done = true;
                break;
            }
            if (done) continue;
        }
        out += text[i++];
    }
    return out;
}

std::vector<std::string> scrubArgv(const std::vector<std::string> &argv)
{
    std::vector<std::string> out;
    out.reserve(argv.size());
    for (size_t i = 0; i < argv.size(); i++)
    {
        if (i == 0)
        {
            const size_t sep = argv[0].find_last_of("/\\");
            out.push_back(sep == std::string::npos ? argv[0] : argv[0].substr(sep + 1));
        }
        else
            out.push_back(scrubPaths(argv[i]));
    }
    return out;
}

} // namespace xpupeak
