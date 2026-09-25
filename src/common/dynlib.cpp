#include <common/dynlib.h>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include <cctype>
#include <filesystem>
#include <string>
#include <system_error>

namespace clpeak {

void *dynOpen(std::initializer_list<const char *> names)
{
  for (const char *n : names)
  {
    if (!n)
      continue;
#if defined(_WIN32)
    HMODULE h = LoadLibraryA(n);
    if (h)
      return reinterpret_cast<void *>(h);
#else
    void *h = dlopen(n, RTLD_NOW | RTLD_LOCAL);
    if (h)
      return h;
#endif
  }
  return nullptr;
}

void *dynSym(void *lib, const char *name)
{
  if (!lib || !name)
    return nullptr;
#if defined(_WIN32)
  return reinterpret_cast<void *>(
      GetProcAddress(reinterpret_cast<HMODULE>(lib), name));
#else
  return dlsym(lib, name);
#endif
}

std::string absoluteModulePath(const char *name)
{
  if (!name || !*name || name[0] == '@')
    return name ? name : std::string();
  const std::string s(name);
  if (s.find_first_of("/\\") == std::string::npos)
    return s;
  std::error_code ec;
  const std::string abs = std::filesystem::absolute(s, ec).string();
  return ec ? s : abs;
}

bool sameModulePath(const std::string &a, const std::string &b)
{
  std::string x = absoluteModulePath(a.c_str());
  std::string y = absoluteModulePath(b.c_str());
#if defined(_WIN32)
  for (std::string *s : {&x, &y})
    for (char &c : *s)
      c = c == '/' ? '\\' : static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
#endif
  if (x == y)
    return true;
  // One file under two names: a symlink (Homebrew's lib/ into its Cellar),
  // or the path a default search resolved against the one a person picked.
  std::error_code ec;
  return !x.empty() && !y.empty() && std::filesystem::equivalent(x, y, ec) && !ec;
}

} // namespace clpeak
