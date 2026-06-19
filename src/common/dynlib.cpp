#include <common/dynlib.h>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <dlfcn.h>
#endif

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

void dynClose(void *lib)
{
  if (!lib)
    return;
#if defined(_WIN32)
  FreeLibrary(reinterpret_cast<HMODULE>(lib));
#else
  dlclose(lib);
#endif
}

} // namespace clpeak
