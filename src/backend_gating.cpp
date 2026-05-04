#include <options.h>
#include <backend_gating.h>

void BackendGating::copyFrom(const CliOptions &opts) {
  enabledTests      = opts.enabledTests;
  enabledCategories = opts.enabledCategories;
}