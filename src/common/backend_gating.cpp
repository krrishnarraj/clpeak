#include <common/options.h>
#include <common/backend_gating.h>

void BackendGating::copyFrom(const CliOptions &opts) {
  enabledTests      = opts.enabledTests;
  enabledCategories = opts.enabledCategories;
}
