#ifndef BACKEND_GATING_H
#define BACKEND_GATING_H

#include <bitset>
#include <clpeak.h>      // Benchmark enum and categoryOf()
#include <result_store.h> // Category enum

struct CliOptions; // forward decl

struct BackendGating {
  std::bitset<static_cast<size_t>(Benchmark::COUNT)> enabledTests;
  std::bitset<4> enabledCategories;

  BackendGating() {
    enabledTests.set();
    enabledCategories.set();
  }

  void copyFrom(const CliOptions &opts);

  bool isTestEnabled(Benchmark b) const {
    return enabledTests.test(static_cast<size_t>(b));
  }

  bool isCategoryEnabled(Category c) const {
    if (c == Category::Unknown) return false;
    return enabledCategories.test(static_cast<size_t>(c));
  }

  bool isAllowed(Benchmark b) const {
    return isCategoryEnabled(categoryOf(b)) && isTestEnabled(b);
  }

  bool isAllowedAs(Benchmark b, Category c) const {
    return isCategoryEnabled(c) && isTestEnabled(b);
  }

  void enableTest(Benchmark b) {
    enabledTests.set(static_cast<size_t>(b));
  }
  void disableAll() {
    enabledTests.reset();
  }
  void enableAll() {
    enabledTests.set();
    enabledCategories.set();
  }
};

#endif // BACKEND_GATING_H