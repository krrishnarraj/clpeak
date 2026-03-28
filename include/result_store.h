#ifndef RESULT_STORE_H
#define RESULT_STORE_H

#include <string>
#include <vector>
#include <map>

// A single benchmark measurement, fully qualified by its provenance.
struct ResultEntry {
    std::string platform;
    std::string device;
    std::string driver;
    std::string test;    // XML group tag, e.g. "global_memory_bandwidth"
    std::string metric;  // sub-tag or derived name, e.g. "float4"
    std::string unit;    // e.g. "gbps", "gflops", "giops", "us"
    float value;

    // Composite key used for baseline lookup in --compare mode.
    // Omits driver so the comparison is meaningful across driver updates.
    std::string key() const
    {
        return platform + "/" + device + "/" + test + "/" + metric;
    }
};

typedef std::vector<ResultEntry>      ResultStore;
typedef std::map<std::string, float>  BaselineMap;

// Build a fast lookup map from a ResultStore (key -> value).
BaselineMap buildBaselineMap(const ResultStore &store);

// Serialize results to a JSON file (array-of-objects, one object per line for
// easy line-by-line parsing by loadJson).
void saveJson(const ResultStore &store, const std::string &filename);

// Serialize results to a CSV file with a header row.
void saveCsv(const ResultStore &store, const std::string &filename);

// Parse a JSON file written by saveJson.  Returns an empty store and prints a
// warning on stderr if the file cannot be opened or contains no valid entries.
ResultStore loadJson(const std::string &filename);

#endif // RESULT_STORE_H
