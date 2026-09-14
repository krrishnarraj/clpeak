#ifndef CLPEAK_BACKEND_REGISTRY_H
#define CLPEAK_BACKEND_REGISTRY_H

#include <memory>
#include <vector>
#include <common/benchmark_enums.h>
#include <common/inventory.h>

class Peak;

// One backend as the run loops see it: its identity, and how to enumerate
// its devices and how to construct it.
struct BackendEntry
{
    Backend id;
    BackendInventory (*enumerate)();
    std::unique_ptr<Peak> (*create)();
};

// Every backend in this build, in Backend order -- the one order --help,
// --list-devices, the GUI catalog, a run and the result document share.
// Built once in src/registry/backend_registry.cpp and sorted by id there,
// so the enum alone decides the order and a backend cannot be registered
// out of place.  The CLI and the GUI bridge iterate this and nothing else.
const std::vector<BackendEntry> &backendRegistry();

#endif // CLPEAK_BACKEND_REGISTRY_H
