#ifndef CLPEAK_COREML_INTERNAL_H
#define CLPEAK_COREML_INTERNAL_H

// Objective-C helpers shared by the .mm files of this backend.  Never
// included from a .cpp: the public surface for the tests is coreml_session.h
// and include/coreml/coreml_peak.h.

#import <CoreML/CoreML.h>

#include <coreml/coreml_peak.h>

// Which kind of silicon a Core ML compute device object is.
CoremlDeviceKind coremlKindOf(id<MLComputeDeviceProtocol> d);

// The MLModelConfiguration that prefers `dev`: the compute-units request
// and, for a GPU, the Metal device to pin.
MLModelConfiguration *coremlConfigurationFor(const coreml_device_info_t &dev);

#endif // CLPEAK_COREML_INTERNAL_H
