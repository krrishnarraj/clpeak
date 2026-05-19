#ifndef CLPEAK_IOS_BRIDGE_H
#define CLPEAK_IOS_BRIDGE_H

#ifdef __cplusplus
extern "C" {
#endif

typedef void (*ClpeakIOSMetricCallback)(void *context,
                                        const char *backend,
                                        const char *platform,
                                        const char *device,
                                        const char *driver,
                                        const char *category,
                                        const char *test,
                                        const char *display,
                                        const char *metric,
                                        const char *unit,
                                        float value,
                                        const char *status,
                                        const char *reason);

typedef void (*ClpeakIOSDeviceCallback)(void *context,
                                        const char *backend,
                                        const char *platform,
                                        const char *device,
                                        const char *driver,
                                        const char *propsJson,
                                        int platformIndex,
                                        int deviceIndex);

typedef struct ClpeakIOSCallbacks {
  ClpeakIOSMetricCallback metric;
  ClpeakIOSDeviceCallback device;
} ClpeakIOSCallbacks;

char *clpeak_ios_copy_backend_catalog_json(void);
void clpeak_ios_free_string(char *value);
const char *clpeak_ios_version(void);

int clpeak_ios_launch(int argc,
                      const char **argv,
                      ClpeakIOSCallbacks callbacks,
                      void *context);

#ifdef __cplusplus
}
#endif

#endif // CLPEAK_IOS_BRIDGE_H
