// NSProcessInfo.thermalState for thermal.cpp, which is C++ (thermal.h).
#import <Foundation/Foundation.h>

int xpupeakIosThermalState()
{
  @autoreleasepool
  {
    return (int)[NSProcessInfo processInfo].thermalState;
  }
}
