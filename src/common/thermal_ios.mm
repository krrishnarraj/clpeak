// NSProcessInfo.thermalState for thermal.cpp, which is C++ (thermal.h).
#import <Foundation/Foundation.h>

int xpuPeakIosThermalState()
{
  @autoreleasepool
  {
    return (int)[NSProcessInfo processInfo].thermalState;
  }
}
