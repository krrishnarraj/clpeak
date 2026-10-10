// NSProcessInfo.thermalState for thermal.cpp, which is C++ (thermal.h).
#import <Foundation/Foundation.h>

int clpeakIosThermalState()
{
  @autoreleasepool
  {
    return (int)[NSProcessInfo processInfo].thermalState;
  }
}
