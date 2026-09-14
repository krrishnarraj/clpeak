#include <common/coreml_cache.h>
#include <common/common.h>

#import <Foundation/Foundation.h>

namespace clpeak {

void purgeCoreMLCompileCache()
{
  @autoreleasepool
  {
    NSFileManager *fm = NSFileManager.defaultManager;
    NSArray<NSString *> *caches =
        NSSearchPathForDirectoriesInDomains(NSCachesDirectory, NSUserDomainMask, YES);
    if (caches.count == 0)
      return;
    // E5RT files the cache under the process name for a plain executable
    // and under the bundle identifier for an app; clear whichever exists.
    NSMutableArray<NSString *> *names = [NSMutableArray new];
    NSString *proc = NSProcessInfo.processInfo.processName;
    if (proc.length)
      [names addObject:proc];
    NSString *bundleId = NSBundle.mainBundle.bundleIdentifier;
    if (bundleId.length && ![names containsObject:bundleId])
      [names addObject:bundleId];
    for (NSString *name in names)
    {
      NSString *dir = [[caches[0] stringByAppendingPathComponent:name]
          stringByAppendingPathComponent:@"com.apple.e5rt.e5bundlecache"];
      BOOL isDir = NO;
      if (![fm fileExistsAtPath:dir isDirectory:&isDir] || !isDir)
        continue;
      NSError *err = nil;
      if ([fm removeItemAtPath:dir error:&err])
        CLPEAK_VLOG("coreml: removed the compile cache at %s\n", dir.UTF8String);
      else
        CLPEAK_VLOG("coreml: could not remove %s: %s\n", dir.UTF8String,
                    err.localizedDescription.UTF8String);
    }
  }
}

} // namespace clpeak
