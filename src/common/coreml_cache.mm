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
    // (~/Library/Caches/clpeak) and under the bundle identifier for an app
    // (~/Library/Caches/kr.clpeak for the unsandboxed macOS app); clear
    // whichever exists.  Inside an app container -- iOS, or a sandboxed
    // macOS app, whose Caches directory is private to the app -- the cache
    // may also sit at the Caches root, and only then is that root ours to
    // clear: outside a container the root-level com.apple.e5rt.e5bundlecache
    // belongs to every other process on the machine.
    NSMutableArray<NSString *> *dirs = [NSMutableArray new];
    NSString *proc = NSProcessInfo.processInfo.processName;
    if (proc.length)
      [dirs addObject:[caches[0] stringByAppendingPathComponent:proc]];
    NSString *bundleId = NSBundle.mainBundle.bundleIdentifier;
    if (bundleId.length && ![bundleId isEqualToString:proc])
      [dirs addObject:[caches[0] stringByAppendingPathComponent:bundleId]];
    if ([caches[0] rangeOfString:@"/Containers/"].location != NSNotFound)
      [dirs addObject:caches[0]];
    for (NSString *base in dirs)
    {
      NSString *dir = [base stringByAppendingPathComponent:@"com.apple.e5rt.e5bundlecache"];
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
