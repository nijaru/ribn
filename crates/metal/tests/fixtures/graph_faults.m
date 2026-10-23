// QUALIFICATION ONLY. Public SDK swizzling; no production interposition/owners.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#import <objc/runtime.h>
#import <dispatch/dispatch.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <string.h>
#include <stdio.h>
#include <stdint.h>

@interface RibnQualControl : NSObject {
@public
    __weak id<MTLBuffer> buffers[5];
    size_t elements[5];
    atomic_size_t captured; // Release-publishes each immutable slot/extent.
    atomic_int terminal;   // 0 pending; 1 real GPU success; 2 real SDK failure.
    int mode;              // Immutable before publication to TLS/callback.
}
// Atomic accessor allows independent signal to race installation, without a
// nonatomic strong-reference read/write race. Signal only after gate is set.
@property(atomic, strong) id<MTLSharedEvent> gate;
@property(nonatomic, strong) dispatch_semaphore_t acknowledgmentGate;
@end
@implementation RibnQualControl
- (instancetype)init {
    self = [super init];
    if (self) {
        atomic_init(&captured, 0);
        atomic_init(&terminal, 0);
    }
    return self;
}
@end

// TLS owns only this weak slot, not the controller or its callback. Destruction
// before the one-shot launch therefore cannot leave a dangling armed pointer.
@interface RibnQualArmSlot : NSObject
@property(nonatomic, weak) RibnQualControl *control;
@end
@implementation RibnQualArmSlot
@end

static pthread_key_t armKey;
static _Thread_local bool forwardingInit;

typedef MPSGraphTensorData *(*QualInitIMP)(id __attribute__((ns_consumed)), SEL,
    id<MTLBuffer>, MPSShape *, MPSDataType) __attribute__((ns_returns_retained));
typedef NSArray<MPSGraphTensorData *> *(*QualRunIMP)(id, SEL, id<MTLCommandQueue>,
    NSArray<MPSGraphTensorData *> *, NSArray<MPSGraphTensorData *> *,
    MPSGraphExecutableExecutionDescriptor *);
static _Atomic(QualInitIMP) originalInit;
static _Atomic(QualRunIMP) originalRun;
static atomic_size_t calls;
size_t ribn_qual_calls(void) { return atomic_load_explicit(&calls, memory_order_relaxed); }

static NSArray<MPSGraphTensorData *> *qual_forward(QualRunIMP original, id receiver, SEL selector,
    id<MTLCommandQueue> queue, NSArray<MPSGraphTensorData *> *inputs,
    NSArray<MPSGraphTensorData *> *results, MPSGraphExecutableExecutionDescriptor *descriptor) {
    atomic_fetch_add_explicit(&calls, 1, memory_order_relaxed);
    return original(receiver, selector, queue, inputs, results, descriptor);
}

static RibnQualControl *qual_armed(void) {
    RibnQualArmSlot *slot = (__bridge RibnQualArmSlot *)pthread_getspecific(armKey);
    return slot.control;
}

static void qual_disarm(void) {
    RibnQualArmSlot *slot = (__bridge RibnQualArmSlot *)pthread_getspecific(armKey);
    slot.control = nil;
}

static void qual_tls_destroy(void *opaque) {
    @try {
        @autoreleasepool { (void)CFBridgingRelease(opaque); }
    } @catch (NSException *exception) { (void)exception; }
}

static void qual_require(BOOL valid, NSString *reason) {
    if (!valid) @throw [NSException exceptionWithName:@"RibnQualSetup"
                                            reason:reason userInfo:nil];
}

// Explicit init-family ABI: receiver consumed, result +1. A plain +0 C function
// cast to IMP would change ownership under ARC, biasing the lifetime experiment.
static MPSGraphTensorData *qual_init(id receiver __attribute__((ns_consumed)),
    SEL selector, id<MTLBuffer> buffer, MPSShape *shape, MPSDataType dataType)
    __attribute__((ns_returns_retained));
static MPSGraphTensorData *qual_init(id receiver __attribute__((ns_consumed)),
    SEL selector, id<MTLBuffer> buffer, MPSShape *shape, MPSDataType dataType) {
    QualInitIMP original = atomic_load_explicit(&originalInit, memory_order_acquire);
    if (forwardingInit) return original(receiver, selector, buffer, shape, dataType);
    RibnQualControl *control = qual_armed();
    @try {
        if (control) {
            size_t index = atomic_load_explicit(&control->captured, memory_order_relaxed);
            qual_require(index < 5 && dataType == MPSDataTypeBFloat16 && shape.count == 2,
                         @"expected at most five user BF16 matrices");
            size_t rows = shape[0].unsignedLongLongValue;
            size_t columns = shape[1].unsignedLongLongValue;
            qual_require(shape[0].longLongValue > 0 && shape[1].longLongValue > 0 &&
                         rows <= SIZE_MAX / sizeof(uint16_t) / columns,
                         @"invalid BF16 matrix extent");
            control->buffers[index] = buffer;
            control->elements[index] = rows * columns;
            atomic_store_explicit(&control->captured, index + 1, memory_order_release);
            if (control->mode == 1) {
                qual_disarm();
                @throw [NSException exceptionWithName:@"RibnQualPrelaunch"
                    reason:@"qualification exception before SDK tensor wrapping/run" userInfo:nil];
            }
        }
        // Do not capture SDK-internal recursive wrappers as user matrices.
        forwardingInit = true;
        @try {
            MPSGraphTensorData *data = original(receiver, selector, buffer, shape, dataType);
            if (control && !data) qual_disarm();
            return data;
        } @finally { forwardingInit = false; }
    } @catch (NSException *exception) {
        if (control) qual_disarm();
        @throw exception; // The production bridge, not this hook, owns this boundary.
    }
}

static NSArray<MPSGraphTensorData *> *qual_run(id receiver, SEL selector,
    id<MTLCommandQueue> queue, NSArray<MPSGraphTensorData *> *inputs,
    NSArray<MPSGraphTensorData *> *results,
    MPSGraphExecutableExecutionDescriptor *descriptor) {
    QualRunIMP original = atomic_load_explicit(&originalRun, memory_order_acquire);
    RibnQualControl *control = qual_armed();
    if (!control) return qual_forward(original, receiver, selector, queue, inputs, results, descriptor);
    // Consume BEFORE any SDK call, including one that throws. Later same-thread
    // executors and SDK-internal work cannot inherit the injection/capture.
    qual_disarm();
    size_t count = atomic_load_explicit(&control->captured, memory_order_acquire);
    qual_require((count == 3 || count == 5) && descriptor != nil &&
                 inputs.count == (count + 1) / 2 && results.count == (count - 1) / 2,
                 @"unexpected production matrix capture/input/output count");
    MPSGraphExecutableCompletionHandler callback = [descriptor.completionHandler copy];
    qual_require(callback != nil, @"production completion callback missing");
    descriptor.completionHandler = ^(NSArray<MPSGraphTensorData *> *values, NSError *failure) {
        // Only control and ORIGINAL callback are captured. In particular neither
        // descriptor, receiver, arrays, buffer, Flight nor model is captured.
        @try {
            @autoreleasepool {
                // Delay acknowledgment, not GPU execution: an unsignaled GPU
                // wait can hit the OS watchdog before the bridge's 30 s deadline.
                if (control->mode == 2)
                    (void)dispatch_semaphore_wait(control.acknowledgmentGate, DISPATCH_TIME_FOREVER);
                if (failure) fprintf(stderr, "real SDK completion error: %s\n", failure.description.UTF8String);
                NSError *reported = failure;
                if (control->mode == 3) {
                    reported = [NSError errorWithDomain:@"RibnQualInjectedTerminal" code:3
                        userInfo:@{NSLocalizedDescriptionKey:
                            @"qualification error after real SDK terminal completion"}];
                }
                callback(values, reported);
            }
            // Must follow the ORIGINAL callback's return, not merely GPU finish.
            atomic_store_explicit(&control->terminal, failure ? 2 : 1, memory_order_release);
        } @catch (NSException *exception) {
            (void)exception; // No false completion proof if original did not return.
        }
    };

    // Mode 2 withholds the real SDK callback until signal; it does not claim a
    // late GPU write. Mode 4 gates actual GPU work, returning an immediate
    // exception so the test can release the event before the OS watchdog.
    if (control->mode == 2)
        return qual_forward(original, receiver, selector, queue, inputs, results, descriptor);
    if (control->mode == 4) {
        id<MTLSharedEvent> gate = [queue.device newSharedEvent];
        qual_require(gate != nil, @"shared event allocation failed");
        gate.signaledValue = 0;
        [descriptor waitForEvent:gate value:1];
        control.gate = gate;
        (void)qual_forward(original, receiver, selector, queue, inputs, results, descriptor);
        @throw [NSException exceptionWithName:@"RibnQualAfterSubmit"
            reason:@"qualification exception after real asynchronous SDK submission" userInfo:nil];
    }
    return qual_forward(original, receiver, selector, queue, inputs, results, descriptor);
}

static BOOL qual_install(void) {
    // Do not load frameworks or reference Graph class symbols. Parent prepares
    // three graphs before arm. A premature arm fails without consuming the once.
    Class tensor = NSClassFromString(@"MPSGraphTensorData");
    Class executable = NSClassFromString(@"MPSGraphExecutable");
    Class descriptor = NSClassFromString(@"MPSGraphExecutableExecutionDescriptor");
    if (!tensor || !executable || !descriptor) return NO;
    static dispatch_once_t once;
    static BOOL installed;
    dispatch_once(&once, ^{
        SEL initSelector = @selector(initWithMTLBuffer:shape:dataType:);
        SEL runSelector = @selector(runAsyncWithMTLCommandQueue:inputsArray:resultsArray:executionDescriptor:);
        Method initMethod = class_getInstanceMethod(tensor, initSelector);
        Method runMethod = class_getInstanceMethod(executable, runSelector);
        if (!initMethod || !runMethod || method_getNumberOfArguments(initMethod) != 5 ||
            method_getNumberOfArguments(runMethod) != 6 ||
            !class_getInstanceMethod(descriptor, @selector(waitForEvent:value:)) ||
            !class_getInstanceMethod(descriptor, @selector(completionHandler)) ||
            !class_getInstanceMethod(descriptor, @selector(setCompletionHandler:)) ||
            pthread_key_create(&armKey, qual_tls_destroy) != 0) return;
        QualInitIMP init = (QualInitIMP)method_getImplementation(initMethod);
        QualRunIMP run = (QualRunIMP)method_getImplementation(runMethod);
        atomic_store_explicit(&originalInit, init, memory_order_release);
        atomic_store_explicit(&originalRun, run, memory_order_release);
        // replaceMethod also handles an inherited implementation without mutating
        // a superclass method. Do not assume a category exchange is sufficient.
        class_replaceMethod(tensor, initSelector, (IMP)qual_init, method_getTypeEncoding(initMethod));
        class_replaceMethod(executable, runSelector, (IMP)qual_run, method_getTypeEncoding(runMethod));
        installed = class_getMethodImplementation(tensor, initSelector) == (IMP)qual_init &&
                    class_getMethodImplementation(executable, runSelector) == (IMP)qual_run;
    });
    return installed;
}

void *ribn_qual_arm(int mode) {
    @try {
        @autoreleasepool {
            if (mode < 0 || mode > 4 || !qual_install()) return NULL;
            RibnQualArmSlot *slot = (__bridge RibnQualArmSlot *)pthread_getspecific(armKey);
            if (slot.control) return NULL; // Do not silently overwrite another arm.
            if (!slot) {
                slot = [RibnQualArmSlot new];
                if (!slot) return NULL;
                void *owned = (__bridge_retained void *)slot;
                if (pthread_setspecific(armKey, owned) != 0) {
                    qual_tls_destroy(owned);
                    return NULL;
                }
            }
            RibnQualControl *control = [RibnQualControl new];
            if (!control) return NULL;
            control->mode = mode;
            if (mode == 2) {
                control.acknowledgmentGate = dispatch_semaphore_create(0);
                if (!control.acknowledgmentGate) return NULL;
            }
            slot.control = control;
            return (__bridge_retained void *)control;
        }
    } @catch (NSException *exception) { (void)exception; return NULL; }
}

void ribn_qual_signal(void *opaque) {
    @try {
        @autoreleasepool {
            RibnQualControl *control = (__bridge RibnQualControl *)opaque;
            if (control.acknowledgmentGate) dispatch_semaphore_signal(control.acknowledgmentGate);
            id<MTLSharedEvent> gate = control.gate;
            if (gate) gate.signaledValue = 1;
        }
    } @catch (NSException *exception) { (void)exception; }
}

size_t ribn_qual_live(void *opaque) {
    @try {
        @autoreleasepool {
            RibnQualControl *control = (__bridge RibnQualControl *)opaque;
            if (!control) return 0;
            size_t count = atomic_load_explicit(&control->captured, memory_order_acquire);
            size_t live = 0;
            for (size_t i = 0; i < count; ++i) {
                id<MTLBuffer> buffer = control->buffers[i];
                if (buffer) ++live;
            }
            return live;
        }
    } @catch (NSException *exception) { (void)exception; return SIZE_MAX; }
}

int ribn_qual_finished(void *opaque) {
    @try {
        @autoreleasepool {
            RibnQualControl *control = (__bridge RibnQualControl *)opaque;
            return control && atomic_load_explicit(&control->terminal, memory_order_acquire) != 0;
        }
    } @catch (NSException *exception) { (void)exception; return 0; }
}

size_t ribn_qual_output(void *opaque, size_t index, uint16_t *dest, size_t requested) {
    @try {
        @autoreleasepool {
            RibnQualControl *control = (__bridge RibnQualControl *)opaque;
            if (!control || !dest ||
                atomic_load_explicit(&control->terminal, memory_order_acquire) != 1) return 0;
            size_t count = atomic_load_explicit(&control->captured, memory_order_acquire);
            if (count != 3 && count != 5) return 0;
            size_t outputs = (count - 1) / 2;
            if (index >= outputs) return 0;
            size_t slot = 1 + outputs + index;
            __attribute__((objc_precise_lifetime)) id<MTLBuffer> buffer = control->buffers[slot];
            if (!buffer || buffer.storageMode != MTLStorageModeShared) return 0;
            size_t available = buffer.length / sizeof(uint16_t);
            if (available > control->elements[slot]) available = control->elements[slot];
            size_t n = requested < available ? requested : available;
            const void *contents = buffer.contents;
            if (!contents || !n) return 0;
            memcpy(dest, contents, n * sizeof(uint16_t));
            return n;
        }
    } @catch (NSException *exception) { (void)exception; return 0; }
}

void ribn_qual_destroy(void *opaque) {
    @try {
        @autoreleasepool { (void)CFBridgingRelease(opaque); }
    } @catch (NSException *exception) { (void)exception; }
}
