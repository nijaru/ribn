// BF16 FFN Graph bridge. Every possible enqueue has callback-owned retirement.
// Native-only loads never initialize MPS or Graph. No Objective-C exception may
// unwind across this C ABI.
#import <Availability.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <dispatch/dispatch.h>
#include <dlfcn.h>
#include <stdint.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <string.h>

static void graph_error(char *error, size_t capacity, const char *message) {
    if (!error || !capacity) return;
    if (!message) message = "unknown MPSGraph error";
    size_t length = strlen(message);
    size_t n = length < capacity ? length : capacity - 1;
    // Do not return a truncated UTF-8 code point.
    if (n < length) while (n && ((unsigned char)message[n] & 0xc0) == 0x80) --n;
    memcpy(error, message, n);
    error[n] = '\0';
}


#if defined(__MAC_26_0) && \
    __MAC_OS_X_VERSION_MAX_ALLOWED >= __MAC_26_0 && \
    __has_include(<MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>)
#define RIBN_GRAPH_COMPILED 1
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

// Resolve classes dynamically rather than introducing mandatory framework load
// commands: native x86_64 deployments may predate MPS/Graph. Keep the libraries
// loaded for process lifetime because executables and callbacks can outlive calls.
static BOOL graph_load_frameworks(void) {
    static dispatch_once_t once;
    static BOOL loaded;
    dispatch_once(&once, ^{
        @try {
        void *mps = dlopen("/System/Library/Frameworks/MetalPerformanceShaders.framework/MetalPerformanceShaders",
                           RTLD_LAZY | RTLD_LOCAL);
        void *graph = mps ? dlopen("/System/Library/Frameworks/MetalPerformanceShadersGraph.framework/MetalPerformanceShadersGraph",
                                  RTLD_LAZY | RTLD_LOCAL) : NULL;
        loaded = graph != NULL;
        for (NSString *name in @[@"MPSGraph", @"MPSGraphDevice", @"MPSGraphCompilationDescriptor",
                                  @"MPSGraphTensorData", @"MPSGraphShapedType",
                                  @"MPSGraphExecutableExecutionDescriptor"])
            loaded = loaded && NSClassFromString(name) != Nil;
        } @catch (NSException *exception) {
            (void)exception;
            loaded = NO;
        }
    });
    return loaded;
}

// The caller's executor Root mutex serializes ALL graph contexts and native
// commands, including run/retire/destroy. Only SDK completion callbacks run outside it.
// A per-context pending slot does not permit parallel work across contexts.
@class RibnGraphFlight;
API_AVAILABLE(macos(26.0))
@interface RibnGraphContext : NSObject
@property(nonatomic, strong) id<MTLDevice> device;
@property(nonatomic, strong) id<MTLCommandQueue> queue;
@property(nonatomic, strong) MPSGraphExecutable *executable;
@property(nonatomic, copy) NSArray<MPSGraphTensor *> *placeholders;
@property(nonatomic, copy) NSArray<MPSGraphTensor *> *targets;
@property(nonatomic, copy) NSArray<NSNumber *> *feedOrder;
@property(nonatomic, copy) NSArray<NSNumber *> *targetOrder;
@property(nonatomic, copy) NSArray<NSNumber *> *inputShape, *weightShape, *outputShape;
@property(nonatomic) size_t inputBytes, weightBytes, outputBytes;
@property(nonatomic) NSUInteger outputs;
@property(nonatomic, strong) RibnGraphFlight *pending;
@property(nonatomic) BOOL faulted;
@end
@implementation RibnGraphContext
@end

// Exactly one SDK callback writes failure/results, then publishes finished with
// release ordering. Readers MUST acquire finished before touching those fields.
// The descriptor block captures only Completion, never Flight or Context: no cycle.
API_AVAILABLE(macos(26.0))
@interface RibnGraphCompletion : NSObject {
@public
    atomic_bool finished;
    dispatch_semaphore_t done;
    NSError *failure;
    NSArray<MPSGraphTensorData *> *results;
}
@end
@implementation RibnGraphCompletion
- (instancetype)init {
    self = [super init];
    if (self) atomic_init(&finished, false);
    return self;
}
@end

// Context owns this envelope BEFORE graph wrapping/submission. Every object an
// uncertain enqueue may access stays here until explicit terminal observation.
API_AVAILABLE(macos(26.0))
@interface RibnGraphFlight : NSObject
@property(nonatomic, strong) id<MTLDevice> device;
@property(nonatomic, strong) id<MTLCommandQueue> queue;
@property(nonatomic, strong) MPSGraphExecutable *executable;
@property(nonatomic, strong) NSArray<id<MTLBuffer>> *buffers;
@property(nonatomic, strong) NSMutableArray<MPSGraphTensorData *> *logicalFeeds;
@property(nonatomic, strong) NSMutableArray<MPSGraphTensorData *> *logicalResults;
@property(nonatomic, strong) NSMutableArray<MPSGraphTensorData *> *feeds;
@property(nonatomic, strong) NSMutableArray<MPSGraphTensorData *> *boundResults;
@property(nonatomic, strong) NSArray<MPSGraphTensorData *> *returnedResults;
@property(nonatomic, strong) MPSGraphExecutableExecutionDescriptor *descriptor;
@property(nonatomic, strong) RibnGraphCompletion *completion;
@property(nonatomic, strong) NSArray<NSNumber *> *outputShape;
@property(nonatomic) NSUInteger outputs;
@property(nonatomic) BOOL possibleLaunch;
@end
@implementation RibnGraphFlight
@end

API_AVAILABLE(macos(26.0))
static void graph_require(BOOL valid, NSString *reason) {
    if (!valid) @throw [NSException exceptionWithName:@"RibnGraphPreparation"
                                            reason:reason userInfo:nil];
}

API_AVAILABLE(macos(26.0))
static size_t graph_bytes(size_t a, size_t b) {
    graph_require(a && b && a <= (size_t)NSIntegerMax && b <= (size_t)NSIntegerMax &&
                  a <= SIZE_MAX / b / 2, @"invalid or overflowing BF16 matrix dimensions");
    return a * b * 2;
}

// Both feeds and targets may be reordered by compilation. Match identity, not
// names or dictionary iteration order, and verify the complete permutation.
API_AVAILABLE(macos(26.0))
static NSArray<NSNumber *> *graph_order(NSArray<MPSGraphTensor *> *compiled,
                                       NSArray<MPSGraphTensor *> *logical) {
    graph_require(compiled && compiled.count == logical.count, @"compiled tensor count mismatch");
    NSMutableArray<NSNumber *> *order = [NSMutableArray array];
    NSUInteger seen = 0;
    for (MPSGraphTensor *tensor in compiled) {
        NSUInteger index = [logical indexOfObjectIdenticalTo:tensor];
        graph_require(index != NSNotFound, @"unknown compiled tensor identity");
        NSUInteger bit = (NSUInteger)1 << index;
        graph_require(!(seen & bit), @"duplicate compiled tensor identity");
        seen |= bit;
        [order addObject:@(index)];
    }
    return [order copy];
}

// All matrix buffers start at offset zero with tightly packed row-major BF16.
// Only Gate/Up pairs at 16/64 rows and Down singles at 64 rows are prepared.
// Returns a +1 context; preparation never submits GPU execution or retains weights.
API_AVAILABLE(macos(26.0))
void *graph_create(void *device, void *queue, size_t rows,
                              size_t columns, size_t inner, size_t outputs,
                              char *error, size_t error_capacity) {
    graph_error(error, error_capacity, "");
    @try {
        @autoreleasepool {
            graph_require(device && queue, @"null device or queue");
            graph_require((outputs == 2 && (rows == 16 || rows == 64)) ||
                          (outputs == 1 && rows == 64), @"unsupported FFN Graph shape/output count");
            RibnGraphContext *ctx = [RibnGraphContext new];
            graph_require(ctx != nil, @"context allocation failed");
            ctx.inputBytes = graph_bytes(rows, inner);
            ctx.weightBytes = graph_bytes(columns, inner);
            ctx.outputBytes = graph_bytes(rows, columns);
            ctx.device = (__bridge id<MTLDevice>)device;
            ctx.queue = (__bridge id<MTLCommandQueue>)queue;
            graph_require([ctx.device conformsToProtocol:@protocol(MTLDevice)] &&
                          [ctx.queue conformsToProtocol:@protocol(MTLCommandQueue)] &&
                          ctx.queue.device == ctx.device, @"queue/device identity mismatch");
            graph_require(ctx.inputBytes <= ctx.device.maxBufferLength &&
                          ctx.weightBytes <= ctx.device.maxBufferLength &&
                          ctx.outputBytes <= ctx.device.maxBufferLength, @"matrix exceeds device buffer limit");
            ctx.outputs = outputs;
            ctx.inputShape = @[@(rows), @(inner)];
            ctx.weightShape = @[@(columns), @(inner)];
            ctx.outputShape = @[@(rows), @(columns)];
            MPSGraph *graph = [NSClassFromString(@"MPSGraph") new];
            graph_require(graph != nil, @"graph allocation failed");
            graph.options = MPSGraphOptionsSynchronizeResults;
            MPSGraphTensor *a = [graph placeholderWithShape:ctx.inputShape
                                                 dataType:MPSDataTypeBFloat16 name:@"A"];
            graph_require(a != nil, @"input placeholder creation failed");
            NSMutableArray<MPSGraphTensor *> *placeholders = [NSMutableArray arrayWithObject:a];
            NSMutableArray<MPSGraphTensor *> *targets = [NSMutableArray array];
            MPSGraphShapedType *atype = [[NSClassFromString(@"MPSGraphShapedType") alloc] initWithShape:ctx.inputShape
                                                                      dataType:MPSDataTypeBFloat16];
            MPSGraphShapedType *wtype = [[NSClassFromString(@"MPSGraphShapedType") alloc] initWithShape:ctx.weightShape
                                                                      dataType:MPSDataTypeBFloat16];
            graph_require(atype && wtype, @"feed type allocation failed");
            NSMutableDictionary *feeds = [NSMutableDictionary dictionaryWithObject:atype forKey:a];
            for (NSUInteger i = 0; i < outputs; ++i) {
                MPSGraphTensor *w = [graph placeholderWithShape:ctx.weightShape
                    dataType:MPSDataTypeBFloat16 name:i == 0 ? @"W0" : @"W1"];
                graph_require(w != nil, @"weight placeholder creation failed");
                MPSGraphTensor *wt = [graph transposeTensor:w dimension:0 withDimension:1 name:nil];
                graph_require(wt != nil, @"transpose creation failed");
                MPSGraphTensor *y = [graph matrixMultiplicationWithPrimaryTensor:a secondaryTensor:wt
                                                                         name:i == 0 ? @"Y0" : @"Y1"];
                graph_require(y && y.dataType == MPSDataTypeBFloat16 &&
                              [y.shape isEqual:ctx.outputShape], @"graph result shape/type mismatch");
                [placeholders addObject:w];
                [targets addObject:y];
                feeds[w] = wtype;
            }
            MPSGraphCompilationDescriptor *cd = [NSClassFromString(@"MPSGraphCompilationDescriptor") new];
            MPSGraphDevice *graphDevice = [NSClassFromString(@"MPSGraphDevice") deviceWithMTLDevice:ctx.device];
            graph_require(cd && graphDevice, @"compilation descriptor/device allocation failed");
            cd.optimizationLevel = MPSGraphOptimizationLevel0;
            cd.reducedPrecisionFastMath = MPSGraphReducedPrecisionFastMathNone;
            cd.waitForCompilationCompletion = YES;
            ctx.executable = [graph compileWithDevice:graphDevice feeds:feeds targetTensors:targets
                                    targetOperations:nil compilationDescriptor:cd];
            graph_require(ctx.executable != nil, @"MPSGraph compilation failed");
            ctx.executable.options = MPSGraphOptionsSynchronizeResults;
            ctx.placeholders = placeholders;
            ctx.targets = targets;
            ctx.feedOrder = graph_order(ctx.executable.feedTensors, ctx.placeholders);
            ctx.targetOrder = graph_order(ctx.executable.targetTensors, ctx.targets);
            return (__bridge_retained void *)ctx;
        }
    } @catch (NSException *exception) {
        // Copy before the exception's autoreleased diagnostic storage disappears.
        @try {
            @autoreleasepool {
                graph_error(error, error_capacity, exception.description.UTF8String);
            }
        } @catch (NSException *reportException) {
            (void)reportException;
            graph_error(error, error_capacity, "MPSGraph preparation raised NSException");
        }
        return NULL;
    }
}

API_AVAILABLE(macos(26.0))
static BOOL graph_buffer(id buffer, id<MTLDevice> device, size_t bytes) {
    return buffer && [buffer conformsToProtocol:@protocol(MTLBuffer)] &&
           [(id<MTLBuffer>)buffer device] == device && [(id<MTLBuffer>)buffer length] >= bytes;
}

API_AVAILABLE(macos(26.0))
static BOOL graph_results(id values, NSUInteger count, NSArray<NSNumber *> *shape) {
    if (![values isKindOfClass:[NSArray class]] || [values count] != count) return NO;
    for (id value in values) {
        if (![value isKindOfClass:[NSClassFromString(@"MPSGraphTensorData") class]]) return NO;
        MPSGraphTensorData *data = value;
        if (data.dataType != MPSDataTypeBFloat16 || ![data.shape isEqual:shape]) return NO;
    }
    return YES;
}

// Reporting must not replace a recoverable result with an exception across C.
API_AVAILABLE(macos(26.0))
static void graph_report(id diagnostic, char *error, size_t capacity, const char *fallback) {
    @try {
        @autoreleasepool {
            graph_error(error, capacity, diagnostic ? [diagnostic description].UTF8String : fallback);
        }
    } @catch (NSException *exception) {
        (void)exception;
        graph_error(error, capacity, fallback);
    }
}

API_AVAILABLE(macos(26.0))
static BOOL graph_finished(RibnGraphFlight *flight) {
    RibnGraphCompletion *completion = flight.completion;
    return completion && atomic_load_explicit(&completion->finished, memory_order_acquire);
}

// No waiting, queue fences or reference-count inference. A failed compute is just
// as physically retireable as a successful one once its SDK callback is terminal.
API_AVAILABLE(macos(26.0))
static BOOL graph_retire(RibnGraphContext *ctx) {
    RibnGraphFlight *flight = ctx.pending;
    if (!flight) return YES;
    if (flight.possibleLaunch && !graph_finished(flight)) return NO;
    // The callback never touches results after its release store.
    RibnGraphCompletion *completion = flight.completion;
    if (completion) {
        completion->results = nil;
        completion->failure = nil;
    }
    ctx.pending = nil;
    return YES;
}

// Called inside the C entry point's exception boundary and autorelease pool.
// installed distinguishes this call's prelaunch envelope from an older pending
// flight that a refused call must never discard. possibleLaunch is set BEFORE
// invoking the asynchronous SDK method, including when that method throws.
API_AVAILABLE(macos(26.0))
static int graph_run(RibnGraphContext *ctx, void *input, void *weight0, void *weight1,
                     void *output0, void *output1, BOOL *installed, BOOL *possibleLaunch,
                     char *error, size_t capacity) {
    if (!ctx) {
        graph_error(error, capacity, "null context");
        return 0;
    }
    if (ctx.faulted || ctx.pending) {
        graph_error(error, capacity, "graph context faulted or awaiting retirement; run refused");
        return -2;
    }
    if (!input || !weight0 || !output0 || (ctx.outputs == 2 && (!weight1 || !output1))) {
        graph_error(error, capacity, "null required matrix buffer");
        return 0;
    }
    NSArray<id<MTLBuffer>> *buffers = ctx.outputs == 2
        ? @[(__bridge id)input, (__bridge id)weight0, (__bridge id)weight1,
            (__bridge id)output0, (__bridge id)output1]
        : @[(__bridge id)input, (__bridge id)weight0, (__bridge id)output0];
    BOOL valid = [ctx.device conformsToProtocol:@protocol(MTLDevice)] &&
                 [ctx.queue conformsToProtocol:@protocol(MTLCommandQueue)] &&
                 ctx.queue.device == ctx.device &&
                 graph_buffer(buffers[0], ctx.device, ctx.inputBytes);
    for (NSUInteger i = 0; i < ctx.outputs; ++i) {
        NSUInteger oi = 1 + ctx.outputs + i;
        valid = valid && graph_buffer(buffers[1 + i], ctx.device, ctx.weightBytes) &&
                         graph_buffer(buffers[oi], ctx.device, ctx.outputBytes);
        for (NSUInteger j = 0; j < oi; ++j) valid = valid && buffers[oi] != buffers[j];
    }
    if (!valid) {
        graph_error(error, capacity, "buffer length/device or output alias preflight failed");
        return 0;
    }

    RibnGraphFlight *flight = [RibnGraphFlight new];
    graph_require(flight != nil, @"flight allocation failed before launch");
    flight.device = ctx.device;
    flight.queue = ctx.queue;
    flight.executable = ctx.executable;
    flight.buffers = buffers;
    flight.outputShape = ctx.outputShape;
    flight.outputs = ctx.outputs;
    *installed = YES;
    ctx.pending = flight;
    // PRELAUNCH INVARIANT: all calls below until possibleLaunch is set only
    // construct host wrappers/descriptors. Any new enqueue-capable call MUST
    // move the marker before that call; clean-rejection cleanup depends on it.

    flight.logicalFeeds = [NSMutableArray array];
    flight.logicalResults = [NSMutableArray array];
    flight.feeds = [NSMutableArray array];
    flight.boundResults = [NSMutableArray array];
    graph_require(flight.logicalFeeds && flight.logicalResults && flight.feeds && flight.boundResults,
                  @"tensor array allocation failed before launch");
    for (NSUInteger i = 0; i < buffers.count; ++i) {
        NSArray<NSNumber *> *shape = i == 0 ? ctx.inputShape :
            (i <= ctx.outputs ? ctx.weightShape : ctx.outputShape);
        MPSGraphTensorData *data = [[NSClassFromString(@"MPSGraphTensorData") alloc] initWithMTLBuffer:buffers[i]
            shape:shape dataType:MPSDataTypeBFloat16];
        graph_require(data != nil, @"BF16 tensor wrapping failed before launch");
        [(i <= ctx.outputs ? flight.logicalFeeds : flight.logicalResults) addObject:data];
    }
    for (NSNumber *index in ctx.feedOrder)
        [flight.feeds addObject:flight.logicalFeeds[index.unsignedIntegerValue]];
    for (NSNumber *index in ctx.targetOrder)
        [flight.boundResults addObject:flight.logicalResults[index.unsignedIntegerValue]];

    RibnGraphCompletion *completion = [RibnGraphCompletion new];
    flight.completion = completion;
    flight.descriptor = [NSClassFromString(@"MPSGraphExecutableExecutionDescriptor") new];
    graph_require(completion && flight.descriptor &&
                  (completion->done = dispatch_semaphore_create(0)),
                  @"completion allocation failed before launch");
    flight.descriptor.waitUntilCompleted = NO;
    flight.descriptor.completionHandler = ^(NSArray<MPSGraphTensorData *> *values, NSError *failure) {
        @try {
            @autoreleasepool {
                completion->failure = failure;
                completion->results = values;
            }
            // Publish only after stores AND callback-pool cleanup succeeded. A
            // callback exception leaves finished false and the full flight held.
            atomic_store_explicit(&completion->finished, true, memory_order_release);
            dispatch_semaphore_signal(completion->done);
        } @catch (NSException *exception) {
            (void)exception;
            // No abort, retry or completion claim: permanent quarantine is safe.
        }
    };
    // This marker precedes the FIRST call that can enqueue any Graph work.
    *possibleLaunch = YES;
    flight.possibleLaunch = YES;
    flight.returnedResults = [flight.executable runAsyncWithMTLCommandQueue:flight.queue
        inputsArray:flight.feeds resultsArray:flight.boundResults executionDescriptor:flight.descriptor];
    // Bound only the wait AFTER the async CPU method returns; a stuck SDK CPU
    // method cannot be interrupted safely by this bridge. Never wait in retire.
    if (!graph_finished(flight)) {
        (void)dispatch_semaphore_wait(completion->done,
            dispatch_time(DISPATCH_TIME_NOW, 30 * (int64_t)NSEC_PER_SEC));
    }
    if (!graph_finished(flight)) {
        ctx.faulted = YES;
        graph_error(error, capacity, "terminal callback absent after 30 seconds; flight retained");
        return -2;
    }
    if (completion->failure) {
        ctx.faulted = YES;
        graph_report(completion->failure, error, capacity, "terminal MPSGraph execution error");
        return -1;
    }
    if (!graph_results(flight.returnedResults, flight.outputs, flight.outputShape) ||
        !graph_results(completion->results, flight.outputs, flight.outputShape)) {
        ctx.faulted = YES;
        graph_error(error, capacity, "terminal graph results have invalid count/BF16 shape/type");
        return -1;
    }
    return 1;
}

// Borrowed arguments must be live Objective-C objects, not arbitrary addresses.
// Inputs are ready/immutable, outputs exclusively caller-owned and unpublished
// until status 1. No output may alias an input/other output, including backing
// aliases (the caller must rule out aliases between distinct MTLBuffer objects).
// weight1/output1 are ignored for a single-output context. No Rust callbacks.
//  1: successful TERMINAL; 0: clean PRELAUNCH rejection (this call enqueued nothing).
// -1: terminal failure; -2: uncertain enqueue or failed cleanup. Both fault the context
// permanently and retain its flight until retire; neither publishes outputs.
API_AVAILABLE(macos(26.0))
int graph_run_entry(void *context, void *input, void *weight0, void *weight1,
                         void *output0, void *output1, char *error, size_t error_capacity) {
    graph_error(error, error_capacity, "");
    RibnGraphContext *ctx = nil;
    BOOL installed = NO, possibleLaunch = NO;
    @try {
        int status;
        @autoreleasepool {
            ctx = (__bridge RibnGraphContext *)context;
            status = graph_run(ctx, input, weight0, weight1, output0, output1,
                               &installed, &possibleLaunch, error, error_capacity);
        }
        // Keep pending through all per-run pool cleanup, even on success. The
        // success path uses the same terminal proof and result-owner cleanup.
        @autoreleasepool {
            if (status == 1) (void)graph_retire(ctx);
            else if (status == 0 && installed) {
                ctx.pending = nil;
                if (ctx.pending) {
                    ctx.faulted = YES;
                    graph_error(error, error_capacity, "prelaunch cleanup retained its flight");
                    return -2;
                }
            }
        }
        return status;
    } @catch (NSException *exception) {
        // Pending was installed before any possible enqueue, so unwinding local
        // owners never drops the device's access envelope. Unknown means retained.
        BOOL clean = !possibleLaunch;
        @try {
            if (installed && !possibleLaunch) {
                ctx.pending = nil;
                clean = ctx.pending == nil;
            }
        } @catch (NSException *cleanupException) {
            (void)cleanupException;
            clean = NO;
        }
        if (!clean) {
            @try { ctx.faulted = YES; }
            @catch (NSException *faultException) { (void)faultException; }
        }
        graph_report(exception, error, error_capacity,
                     clean ? "MPSGraph clean prelaunch exception"
                           : "MPSGraph enqueue or cleanup uncertain; flight retained");
        return clean ? 0 : -2;
    }
}

// 1: no pending access remains (including terminal failed compute); 0: unknown
// or unfinished, retaining everything. Idempotent, nonblocking, never reopens run.
API_AVAILABLE(macos(26.0))
int graph_retire_entry(void *context, char *error, size_t error_capacity) {
    graph_error(error, error_capacity, "");
    @try {
        @autoreleasepool {
            RibnGraphContext *ctx = (__bridge RibnGraphContext *)context;
            if (graph_retire(ctx)) return 1;
            graph_error(error, error_capacity, "graph callback unfinished or unknown; flight retained");
            return 0;
        }
    } @catch (NSException *exception) {
        graph_report(exception, error, error_capacity, "MPSGraph retirement exception");
        return 0;
    }
}

// Consume create's +1 ONLY after terminal retirement. A 0 leaves that +1 untouched;
// the caller must retain the ENTIRE execution owner, not just this C context.
// Explicit Root retirement is the main path; this is defensive quarantine.
API_AVAILABLE(macos(26.0))
int graph_destroy(void *context) {
    BOOL consumed = NO;
    @try {
        @autoreleasepool {
            if (!context) return 1;
            __attribute__((objc_precise_lifetime)) RibnGraphContext *ctx =
                (__bridge RibnGraphContext *)context;
            if (!graph_retire(ctx)) return 0;
            // ctx pins a nonfinal reference while consuming the caller's +1.
            CFRelease((CFTypeRef)context);
            consumed = YES;
        }
        return 1;
    } @catch (NSException *exception) {
        (void)exception;
        // A terminal destructor exception cannot resurrect a consumed +1.
        return consumed ? 1 : 0;
    }
}


#else
#define RIBN_GRAPH_COMPILED 0
#endif

int ribn_graph_available(void) {
    @try {
        @autoreleasepool {
#if RIBN_GRAPH_COMPILED
            if (@available(macOS 26.0, *)) return graph_load_frameworks() ? 1 : 0;
#endif
            return 0;
        }
    } @catch (NSException *exception) {
        (void)exception;
        return 0;
    }
}

void *ribn_graph_create(void *device, void *queue, size_t rows, size_t columns,
                        size_t inner, size_t outputs, char *error, size_t capacity) {
#if RIBN_GRAPH_COMPILED
    if (@available(macOS 26.0, *)) {
        if (ribn_graph_available() == 1)
            return graph_create(device, queue, rows, columns, inner, outputs, error, capacity);
    }
#else
    (void)device; (void)queue; (void)rows; (void)columns; (void)inner; (void)outputs;
#endif
    graph_error(error, capacity, "BF16 Graph SDK/runtime/framework support unavailable");
    return NULL;
}

int ribn_graph_run(void *context, void *input, void *w0, void *w1, void *o0, void *o1,
                   char *error, size_t capacity) {
#if RIBN_GRAPH_COMPILED
    if (@available(macOS 26.0, *))
        return graph_run_entry(context, input, w0, w1, o0, o1, error, capacity);
#else
    (void)input; (void)w0; (void)w1; (void)o0; (void)o1;
#endif
    graph_error(error, capacity, "BF16 Graph unavailable");
    return context ? -2 : 0;
}

int ribn_graph_retire(void *context, char *error, size_t capacity) {
#if RIBN_GRAPH_COMPILED
    if (@available(macOS 26.0, *)) return graph_retire_entry(context, error, capacity);
#endif
    graph_error(error, capacity, context ? "BF16 Graph unavailable" : "");
    return context ? 0 : 1;
}

int ribn_graph_destroy(void *context) {
#if RIBN_GRAPH_COMPILED
    if (@available(macOS 26.0, *)) return graph_destroy(context);
#endif
    return context ? 0 : 1;
}
