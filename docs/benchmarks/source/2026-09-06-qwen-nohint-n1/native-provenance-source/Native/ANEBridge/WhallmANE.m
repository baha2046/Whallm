// Direct Apple Neural Engine bridge for the Qwen Prefill research path.
//
// The private API lifecycle and IOSurface request shape are adapted from
// maderix/ANE (MIT, Copyright (c) 2026 maderix).

#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <dlfcn.h>
#import <objc/message.h>
#import <objc/runtime.h>

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
  __strong id model;
  __strong id request;
  __strong NSString *temporaryDirectory;
  IOSurfaceRef input;
  IOSurfaceRef output;
  size_t inputElements;
  size_t outputElements;
  bool ownsTemporaryDirectory;
} WhallmANEProjection;

static Class gDescriptorClass;
static Class gModelClass;
static Class gRequestClass;
static Class gIOSurfaceClass;
static char gLastError[1024];

static void set_error(NSString *message) {
  const char *text = message.UTF8String ?: "unknown Apple Neural Engine error";
  snprintf(gLastError, sizeof(gLastError), "%s", text);
}

static void set_nserror(NSString *operation, NSError *error) {
  NSString *detail = error.localizedDescription ?: @"unknown error";
  set_error([NSString stringWithFormat:@"%@: %@", operation, detail]);
}

static bool has_class_method(Class cls, SEL selector) {
  return cls && class_getClassMethod(cls, selector) != NULL;
}

static bool has_instance_method(Class cls, SEL selector) {
  return cls && class_getInstanceMethod(cls, selector) != NULL;
}

static int initialize_framework(void) {
  static dispatch_once_t once;
  static int result = -1;
  dispatch_once(&once, ^{
    void *framework = dlopen(
      "/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine",
      RTLD_NOW | RTLD_LOCAL
    );
    if (!framework) {
      set_error(@"AppleNeuralEngine.framework could not be loaded");
      return;
    }

    gDescriptorClass = NSClassFromString(@"_ANEInMemoryModelDescriptor");
    gModelClass = NSClassFromString(@"_ANEInMemoryModel");
    gRequestClass = NSClassFromString(@"_ANERequest");
    gIOSurfaceClass = NSClassFromString(@"_ANEIOSurfaceObject");

    if (!has_class_method(
          gDescriptorClass,
          NSSelectorFromString(@"modelWithMILText:weights:optionsPlist:")) ||
        !has_class_method(
          gModelClass,
          NSSelectorFromString(@"inMemoryModelWithDescriptor:")) ||
        !has_instance_method(
          gModelClass,
          NSSelectorFromString(@"compileWithQoS:options:error:")) ||
        !has_instance_method(
          gModelClass,
          NSSelectorFromString(@"loadWithQoS:options:error:")) ||
        !has_instance_method(
          gModelClass,
          NSSelectorFromString(@"evaluateWithQoS:options:request:error:")) ||
        !has_instance_method(
          gModelClass,
          NSSelectorFromString(@"hexStringIdentifier")) ||
        !has_class_method(
          gRequestClass,
          NSSelectorFromString(
            @"requestWithInputs:inputIndices:outputs:outputIndices:"
             "weightsBuffer:perfStats:procedureIndex:")) ||
        !has_class_method(
          gIOSurfaceClass,
          NSSelectorFromString(@"objectWithIOSurface:"))) {
      set_error(@"AppleNeuralEngine.framework does not match the required interface");
      return;
    }
    result = 0;
  });
  return result;
}

static IOSurfaceRef make_surface(size_t bytes) {
  return IOSurfaceCreate((__bridge CFDictionaryRef)@{
    (id)kIOSurfaceWidth: @(bytes),
    (id)kIOSurfaceHeight: @1,
    (id)kIOSurfaceBytesPerElement: @1,
    (id)kIOSurfaceBytesPerRow: @(bytes),
    (id)kIOSurfaceAllocSize: @(bytes),
    (id)kIOSurfacePixelFormat: @0,
  });
}

static NSData *make_weight_blob(const uint16_t *weight, size_t elements) {
  size_t weightBytes = elements * sizeof(uint16_t);
  size_t totalBytes = 128 + weightBytes;
  uint8_t *blob = calloc(totalBytes, 1);
  if (!blob) return nil;
  blob[0] = 0x01;
  blob[4] = 0x02;
  blob[64] = 0xEF;
  blob[65] = 0xBE;
  blob[66] = 0xAD;
  blob[67] = 0xDE;
  blob[68] = 0x01;
  *(uint32_t *)(blob + 72) = (uint32_t)weightBytes;
  *(uint32_t *)(blob + 80) = 128;
  memcpy(blob + 128, weight, weightBytes);
  return [NSData dataWithBytesNoCopy:blob length:totalBytes freeWhenDone:YES];
}

static NSData *make_mil(int inputChannels, int outputChannels, int spatial) {
  NSString *text = [NSString stringWithFormat:
    @"program(1.3)\n"
     "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, "
     "{\"coremlc-version\", \"3505.4.1\"}, "
     "{\"coremltools-component-milinternal\", \"\"}, "
     "{\"coremltools-version\", \"9.0\"}})]\n"
     "{\n"
     "  func main<ios18>(tensor<fp16, [1, 1, %d, %d]> input) {\n"
     "    tensor<int32, [4]> input_perm = const()[name = string(\"input_perm\"), val = tensor<int32, [4]>([0, 3, 1, 2])];\n"
     "    tensor<fp16, [1, %d, 1, %d]> x = transpose(perm = input_perm, x = input)[name = string(\"input_transpose\")];\n"
     "    string pad_type = const()[name = string(\"pad_type\"), val = string(\"valid\")];\n"
     "    tensor<int32, [2]> strides = const()[name = string(\"strides\"), val = tensor<int32, [2]>([1, 1])];\n"
     "    tensor<int32, [4]> pad = const()[name = string(\"pad\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n"
     "    tensor<int32, [2]> dilations = const()[name = string(\"dilations\"), val = tensor<int32, [2]>([1, 1])];\n"
     "    int32 groups = const()[name = string(\"groups\"), val = int32(1)];\n"
     "    tensor<fp16, [%d, %d, 1, 1]> weight = const()[name = string(\"weight\"), "
     "val = tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(64)))];\n"
     "    tensor<fp16, [1, %d, 1, %d]> projected = conv(dilations = dilations, groups = groups, "
     "pad = pad, pad_type = pad_type, strides = strides, weight = weight, x = x)"
     "[name = string(\"projection\")];\n"
     "    tensor<int32, [4]> output_perm = const()[name = string(\"output_perm\"), val = tensor<int32, [4]>([0, 2, 3, 1])];\n"
     "    tensor<fp16, [1, 1, %d, %d]> y = transpose(perm = output_perm, x = projected)[name = string(\"output_transpose\")];\n"
     "  } -> (y);\n"
     "}\n",
    spatial, inputChannels, inputChannels, spatial,
    outputChannels, inputChannels, outputChannels, inputChannels,
    outputChannels, spatial, spatial, outputChannels
  ];
  return [text dataUsingEncoding:NSUTF8StringEncoding];
}

static void destroy_projection(WhallmANEProjection *projection) {
  if (!projection) return;
  @autoreleasepool {
    if (projection->model &&
        has_instance_method(
          gModelClass, NSSelectorFromString(@"unloadWithQoS:error:"))) {
      NSError *error = nil;
      ((BOOL (*)(id, SEL, unsigned int, NSError **))objc_msgSend)(
        projection->model,
        NSSelectorFromString(@"unloadWithQoS:error:"),
        21,
        &error
      );
    }
    if (projection->input) CFRelease(projection->input);
    if (projection->output) CFRelease(projection->output);
    if (projection->ownsTemporaryDirectory && projection->temporaryDirectory) {
      [[NSFileManager defaultManager]
        removeItemAtPath:projection->temporaryDirectory
        error:nil];
    }
    projection->model = nil;
    projection->request = nil;
    projection->temporaryDirectory = nil;
  }
  free(projection);
}

__attribute__((visibility("default")))
const char *whallm_ane_last_error(void) {
  return gLastError;
}

__attribute__((visibility("default")))
void *whallm_ane_projection_create(
  const uint16_t *weight,
  size_t weightElements,
  int inputChannels,
  int outputChannels,
  int spatial
) {
  @autoreleasepool {
    gLastError[0] = '\0';
    if (initialize_framework() != 0) return NULL;
    if (!weight || inputChannels <= 0 || outputChannels <= 0 || spatial <= 0 ||
        weightElements != (size_t)inputChannels * (size_t)outputChannels) {
      set_error(@"ANE projection shape is invalid");
      return NULL;
    }

    NSData *mil = make_mil(inputChannels, outputChannels, spatial);
    NSData *blob = make_weight_blob(weight, weightElements);
    if (!mil || !blob) {
      set_error(@"ANE projection allocation failed");
      return NULL;
    }
    NSDictionary *weights = @{
      @"@model_path/weights/weight.bin": @{
        @"offset": @0,
        @"data": blob,
      },
    };

    id descriptor = ((id (*)(Class, SEL, id, id, id))objc_msgSend)(
      gDescriptorClass,
      NSSelectorFromString(@"modelWithMILText:weights:optionsPlist:"),
      mil,
      weights,
      nil
    );
    if (!descriptor) {
      set_error(@"ANE model descriptor creation failed");
      return NULL;
    }
    id model = ((id (*)(Class, SEL, id))objc_msgSend)(
      gModelClass,
      NSSelectorFromString(@"inMemoryModelWithDescriptor:"),
      descriptor
    );
    if (!model) {
      set_error(@"ANE in-memory model creation failed");
      return NULL;
    }

    id identifier = ((id (*)(id, SEL))objc_msgSend)(
      model, NSSelectorFromString(@"hexStringIdentifier"));
    if (![identifier isKindOfClass:NSString.class]) {
      set_error(@"ANE model identifier is unavailable");
      return NULL;
    }

    NSString *temporaryDirectory =
      [NSTemporaryDirectory() stringByAppendingPathComponent:identifier];
    NSFileManager *files = NSFileManager.defaultManager;
    BOOL existed = [files fileExistsAtPath:temporaryDirectory];
    NSString *weightDirectory =
      [temporaryDirectory stringByAppendingPathComponent:@"weights"];
    NSError *error = nil;
    if (![files createDirectoryAtPath:weightDirectory
          withIntermediateDirectories:YES attributes:nil error:&error]) {
      set_nserror(@"ANE temporary directory creation failed", error);
      return NULL;
    }
    if (![mil writeToFile:
          [temporaryDirectory stringByAppendingPathComponent:@"model.mil"]
          options:NSDataWritingAtomic error:&error] ||
        ![blob writeToFile:
          [weightDirectory stringByAppendingPathComponent:@"weight.bin"]
          options:NSDataWritingAtomic error:&error]) {
      set_nserror(@"ANE model file creation failed", error);
      if (!existed) [files removeItemAtPath:temporaryDirectory error:nil];
      return NULL;
    }

    BOOL ok = ((BOOL (*)(id, SEL, unsigned int, id, NSError **))objc_msgSend)(
      model,
      NSSelectorFromString(@"compileWithQoS:options:error:"),
      21,
      @{},
      &error
    );
    if (!ok) {
      set_nserror(@"ANE compile failed", error);
      if (!existed) [files removeItemAtPath:temporaryDirectory error:nil];
      return NULL;
    }
    error = nil;
    ok = ((BOOL (*)(id, SEL, unsigned int, id, NSError **))objc_msgSend)(
      model,
      NSSelectorFromString(@"loadWithQoS:options:error:"),
      21,
      @{},
      &error
    );
    if (!ok) {
      set_nserror(@"ANE load failed", error);
      if (!existed) [files removeItemAtPath:temporaryDirectory error:nil];
      return NULL;
    }

    WhallmANEProjection *projection = calloc(1, sizeof(*projection));
    if (!projection) {
      set_error(@"ANE projection allocation failed");
      return NULL;
    }
    projection->model = model;
    projection->temporaryDirectory = temporaryDirectory;
    projection->ownsTemporaryDirectory = !existed;
    projection->inputElements = (size_t)inputChannels * (size_t)spatial;
    projection->outputElements = (size_t)outputChannels * (size_t)spatial;
    projection->input = make_surface(projection->inputElements * sizeof(uint16_t));
    projection->output = make_surface(projection->outputElements * sizeof(uint16_t));
    if (!projection->input || !projection->output) {
      set_error(@"ANE IOSurface allocation failed");
      destroy_projection(projection);
      return NULL;
    }

    id inputObject = ((id (*)(Class, SEL, IOSurfaceRef))objc_msgSend)(
      gIOSurfaceClass,
      NSSelectorFromString(@"objectWithIOSurface:"),
      projection->input
    );
    id outputObject = ((id (*)(Class, SEL, IOSurfaceRef))objc_msgSend)(
      gIOSurfaceClass,
      NSSelectorFromString(@"objectWithIOSurface:"),
      projection->output
    );
    projection->request =
      ((id (*)(Class, SEL, id, id, id, id, id, id, id))objc_msgSend)(
        gRequestClass,
        NSSelectorFromString(
          @"requestWithInputs:inputIndices:outputs:outputIndices:"
           "weightsBuffer:perfStats:procedureIndex:"),
        @[inputObject],
        @[@0],
        @[outputObject],
        @[@0],
        nil,
        nil,
        @0
      );
    if (!projection->request) {
      set_error(@"ANE request creation failed");
      destroy_projection(projection);
      return NULL;
    }
    return projection;
  }
}

__attribute__((visibility("default")))
int whallm_ane_projection_evaluate(
  void *opaqueProjection,
  const uint16_t *input,
  size_t inputElements,
  uint16_t *output,
  size_t outputElements
) {
  @autoreleasepool {
    gLastError[0] = '\0';
    WhallmANEProjection *projection = opaqueProjection;
    if (!projection || !input || !output ||
        inputElements != projection->inputElements ||
        outputElements != projection->outputElements) {
      set_error(@"ANE evaluate shape is invalid");
      return -1;
    }
    if (IOSurfaceLock(projection->input, 0, NULL) != kIOReturnSuccess) {
      set_error(@"ANE input IOSurface lock failed");
      return -1;
    }
    memcpy(
      IOSurfaceGetBaseAddress(projection->input),
      input,
      inputElements * sizeof(uint16_t)
    );
    IOSurfaceUnlock(projection->input, 0, NULL);

    NSError *error = nil;
    BOOL ok = ((BOOL (*)(id, SEL, unsigned int, id, id, NSError **))objc_msgSend)(
      projection->model,
      NSSelectorFromString(@"evaluateWithQoS:options:request:error:"),
      21,
      @{},
      projection->request,
      &error
    );
    if (!ok) {
      set_nserror(@"ANE evaluate failed", error);
      return -1;
    }
    if (IOSurfaceLock(
          projection->output, kIOSurfaceLockReadOnly, NULL) != kIOReturnSuccess) {
      set_error(@"ANE output IOSurface lock failed");
      return -1;
    }
    memcpy(
      output,
      IOSurfaceGetBaseAddress(projection->output),
      outputElements * sizeof(uint16_t)
    );
    IOSurfaceUnlock(projection->output, kIOSurfaceLockReadOnly, NULL);
    return 0;
  }
}

__attribute__((visibility("default")))
void whallm_ane_projection_destroy(void *opaqueProjection) {
  destroy_projection(opaqueProjection);
}
