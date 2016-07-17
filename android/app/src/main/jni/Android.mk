LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)
LOCAL_MODULE            := clpeak
LOCAL_CFLAGS            += -fPIC -O2 -DANDROID_LOGGER -DUSE_STUB_OPENCL
LOCAL_CXXFLAGS          += -fPIC -O2 -fexceptions -std=c++11
LOCAL_STATIC_LIBRARIES  := OpenCL
LOCAL_LDLIBS            := -ldl -llog -latomic

ROOT_FOLDER             := ../../../../..
SRC_FOLDER              := $(ROOT_FOLDER)/src

LOCAL_C_INCLUDES +=                             \
    $(LOCAL_PATH)/libopencl-stub/include/       \
    $(LOCAL_PATH)/$(ROOT_FOLDER)/include/       \
    $(LOCAL_PATH)/$(SRC_FOLDER)/kernels/

LOCAL_SRC_FILES :=                              \
    $(SRC_FOLDER)/common.cpp                    \
    $(SRC_FOLDER)/clpeak.cpp                    \
    $(SRC_FOLDER)/global_bandwidth.cpp          \
    $(SRC_FOLDER)/compute_sp.cpp                \
    $(SRC_FOLDER)/compute_dp.cpp                \
    $(SRC_FOLDER)/compute_integer.cpp           \
    $(SRC_FOLDER)/transfer_bandwidth.cpp        \
    $(SRC_FOLDER)/kernel_latency.cpp            \
    $(SRC_FOLDER)/entry_android.cpp             \
    $(SRC_FOLDER)/options.cpp                   \
    $(SRC_FOLDER)/logger_android.cpp

include $(BUILD_SHARED_LIBRARY)

include $(LOCAL_PATH)/libopencl-stub/Android.mk

