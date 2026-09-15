#include <mach/mach_time.h>
#include <os/log.h>
#include <os/signpost.h>
#include <stdint.h>

// A timestamp inside the signpost call bounds Python-to-trace alignment error.
uint64_t probe_event(const char *label) {
    static os_log_t log;
    static mach_timebase_info_data_t timebase;
    if (!log) {
        log = os_log_create("org.whallm.research.decode", OS_LOG_CATEGORY_POINTS_OF_INTEREST);
        mach_timebase_info(&timebase);
    }
    uint64_t now = mach_absolute_time() * timebase.numer / timebase.denom;
    os_signpost_event_emit(log, OS_SIGNPOST_ID_EXCLUSIVE, "DecodeProbe",
                          "%{public}s clock=%{public}llu", label, now);
    return now;
}
