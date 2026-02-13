#include <bcc/helpers.h>

#define MAX_RULES 24
#define MAX_PREFIX_LEN 24
#define COMM_LEN 16
#define RINGBUF_PAGES __RINGBUF_PAGES__

struct rule_t {
    u32 enabled;
    u32 event_type; // 0 open
    u32 prefix_len;
    u32 comm_len;
    u32 pid;
    u32 tid;
    u32 uid;
    char prefix[MAX_PREFIX_LEN];
    char comm[COMM_LEN];
};

struct data_t {
    u64 ts;
    u32 pid;
    u32 tid;
    u32 uid;
    u32 flags;
    char comm[COMM_LEN];
    char filename[256];
};

BPF_ARRAY(rules, struct rule_t, MAX_RULES);
BPF_ARRAY(rule_count, u32, 1);
BPF_RINGBUF_OUTPUT(events, RINGBUF_PAGES);

/* No early return/break in loop so clang can unroll (fixed trip count COMM_LEN). */
static __inline int comm_matches(struct data_t *data, const struct rule_t *rule) {
    int match = 1;
    if (rule->comm_len == 0) {
        return 1;
    }
#pragma unroll
    for (int i = 0; i < COMM_LEN; i++) {
        if (i < rule->comm_len) {
            if (data->comm[i] != rule->comm[i] || data->comm[i] == 0) {
                match = 0;
            }
        }
    }
    return match;
}

/* No early return/break in loop so clang can unroll (fixed trip count MAX_PREFIX_LEN). */
static __inline int prefix_matches(struct data_t *data, const struct rule_t *rule) {
    int match = 1;
    if (rule->prefix_len == 0) {
        return 1;
    }
    // If filename is empty, match if prefix is "/" (root matches everything including empty)
    if (data->filename[0] == 0) {
        return (rule->prefix_len == 1 && rule->prefix[0] == '/');
    }
#pragma unroll
    for (int i = 0; i < MAX_PREFIX_LEN; i++) {
        if (i < rule->prefix_len) {
            if (data->filename[i] != rule->prefix[i] || data->filename[i] == 0) {
                match = 0;
            }
        }
    }
    return match;
}

/* No break/continue/return in loop so clang can unroll (fixed trip count MAX_RULES). */
static __inline int rule_allows(struct data_t *data) {
    u32 idx = 0;
    u32 *count = rule_count.lookup(&idx);
    if (!count || *count == 0) {
        return 1;
    }
    int allowed = 0;
#pragma unroll
    for (int i = 0; i < MAX_RULES; i++) {
        if (i < *count) {
            u32 key = i;
            struct rule_t *rule = rules.lookup(&key);
            if (rule && rule->enabled && rule->event_type == 0 &&
                (rule->pid == 0 || rule->pid == data->pid) &&
                (rule->tid == 0 || rule->tid == data->tid) &&
                (rule->uid == 0 || rule->uid == data->uid) &&
                comm_matches(data, rule) && prefix_matches(data, rule)) {
                allowed = 1;
            }
        }
    }
    return allowed;
}

TRACEPOINT_PROBE(syscalls, sys_enter_open) {
    struct data_t data = {};
    u64 pid_tgid = bpf_get_current_pid_tgid();
    data.pid = pid_tgid >> 32;
    data.tid = pid_tgid;
    data.uid = bpf_get_current_uid_gid();
    data.flags = args->flags;
    data.ts = bpf_ktime_get_ns();
    bpf_get_current_comm(&data.comm, sizeof(data.comm));
    
    if (args->filename != 0) {
        bpf_probe_read_user_str(&data.filename, sizeof(data.filename), args->filename);
    }
    
    if (!rule_allows(&data)) {
        return 0;
    }
    
    events.ringbuf_output(&data, sizeof(data), 0);
    return 0;
}

TRACEPOINT_PROBE(syscalls, sys_enter_openat) {
    struct data_t data = {};
    u64 pid_tgid = bpf_get_current_pid_tgid();
    data.pid = pid_tgid >> 32;
    data.tid = pid_tgid;
    data.uid = bpf_get_current_uid_gid();
    data.flags = args->flags;
    data.ts = bpf_ktime_get_ns();
    bpf_get_current_comm(&data.comm, sizeof(data.comm));
    
    if (args->filename != 0) {
        bpf_probe_read_user_str(&data.filename, sizeof(data.filename), args->filename);
    }
    
    if (!rule_allows(&data)) {
        return 0;
    }
    
    events.ringbuf_output(&data, sizeof(data), 0);
    return 0;
}
