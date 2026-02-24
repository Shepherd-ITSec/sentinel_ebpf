#include <bcc/helpers.h>

#define MAX_RULES 24
#define MAX_PREFIX_LEN 24
#define COMM_LEN 16
#define RINGBUF_PAGES __RINGBUF_PAGES__

struct rule_t {
    u32 enabled;
    u32 event_id; // 0 wildcard, otherwise curated event id
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
    u32 event_id;
    u32 pid;
    u32 tid;
    u32 uid;
    u32 flags;
    s64 arg0;
    s64 arg1;
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
            if (rule && rule->enabled &&
                (rule->event_id == 0 || rule->event_id == data->event_id) &&
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

static __inline int fill_common(struct data_t *data, u32 event_id) {
    u64 pid_tgid = bpf_get_current_pid_tgid();
    data->pid = pid_tgid >> 32;
    data->tid = pid_tgid;
    data->uid = bpf_get_current_uid_gid();
    data->ts = bpf_ktime_get_ns();
    data->event_id = event_id;
    bpf_get_current_comm(&data->comm, sizeof(data->comm));
    return 0;
}

static __inline int submit_if_allowed(struct data_t *data) {
    if (!rule_allows(data)) {
        return 0;
    }
    events.ringbuf_output(data, sizeof(*data), 0);
    return 0;
}

static __inline int is_enriched_syscall(long id) {
    return id == 2 || id == 3 || id == 41 || id == 42 || id == 59 ||
           id == 82 || id == 87 || id == 90 || id == 91 || id == 92 ||
           id == 93 || id == 257 || id == 263 || id == 264 || id == 437;
}

TRACEPOINT_PROBE(syscalls, sys_enter_open) {
    struct data_t data = {};
    fill_common(&data, 2);
    data.flags = args->flags;
    if (args->filename != 0) {
        bpf_probe_read_user_str(&data.filename, sizeof(data.filename), args->filename);
    }
    data.arg0 = args->flags;
    return submit_if_allowed(&data);
}

TRACEPOINT_PROBE(syscalls, sys_enter_openat) {
    struct data_t data = {};
    fill_common(&data, 257);
    data.flags = args->flags;
    if (args->filename != 0) {
        bpf_probe_read_user_str(&data.filename, sizeof(data.filename), args->filename);
    }
    data.arg0 = args->dfd;
    data.arg1 = args->flags;
    return submit_if_allowed(&data);
}

TRACEPOINT_PROBE(syscalls, sys_enter_openat2) {
    struct data_t data = {};
    fill_common(&data, 437);
    if (args->filename != 0) {
        bpf_probe_read_user_str(&data.filename, sizeof(data.filename), args->filename);
    }
    data.arg0 = args->dfd;
    data.arg1 = args->usize;
    return submit_if_allowed(&data);
}

TRACEPOINT_PROBE(syscalls, sys_enter_execve) {
    struct data_t data = {};
    fill_common(&data, 59);
    if (args->filename != 0) {
        bpf_probe_read_user_str(&data.filename, sizeof(data.filename), args->filename);
    }
    return submit_if_allowed(&data);
}

TRACEPOINT_PROBE(syscalls, sys_enter_socket) {
    struct data_t data = {};
    fill_common(&data, 41);
    data.arg0 = args->family;
    data.arg1 = args->type;
    data.flags = args->type;
    return submit_if_allowed(&data);
}

TRACEPOINT_PROBE(syscalls, sys_enter_connect) {
    struct data_t data = {};
    fill_common(&data, 42);
    data.arg0 = args->fd;
    data.arg1 = args->addrlen;
    return submit_if_allowed(&data);
}

TRACEPOINT_PROBE(syscalls, sys_enter_close) {
    struct data_t data = {};
    fill_common(&data, 3);
    data.arg0 = args->fd;
    return submit_if_allowed(&data);
}

TRACEPOINT_PROBE(syscalls, sys_enter_unlink) {
    struct data_t data = {};
    fill_common(&data, 87);
    if (args->pathname != 0) {
        bpf_probe_read_user_str(&data.filename, sizeof(data.filename), args->pathname);
    }
    return submit_if_allowed(&data);
}

TRACEPOINT_PROBE(syscalls, sys_enter_unlinkat) {
    struct data_t data = {};
    fill_common(&data, 263);
    if (args->pathname != 0) {
        bpf_probe_read_user_str(&data.filename, sizeof(data.filename), args->pathname);
    }
    data.arg0 = args->dfd;
    data.arg1 = args->flag;
    return submit_if_allowed(&data);
}

TRACEPOINT_PROBE(syscalls, sys_enter_rename) {
    struct data_t data = {};
    fill_common(&data, 82);
    if (args->oldname != 0) {
        bpf_probe_read_user_str(&data.filename, sizeof(data.filename), args->oldname);
    }
    return submit_if_allowed(&data);
}

TRACEPOINT_PROBE(syscalls, sys_enter_renameat) {
    struct data_t data = {};
    fill_common(&data, 264);
    if (args->oldname != 0) {
        bpf_probe_read_user_str(&data.filename, sizeof(data.filename), args->oldname);
    }
    data.arg0 = args->olddfd;
    data.arg1 = args->newdfd;
    return submit_if_allowed(&data);
}

TRACEPOINT_PROBE(syscalls, sys_enter_chmod) {
    struct data_t data = {};
    fill_common(&data, 90);
    if (args->filename != 0) {
        bpf_probe_read_user_str(&data.filename, sizeof(data.filename), args->filename);
    }
    data.arg0 = args->mode;
    return submit_if_allowed(&data);
}

TRACEPOINT_PROBE(syscalls, sys_enter_fchmod) {
    struct data_t data = {};
    fill_common(&data, 91);
    data.arg0 = args->fd;
    data.arg1 = args->mode;
    return submit_if_allowed(&data);
}

TRACEPOINT_PROBE(syscalls, sys_enter_chown) {
    struct data_t data = {};
    fill_common(&data, 92);
    if (args->filename != 0) {
        bpf_probe_read_user_str(&data.filename, sizeof(data.filename), args->filename);
    }
    data.arg0 = args->user;
    data.arg1 = args->group;
    return submit_if_allowed(&data);
}

TRACEPOINT_PROBE(syscalls, sys_enter_fchown) {
    struct data_t data = {};
    fill_common(&data, 93);
    data.arg0 = args->fd;
    data.arg1 = args->user;
    return submit_if_allowed(&data);
}

TRACEPOINT_PROBE(raw_syscalls, sys_enter) {
    struct data_t data = {};
    long id = args->id;
    if (is_enriched_syscall(id)) {
        return 0;
    }
    fill_common(&data, (u32)id);
    data.arg0 = (s64)args->args[0];
    data.arg1 = (s64)args->args[1];
    return submit_if_allowed(&data);
}
