#include <bcc/helpers.h>

#define MAX_RULES 24
#define MAX_PREFIX_LEN 24
#define COMM_LEN 16
#define RINGBUF_PAGES __RINGBUF_PAGES__
/* Sentinel for "match any" in rules; avoids collision with syscall number 0 (read), uid 0 (root), etc. */
#define WILDCARD 0xFFFFFFFFU

struct rule_t {
    u32 enabled;
    u32 syscall_nr; /* WILDCARD = match any, otherwise Linux syscall number */
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
    u32 syscall_nr;
    u32 pid;
    u32 tid;
    u32 uid;
    u32 flags;
    s64 arg0;
    s64 arg1;
    s64 return_value;
    char comm[COMM_LEN];
    char filename[256];
};

BPF_ARRAY(rules, struct rule_t, MAX_RULES);
BPF_ARRAY(rule_count, u32, 1);
BPF_RINGBUF_OUTPUT(events, RINGBUF_PAGES);

/* fd->path cache for enriching read/write. Key: (pid<<32)|fd, value: path. */
struct path_val_t { char path[256]; };
BPF_HASH(open_path_temp, u64, struct path_val_t);   /* pid_tgid -> path (enter->exit) */
BPF_HASH(fd_to_path, u64, struct path_val_t);       /* (pid<<32)|fd -> path */
BPF_HASH(pid_to_parent, u32, u32);                  /* child_pid -> parent_pid for fd inheritance on fork */
/* Per-CPU scratch to avoid stack overflow when open* needs both pend and path_val_t. */
BPF_PERCPU_ARRAY(path_scratch, struct path_val_t, 1);
/* Per-CPU scratch for submit_from_pending (data_t is large). */
BPF_PERCPU_ARRAY(data_scratch, struct data_t, 1);

/* Pending for syscalls with meaningful return value (read, write, open, close, socket, etc.). */
struct syscall_pending_t {
    u64 ts;
    u32 syscall_nr;
    u32 pid;
    u32 tid;
    u32 uid;
    u32 flags;
    s64 arg0;
    s64 arg1;
    char comm[COMM_LEN];
    char filename[256];
};
BPF_HASH(syscall_pending, u64, struct syscall_pending_t);
/* Per-CPU scratch for syscall_pending_t in enter handlers (avoids stack overflow). */
BPF_PERCPU_ARRAY(pend_scratch, struct syscall_pending_t, 1);

#if ENABLE_CONNECT
#define SENTINEL_AF_INET 2
/* IPv4 peer from connect(2); userspace formats attributes.fd_sock_*. */
struct sock_enrich_val_t {
    u32 valid;
    u32 remote_ip_be;
    u16 remote_port_be;
    u16 _pad;
};
BPF_HASH(connect_peer_temp, u64, struct sock_enrich_val_t);
BPF_HASH(fd_to_sock, u64, struct sock_enrich_val_t);
BPF_PERCPU_ARRAY(sock_enrich_scratch, struct sock_enrich_val_t, 1);

static __inline void fd_sock_map_remove(u32 pid, u32 fd) {
    u64 key = ((u64)pid << 32) | (u32)fd;
    fd_to_sock.delete(&key);
}
#endif

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
                (rule->syscall_nr == WILDCARD || rule->syscall_nr == data->syscall_nr) &&
                (rule->pid == WILDCARD || rule->pid == data->pid) &&
                (rule->tid == WILDCARD || rule->tid == data->tid) &&
                (rule->uid == WILDCARD || rule->uid == data->uid) &&
                comm_matches(data, rule) && prefix_matches(data, rule)) {
                allowed = 1;
            }
        }
    }
    return allowed;
}

static __inline int fill_common(struct data_t *data, u32 syscall_nr) {
    u64 pid_tgid = bpf_get_current_pid_tgid();
    data->pid = pid_tgid >> 32;
    data->tid = pid_tgid;
    data->uid = bpf_get_current_uid_gid();
    data->ts = bpf_ktime_get_ns();
    data->syscall_nr = syscall_nr;
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

static __inline void pend_from_common(struct syscall_pending_t *pend, u32 syscall_nr) {
    u64 pid_tgid = bpf_get_current_pid_tgid();
    pend->ts = bpf_ktime_get_ns();
    pend->syscall_nr = syscall_nr;
    pend->pid = pid_tgid >> 32;
    pend->tid = pid_tgid;
    pend->uid = bpf_get_current_uid_gid();
    pend->flags = 0;
    pend->arg0 = 0;
    pend->arg1 = 0;
    bpf_get_current_comm(&pend->comm, sizeof(pend->comm));
    pend->filename[0] = 0;
}

static __inline void submit_from_pending(struct syscall_pending_t *pend, s64 ret) {
    u32 zero = 0;
    struct data_t *data = data_scratch.lookup(&zero);
    if (!data) return;
    data->ts = pend->ts;
    data->syscall_nr = pend->syscall_nr;
    data->pid = pend->pid;
    data->tid = pend->tid;
    data->uid = pend->uid;
    data->flags = pend->flags;
    data->arg0 = pend->arg0;
    data->arg1 = pend->arg1;
    data->return_value = ret;
    __builtin_memcpy(&data->comm, &pend->comm, sizeof(data->comm));
    __builtin_memcpy(&data->filename, &pend->filename, sizeof(data->filename));
    submit_if_allowed(data);
}

static __inline int is_enriched_syscall(long id) {
    return 0
#if ENABLE_READ
        || id == 0
#endif
#if ENABLE_WRITE
        || id == 1
#endif
#if ENABLE_OPEN
        || id == 2
#endif
#if ENABLE_CLOSE
        || id == 3
#endif
#if ENABLE_FSTAT
        || id == 5
#endif
#if ENABLE_POLL
        || id == 7
#endif
#if ENABLE_MMAP
        || id == 9
#endif
#if ENABLE_MPROTECT
        || id == 10
#endif
#if ENABLE_IOCTL
        || id == 16
#endif
#if ENABLE_SOCKET
        || id == 41
#endif
#if ENABLE_CONNECT
        || id == 42
#endif
#if ENABLE_ACCEPT
        || id == 43
#endif
#if ENABLE_SENDTO
        || id == 44
#endif
#if ENABLE_RECVFROM
        || id == 45
#endif
#if ENABLE_BIND
        || id == 49
#endif
#if ENABLE_LISTEN
        || id == 50
#endif
#if ENABLE_GETSOCKNAME
        || id == 51
#endif
#if ENABLE_GETPEERNAME
        || id == 52
#endif
#if ENABLE_SETSOCKOPT
        || id == 54
#endif
#if ENABLE_FORK
        || id == 57
#endif
#if ENABLE_EXECVE
        || id == 59
#endif
#if ENABLE_FCNTL
        || id == 72
#endif
#if ENABLE_RENAME
        || id == 82
#endif
#if ENABLE_UNLINK
        || id == 87
#endif
#if ENABLE_CHMOD
        || id == 90
#endif
#if ENABLE_FCHMOD
        || id == 91
#endif
#if ENABLE_CHOWN
        || id == 92
#endif
#if ENABLE_FCHOWN
        || id == 93
#endif
#if ENABLE_OPENAT
        || id == 257
#endif
#if ENABLE_UNLINKAT
        || id == 263
#endif
#if ENABLE_RENAMEAT
        || id == 264
#endif
#if ENABLE_SET_ROBUST_LIST
        || id == 273
#endif
#if ENABLE_ACCEPT4
        || id == 288
#endif
#if ENABLE_SENDMMSG
        || id == 307
#endif
#if ENABLE_OPENAT2
        || id == 437
#endif
    ;
}

static __inline void fd_map_store_path(u32 pid, u32 fd, const char *path) {
    u64 key = ((u64)pid << 32) | (u32)fd;
    u32 zero = 0;
    struct path_val_t *val = path_scratch.lookup(&zero);
    if (val) {
        __builtin_memcpy(val->path, path, sizeof(val->path));
        fd_to_path.update(&key, val);
    }
}

static __inline void fd_map_remove(u32 pid, u32 fd) {
    u64 key = ((u64)pid << 32) | (u32)fd;
    fd_to_path.delete(&key);
}

static __inline void fd_map_lookup(u32 pid, u32 fd, struct data_t *data) {
    u64 key = ((u64)pid << 32) | (u32)fd;
    struct path_val_t *val = fd_to_path.lookup(&key);
    if (val) {
        __builtin_memcpy(data->filename, val->path, sizeof(data->filename));
    }
}

/* Lookup path for (pid, fd); on fork, child inherits fds so fall back to parent. */
static __inline void fd_map_lookup_buf(u32 pid, u32 fd, char *dest) {
    u32 cur = pid;
    for (int i = 0; i < 8; i++) {
        u64 key = ((u64)cur << 32) | (u32)fd;
        struct path_val_t *val = fd_to_path.lookup(&key);
        if (val) {
            __builtin_memcpy(dest, val->path, 256);
            return;
        }
        u32 *parent = pid_to_parent.lookup(&cur);
        if (!parent) return;
        cur = *parent;
    }
}

/* Get pend from per-CPU scratch; returns NULL if lookup fails. */
static __inline struct syscall_pending_t *pend_scratch_get(void) {
    u32 zero = 0;
    return pend_scratch.lookup(&zero);
}

/* Store path in open_path_temp using per-CPU scratch (avoids stack overflow). */
static __inline void open_path_temp_store(u64 pid_tgid, const char *path) {
    u32 zero = 0;
    struct path_val_t *pv = path_scratch.lookup(&zero);
    if (pv) {
        __builtin_memcpy(pv->path, path, 256);
        open_path_temp.update(&pid_tgid, pv);
    }
}

/* read(0) and write(1) enriched via fd->path cache from open/close */

#if (ENABLE_OPEN || ENABLE_READ || ENABLE_WRITE)
#if ENABLE_OPEN
TRACEPOINT_PROBE(syscalls, sys_enter_open) {
    u64 pid_tgid = bpf_get_current_pid_tgid();
    struct syscall_pending_t *pend = pend_scratch_get();
    if (!pend) return 0;
    pend_from_common(pend, 2);
    pend->flags = args->flags;
    if (args->filename != 0) {
        bpf_probe_read_user_str(&pend->filename, sizeof(pend->filename), args->filename);
    }
    pend->arg0 = args->flags;
    syscall_pending.update(&pid_tgid, pend);
#if (ENABLE_READ || ENABLE_WRITE)
    open_path_temp_store(pid_tgid, pend->filename);
#endif
    return 0;
}
#endif
#if ENABLE_OPEN
TRACEPOINT_PROBE(syscalls, sys_exit_open) {
    long ret = args->ret;
    u64 pid_tgid = bpf_get_current_pid_tgid();
    u32 pid = pid_tgid >> 32;
#if (ENABLE_READ || ENABLE_WRITE)
    if (ret >= 0) {
        struct path_val_t *pv = open_path_temp.lookup(&pid_tgid);
        if (pv) {
            fd_map_store_path(pid, (u32)ret, pv->path);
            open_path_temp.delete(&pid_tgid);
        }
    }
#endif
#if ENABLE_OPEN
    struct syscall_pending_t *pend = syscall_pending.lookup(&pid_tgid);
    if (pend) {
        submit_from_pending(pend, ret);
        syscall_pending.delete(&pid_tgid);
    }
#endif
    return 0;
}
#endif
#endif

#if (ENABLE_OPENAT || ENABLE_READ || ENABLE_WRITE)
#if ENABLE_OPENAT
TRACEPOINT_PROBE(syscalls, sys_enter_openat) {
    u64 pid_tgid = bpf_get_current_pid_tgid();
    struct syscall_pending_t *pend = pend_scratch_get();
    if (!pend) return 0;
    pend_from_common(pend, 257);
    pend->flags = args->flags;
    if (args->filename != 0) {
        bpf_probe_read_user_str(&pend->filename, sizeof(pend->filename), args->filename);
    }
    pend->arg0 = args->dfd;
    pend->arg1 = args->flags;
    syscall_pending.update(&pid_tgid, pend);
#if (ENABLE_READ || ENABLE_WRITE)
    open_path_temp_store(pid_tgid, pend->filename);
#endif
    return 0;
}
#endif
#if ENABLE_OPENAT
TRACEPOINT_PROBE(syscalls, sys_exit_openat) {
    long ret = args->ret;
    u64 pid_tgid = bpf_get_current_pid_tgid();
    u32 pid = pid_tgid >> 32;
#if (ENABLE_READ || ENABLE_WRITE)
    if (ret >= 0) {
        struct path_val_t *pv = open_path_temp.lookup(&pid_tgid);
        if (pv) {
            fd_map_store_path(pid, (u32)ret, pv->path);
            open_path_temp.delete(&pid_tgid);
        }
    }
#endif
#if ENABLE_OPENAT
    struct syscall_pending_t *pend = syscall_pending.lookup(&pid_tgid);
    if (pend) {
        submit_from_pending(pend, ret);
        syscall_pending.delete(&pid_tgid);
    }
#endif
    return 0;
}
#endif
#endif

#if (ENABLE_OPENAT2 || ENABLE_READ || ENABLE_WRITE)
#if ENABLE_OPENAT2
TRACEPOINT_PROBE(syscalls, sys_enter_openat2) {
    u64 pid_tgid = bpf_get_current_pid_tgid();
    struct syscall_pending_t *pend = pend_scratch_get();
    if (!pend) return 0;
    pend_from_common(pend, 437);
    if (args->filename != 0) {
        bpf_probe_read_user_str(&pend->filename, sizeof(pend->filename), args->filename);
    }
    pend->arg0 = args->dfd;
    pend->arg1 = args->usize;
    syscall_pending.update(&pid_tgid, pend);
#if (ENABLE_READ || ENABLE_WRITE)
    open_path_temp_store(pid_tgid, pend->filename);
#endif
    return 0;
}
#endif
#if ENABLE_OPENAT2
TRACEPOINT_PROBE(syscalls, sys_exit_openat2) {
    long ret = args->ret;
    u64 pid_tgid = bpf_get_current_pid_tgid();
    u32 pid = pid_tgid >> 32;
#if (ENABLE_READ || ENABLE_WRITE)
    if (ret >= 0) {
        struct path_val_t *pv = open_path_temp.lookup(&pid_tgid);
        if (pv) {
            fd_map_store_path(pid, (u32)ret, pv->path);
            open_path_temp.delete(&pid_tgid);
        }
    }
#endif
#if ENABLE_OPENAT2
    struct syscall_pending_t *pend = syscall_pending.lookup(&pid_tgid);
    if (pend) {
        submit_from_pending(pend, ret);
        syscall_pending.delete(&pid_tgid);
    }
#endif
    return 0;
}
#endif
#endif

#if ENABLE_EXECVE
TRACEPOINT_PROBE(syscalls, sys_enter_execve) {
    u64 pid_tgid = bpf_get_current_pid_tgid();
    struct syscall_pending_t *pend = pend_scratch_get();
    if (!pend) return 0;
    pend_from_common(pend, 59);
    if (args->filename != 0) {
        bpf_probe_read_user_str(&pend->filename, sizeof(pend->filename), args->filename);
    }
    syscall_pending.update(&pid_tgid, pend);
    return 0;
}
TRACEPOINT_PROBE(syscalls, sys_exit_execve) {
    long ret = args->ret;
    u64 pid_tgid = bpf_get_current_pid_tgid();
    struct syscall_pending_t *pend = syscall_pending.lookup(&pid_tgid);
    if (pend) {
        submit_from_pending(pend, ret);
        syscall_pending.delete(&pid_tgid);
    }
    return 0;
}
#endif

#if ENABLE_SOCKET
TRACEPOINT_PROBE(syscalls, sys_enter_socket) {
    u64 pid_tgid = bpf_get_current_pid_tgid();
    struct syscall_pending_t *pend = pend_scratch_get();
    if (!pend) return 0;
    pend_from_common(pend, 41);
    pend->arg0 = args->family;
    pend->arg1 = args->type;
    pend->flags = args->type;
    syscall_pending.update(&pid_tgid, pend);
    return 0;
}
TRACEPOINT_PROBE(syscalls, sys_exit_socket) {
    long ret = args->ret;
    u64 pid_tgid = bpf_get_current_pid_tgid();
    struct syscall_pending_t *pend = syscall_pending.lookup(&pid_tgid);
    if (pend) {
        submit_from_pending(pend, ret);
        syscall_pending.delete(&pid_tgid);
    }
    return 0;
}
#endif

#if ENABLE_CONNECT
TRACEPOINT_PROBE(syscalls, sys_enter_connect) {
    u64 pid_tgid = bpf_get_current_pid_tgid();
    u32 zero = 0;
    struct sock_enrich_val_t *se = sock_enrich_scratch.lookup(&zero);
    if (se) {
        se->valid = 0;
        se->remote_ip_be = 0;
        se->remote_port_be = 0;
        se->_pad = 0;
        if (args->uservaddr != 0 && args->addrlen >= 8) {
            u16 family = 0;
            bpf_probe_read_user(&family, sizeof(family), (void *)args->uservaddr);
            if (family == SENTINEL_AF_INET) {
                struct {
                    u16 sin_family;
                    u16 sin_port;
                    u32 sin_addr;
                } sa;
                bpf_probe_read_user(&sa, sizeof(sa), (void *)args->uservaddr);
                se->valid = 1;
                se->remote_ip_be = sa.sin_addr;
                se->remote_port_be = sa.sin_port;
            }
        }
        connect_peer_temp.update(&pid_tgid, se);
    }
    struct syscall_pending_t *pend = pend_scratch_get();
    if (!pend) return 0;
    pend_from_common(pend, 42);
    pend->arg0 = args->fd;
    pend->arg1 = args->addrlen;
    syscall_pending.update(&pid_tgid, pend);
    return 0;
}
TRACEPOINT_PROBE(syscalls, sys_exit_connect) {
    long ret = args->ret;
    u64 pid_tgid = bpf_get_current_pid_tgid();
    u32 pid = pid_tgid >> 32;
    struct syscall_pending_t *pend = syscall_pending.lookup(&pid_tgid);
    if (pend) {
        if (ret == 0) {
            struct sock_enrich_val_t *st = connect_peer_temp.lookup(&pid_tgid);
            u32 z = 0;
            struct sock_enrich_val_t *slot = sock_enrich_scratch.lookup(&z);
            if (st && slot && st->valid) {
                u64 keyfd = ((u64)pid << 32) | (u32)pend->arg0;
                slot->valid = st->valid;
                slot->remote_ip_be = st->remote_ip_be;
                slot->remote_port_be = st->remote_port_be;
                slot->_pad = 0;
                fd_to_sock.update(&keyfd, slot);
            }
        }
        connect_peer_temp.delete(&pid_tgid);
        submit_from_pending(pend, ret);
        syscall_pending.delete(&pid_tgid);
    }
    return 0;
}
#endif

#if ENABLE_BIND
TRACEPOINT_PROBE(syscalls, sys_enter_bind) {
    u64 pid_tgid = bpf_get_current_pid_tgid();
    struct syscall_pending_t *pend = pend_scratch_get();
    if (!pend) return 0;
    pend_from_common(pend, 49);
    pend->arg0 = args->fd;
    pend->arg1 = args->addrlen;
    syscall_pending.update(&pid_tgid, pend);
    return 0;
}
TRACEPOINT_PROBE(syscalls, sys_exit_bind) {
    long ret = args->ret;
    u64 pid_tgid = bpf_get_current_pid_tgid();
    struct syscall_pending_t *pend = syscall_pending.lookup(&pid_tgid);
    if (pend) {
        submit_from_pending(pend, ret);
        syscall_pending.delete(&pid_tgid);
    }
    return 0;
}
#endif

#if ENABLE_LISTEN
TRACEPOINT_PROBE(syscalls, sys_enter_listen) {
    u64 pid_tgid = bpf_get_current_pid_tgid();
    struct syscall_pending_t *pend = pend_scratch_get();
    if (!pend) return 0;
    pend_from_common(pend, 50);
    pend->arg0 = args->fd;
    pend->arg1 = args->backlog;
    syscall_pending.update(&pid_tgid, pend);
    return 0;
}
TRACEPOINT_PROBE(syscalls, sys_exit_listen) {
    long ret = args->ret;
    u64 pid_tgid = bpf_get_current_pid_tgid();
    struct syscall_pending_t *pend = syscall_pending.lookup(&pid_tgid);
    if (pend) {
        submit_from_pending(pend, ret);
        syscall_pending.delete(&pid_tgid);
    }
    return 0;
}
#endif

#if ENABLE_ACCEPT
TRACEPOINT_PROBE(syscalls, sys_enter_accept) {
    u64 pid_tgid = bpf_get_current_pid_tgid();
    struct syscall_pending_t *pend = pend_scratch_get();
    if (!pend) return 0;
    pend_from_common(pend, 43);
    pend->arg0 = args->fd;
    pend->arg1 = 0;
    syscall_pending.update(&pid_tgid, pend);
    return 0;
}
TRACEPOINT_PROBE(syscalls, sys_exit_accept) {
    long ret = args->ret;
    u64 pid_tgid = bpf_get_current_pid_tgid();
    struct syscall_pending_t *pend = syscall_pending.lookup(&pid_tgid);
    if (pend) {
        submit_from_pending(pend, ret);
        syscall_pending.delete(&pid_tgid);
    }
    return 0;
}
#endif

#if ENABLE_ACCEPT4
TRACEPOINT_PROBE(syscalls, sys_enter_accept4) {
    u64 pid_tgid = bpf_get_current_pid_tgid();
    struct syscall_pending_t *pend = pend_scratch_get();
    if (!pend) return 0;
    pend_from_common(pend, 288);
    pend->arg0 = args->fd;
    pend->arg1 = args->flags;
    syscall_pending.update(&pid_tgid, pend);
    return 0;
}
TRACEPOINT_PROBE(syscalls, sys_exit_accept4) {
    long ret = args->ret;
    u64 pid_tgid = bpf_get_current_pid_tgid();
    struct syscall_pending_t *pend = syscall_pending.lookup(&pid_tgid);
    if (pend) {
        submit_from_pending(pend, ret);
        syscall_pending.delete(&pid_tgid);
    }
    return 0;
}
#endif

#if ENABLE_FORK
TRACEPOINT_PROBE(syscalls, sys_enter_fork) {
    u64 pid_tgid = bpf_get_current_pid_tgid();
    struct syscall_pending_t *pend = pend_scratch_get();
    if (!pend) return 0;
    pend_from_common(pend, 57);
    syscall_pending.update(&pid_tgid, pend);
    return 0;
}
TRACEPOINT_PROBE(syscalls, sys_exit_fork) {
    long ret = args->ret;
    u64 pid_tgid = bpf_get_current_pid_tgid();
    u32 parent_pid = pid_tgid >> 32;
    if (ret > 0) {
        u32 child_pid = (u32)ret;
        pid_to_parent.update(&child_pid, &parent_pid);
    }
    struct syscall_pending_t *pend = syscall_pending.lookup(&pid_tgid);
    if (pend) {
        submit_from_pending(pend, ret);
        syscall_pending.delete(&pid_tgid);
    }
    return 0;
}
#endif

#if (ENABLE_CLOSE || ENABLE_READ || ENABLE_WRITE)
#if ENABLE_CLOSE
TRACEPOINT_PROBE(syscalls, sys_enter_close) {
    u64 pid_tgid = bpf_get_current_pid_tgid();
    u32 pid = pid_tgid >> 32;
    struct syscall_pending_t *pend = pend_scratch_get();
    if (!pend) return 0;
    pend_from_common(pend, 3);
    pend->arg0 = args->fd;
    syscall_pending.update(&pid_tgid, pend);
#if (ENABLE_READ || ENABLE_WRITE)
    fd_map_remove(pid, (u32)args->fd);
#endif
#if ENABLE_CONNECT
    fd_sock_map_remove(pid, (u32)args->fd);
#endif
    return 0;
}
#endif
#if ENABLE_CLOSE
TRACEPOINT_PROBE(syscalls, sys_exit_close) {
    long ret = args->ret;
    u64 pid_tgid = bpf_get_current_pid_tgid();
    struct syscall_pending_t *pend = syscall_pending.lookup(&pid_tgid);
    if (pend) {
        submit_from_pending(pend, ret);
        syscall_pending.delete(&pid_tgid);
    }
    return 0;
}
#endif
#endif

#if ENABLE_READ
TRACEPOINT_PROBE(syscalls, sys_enter_read) {
    u64 pid_tgid = bpf_get_current_pid_tgid();
    struct syscall_pending_t *pend = pend_scratch_get();
    if (!pend) return 0;
    pend_from_common(pend, 0);
    pend->arg0 = args->fd;
    pend->arg1 = args->count;
    fd_map_lookup_buf(pend->pid, (u32)args->fd, pend->filename);
    syscall_pending.update(&pid_tgid, pend);
    return 0;
}
TRACEPOINT_PROBE(syscalls, sys_exit_read) {
    long ret = args->ret;
    u64 pid_tgid = bpf_get_current_pid_tgid();
    struct syscall_pending_t *pend = syscall_pending.lookup(&pid_tgid);
    if (pend) {
        submit_from_pending(pend, ret);
        syscall_pending.delete(&pid_tgid);
    }
    return 0;
}
#endif

#if ENABLE_WRITE
TRACEPOINT_PROBE(syscalls, sys_enter_write) {
    u64 pid_tgid = bpf_get_current_pid_tgid();
    struct syscall_pending_t *pend = pend_scratch_get();
    if (!pend) return 0;
    pend_from_common(pend, 1);
    pend->arg0 = args->fd;
    pend->arg1 = args->count;
    fd_map_lookup_buf(pend->pid, (u32)args->fd, pend->filename);
    syscall_pending.update(&pid_tgid, pend);
    return 0;
}
TRACEPOINT_PROBE(syscalls, sys_exit_write) {
    long ret = args->ret;
    u64 pid_tgid = bpf_get_current_pid_tgid();
    struct syscall_pending_t *pend = syscall_pending.lookup(&pid_tgid);
    if (pend) {
        submit_from_pending(pend, ret);
        syscall_pending.delete(&pid_tgid);
    }
    return 0;
}
#endif

#if ENABLE_UNLINK
TRACEPOINT_PROBE(syscalls, sys_enter_unlink) {
    struct data_t data = {};
    fill_common(&data, 87);
    if (args->pathname != 0) {
        bpf_probe_read_user_str(&data.filename, sizeof(data.filename), args->pathname);
    }
    return submit_if_allowed(&data);
}
#endif

#if ENABLE_UNLINKAT
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
#endif

#if ENABLE_RENAME
TRACEPOINT_PROBE(syscalls, sys_enter_rename) {
    struct data_t data = {};
    fill_common(&data, 82);
    if (args->oldname != 0) {
        bpf_probe_read_user_str(&data.filename, sizeof(data.filename), args->oldname);
    }
    return submit_if_allowed(&data);
}
#endif

#if ENABLE_RENAMEAT
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
#endif

#if ENABLE_CHMOD
TRACEPOINT_PROBE(syscalls, sys_enter_chmod) {
    struct data_t data = {};
    fill_common(&data, 90);
    if (args->filename != 0) {
        bpf_probe_read_user_str(&data.filename, sizeof(data.filename), args->filename);
    }
    data.arg0 = args->mode;
    return submit_if_allowed(&data);
}
#endif

#if ENABLE_FCHMOD
TRACEPOINT_PROBE(syscalls, sys_enter_fchmod) {
    struct data_t data = {};
    fill_common(&data, 91);
    data.arg0 = args->fd;
    data.arg1 = args->mode;
    fd_map_lookup_buf(data.pid, (u32)args->fd, data.filename);
    return submit_if_allowed(&data);
}
#endif

#if ENABLE_CHOWN
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
#endif

#if ENABLE_FCHOWN
TRACEPOINT_PROBE(syscalls, sys_enter_fchown) {
    struct data_t data = {};
    fill_common(&data, 93);
    data.arg0 = args->fd;
    data.arg1 = args->user;
    fd_map_lookup_buf(data.pid, (u32)args->fd, data.filename);
    return submit_if_allowed(&data);
}
#endif

#if ENABLE_FSTAT
TRACEPOINT_PROBE(syscalls, sys_enter_fstat) {
    u64 pid_tgid = bpf_get_current_pid_tgid();
    struct syscall_pending_t *pend = pend_scratch_get();
    if (!pend) return 0;
    pend_from_common(pend, 5);
    pend->arg0 = args->fd;
    pend->arg1 = 0;
    fd_map_lookup_buf(pend->pid, (u32)args->fd, pend->filename);
    syscall_pending.update(&pid_tgid, pend);
    return 0;
}
TRACEPOINT_PROBE(syscalls, sys_exit_fstat) {
    long ret = args->ret;
    u64 pid_tgid = bpf_get_current_pid_tgid();
    struct syscall_pending_t *pend = syscall_pending.lookup(&pid_tgid);
    if (pend) {
        submit_from_pending(pend, ret);
        syscall_pending.delete(&pid_tgid);
    }
    return 0;
}
#endif

#if ENABLE_POLL
TRACEPOINT_PROBE(syscalls, sys_enter_poll) {
    u64 pid_tgid = bpf_get_current_pid_tgid();
    struct syscall_pending_t *pend = pend_scratch_get();
    if (!pend) return 0;
    pend_from_common(pend, 7);
    pend->arg0 = args->nfds;
    /* field name is kernel-dependent; on modern kernels this is timeout_msecs */
    pend->arg1 = args->timeout_msecs;
    syscall_pending.update(&pid_tgid, pend);
    return 0;
}
TRACEPOINT_PROBE(syscalls, sys_exit_poll) {
    long ret = args->ret;
    u64 pid_tgid = bpf_get_current_pid_tgid();
    struct syscall_pending_t *pend = syscall_pending.lookup(&pid_tgid);
    if (pend) {
        submit_from_pending(pend, ret);
        syscall_pending.delete(&pid_tgid);
    }
    return 0;
}
#endif

#if ENABLE_MMAP
TRACEPOINT_PROBE(syscalls, sys_enter_mmap) {
    u64 pid_tgid = bpf_get_current_pid_tgid();
    struct syscall_pending_t *pend = pend_scratch_get();
    if (!pend) return 0;
    pend_from_common(pend, 9);
    pend->arg0 = (s64)args->len;
    pend->arg1 = (s64)args->flags;
    syscall_pending.update(&pid_tgid, pend);
    return 0;
}
TRACEPOINT_PROBE(syscalls, sys_exit_mmap) {
    long ret = args->ret;
    u64 pid_tgid = bpf_get_current_pid_tgid();
    struct syscall_pending_t *pend = syscall_pending.lookup(&pid_tgid);
    if (pend) {
        submit_from_pending(pend, ret);
        syscall_pending.delete(&pid_tgid);
    }
    return 0;
}
#endif

#if ENABLE_MPROTECT
TRACEPOINT_PROBE(syscalls, sys_enter_mprotect) {
    u64 pid_tgid = bpf_get_current_pid_tgid();
    struct syscall_pending_t *pend = pend_scratch_get();
    if (!pend) return 0;
    pend_from_common(pend, 10);
    pend->arg0 = (s64)args->start;
    pend->arg1 = (s64)args->len;
    syscall_pending.update(&pid_tgid, pend);
    return 0;
}
TRACEPOINT_PROBE(syscalls, sys_exit_mprotect) {
    long ret = args->ret;
    u64 pid_tgid = bpf_get_current_pid_tgid();
    struct syscall_pending_t *pend = syscall_pending.lookup(&pid_tgid);
    if (pend) {
        submit_from_pending(pend, ret);
        syscall_pending.delete(&pid_tgid);
    }
    return 0;
}
#endif

#if ENABLE_IOCTL
TRACEPOINT_PROBE(syscalls, sys_enter_ioctl) {
    u64 pid_tgid = bpf_get_current_pid_tgid();
    struct syscall_pending_t *pend = pend_scratch_get();
    if (!pend) return 0;
    pend_from_common(pend, 16);
    pend->arg0 = args->fd;
    pend->arg1 = args->cmd;
    fd_map_lookup_buf(pend->pid, (u32)args->fd, pend->filename);
    syscall_pending.update(&pid_tgid, pend);
    return 0;
}
TRACEPOINT_PROBE(syscalls, sys_exit_ioctl) {
    long ret = args->ret;
    u64 pid_tgid = bpf_get_current_pid_tgid();
    struct syscall_pending_t *pend = syscall_pending.lookup(&pid_tgid);
    if (pend) {
        submit_from_pending(pend, ret);
        syscall_pending.delete(&pid_tgid);
    }
    return 0;
}
#endif

#if ENABLE_SENDTO
TRACEPOINT_PROBE(syscalls, sys_enter_sendto) {
    u64 pid_tgid = bpf_get_current_pid_tgid();
    struct syscall_pending_t *pend = pend_scratch_get();
    if (!pend) return 0;
    pend_from_common(pend, 44);
    pend->arg0 = args->fd;
    pend->arg1 = args->len;
    syscall_pending.update(&pid_tgid, pend);
    return 0;
}
TRACEPOINT_PROBE(syscalls, sys_exit_sendto) {
    long ret = args->ret;
    u64 pid_tgid = bpf_get_current_pid_tgid();
    struct syscall_pending_t *pend = syscall_pending.lookup(&pid_tgid);
    if (pend) {
        submit_from_pending(pend, ret);
        syscall_pending.delete(&pid_tgid);
    }
    return 0;
}
#endif

#if ENABLE_RECVFROM
TRACEPOINT_PROBE(syscalls, sys_enter_recvfrom) {
    u64 pid_tgid = bpf_get_current_pid_tgid();
    struct syscall_pending_t *pend = pend_scratch_get();
    if (!pend) return 0;
    pend_from_common(pend, 45);
    pend->arg0 = args->fd;
    pend->arg1 = args->size;
    syscall_pending.update(&pid_tgid, pend);
    return 0;
}
TRACEPOINT_PROBE(syscalls, sys_exit_recvfrom) {
    long ret = args->ret;
    u64 pid_tgid = bpf_get_current_pid_tgid();
    struct syscall_pending_t *pend = syscall_pending.lookup(&pid_tgid);
    if (pend) {
        submit_from_pending(pend, ret);
        syscall_pending.delete(&pid_tgid);
    }
    return 0;
}
#endif

#if ENABLE_GETSOCKNAME
TRACEPOINT_PROBE(syscalls, sys_enter_getsockname) {
    u64 pid_tgid = bpf_get_current_pid_tgid();
    struct syscall_pending_t *pend = pend_scratch_get();
    if (!pend) return 0;
    pend_from_common(pend, 51);
    pend->arg0 = args->fd;
    pend->arg1 = 0;
    syscall_pending.update(&pid_tgid, pend);
    return 0;
}
TRACEPOINT_PROBE(syscalls, sys_exit_getsockname) {
    long ret = args->ret;
    u64 pid_tgid = bpf_get_current_pid_tgid();
    struct syscall_pending_t *pend = syscall_pending.lookup(&pid_tgid);
    if (pend) {
        submit_from_pending(pend, ret);
        syscall_pending.delete(&pid_tgid);
    }
    return 0;
}
#endif

#if ENABLE_GETPEERNAME
TRACEPOINT_PROBE(syscalls, sys_enter_getpeername) {
    u64 pid_tgid = bpf_get_current_pid_tgid();
    struct syscall_pending_t *pend = pend_scratch_get();
    if (!pend) return 0;
    pend_from_common(pend, 52);
    pend->arg0 = args->fd;
    pend->arg1 = 0;
    syscall_pending.update(&pid_tgid, pend);
    return 0;
}
TRACEPOINT_PROBE(syscalls, sys_exit_getpeername) {
    long ret = args->ret;
    u64 pid_tgid = bpf_get_current_pid_tgid();
    struct syscall_pending_t *pend = syscall_pending.lookup(&pid_tgid);
    if (pend) {
        submit_from_pending(pend, ret);
        syscall_pending.delete(&pid_tgid);
    }
    return 0;
}
#endif

#if ENABLE_SETSOCKOPT
TRACEPOINT_PROBE(syscalls, sys_enter_setsockopt) {
    u64 pid_tgid = bpf_get_current_pid_tgid();
    struct syscall_pending_t *pend = pend_scratch_get();
    if (!pend) return 0;
    pend_from_common(pend, 54);
    pend->arg0 = args->fd;
    pend->arg1 = args->optname;
    syscall_pending.update(&pid_tgid, pend);
    return 0;
}
TRACEPOINT_PROBE(syscalls, sys_exit_setsockopt) {
    long ret = args->ret;
    u64 pid_tgid = bpf_get_current_pid_tgid();
    struct syscall_pending_t *pend = syscall_pending.lookup(&pid_tgid);
    if (pend) {
        submit_from_pending(pend, ret);
        syscall_pending.delete(&pid_tgid);
    }
    return 0;
}
#endif

#if ENABLE_FCNTL
TRACEPOINT_PROBE(syscalls, sys_enter_fcntl) {
    u64 pid_tgid = bpf_get_current_pid_tgid();
    struct syscall_pending_t *pend = pend_scratch_get();
    if (!pend) return 0;
    pend_from_common(pend, 72);
    pend->arg0 = args->fd;
    pend->arg1 = args->cmd;
    fd_map_lookup_buf(pend->pid, (u32)args->fd, pend->filename);
    syscall_pending.update(&pid_tgid, pend);
    return 0;
}
TRACEPOINT_PROBE(syscalls, sys_exit_fcntl) {
    long ret = args->ret;
    u64 pid_tgid = bpf_get_current_pid_tgid();
    struct syscall_pending_t *pend = syscall_pending.lookup(&pid_tgid);
    if (pend) {
        submit_from_pending(pend, ret);
        syscall_pending.delete(&pid_tgid);
    }
    return 0;
}
#endif

#if ENABLE_SET_ROBUST_LIST
TRACEPOINT_PROBE(syscalls, sys_enter_set_robust_list) {
    u64 pid_tgid = bpf_get_current_pid_tgid();
    struct syscall_pending_t *pend = pend_scratch_get();
    if (!pend) return 0;
    pend_from_common(pend, 273);
    pend->arg0 = (s64)args->head;
    pend->arg1 = (s64)args->len;
    syscall_pending.update(&pid_tgid, pend);
    return 0;
}
TRACEPOINT_PROBE(syscalls, sys_exit_set_robust_list) {
    long ret = args->ret;
    u64 pid_tgid = bpf_get_current_pid_tgid();
    struct syscall_pending_t *pend = syscall_pending.lookup(&pid_tgid);
    if (pend) {
        submit_from_pending(pend, ret);
        syscall_pending.delete(&pid_tgid);
    }
    return 0;
}
#endif

#if ENABLE_SENDMMSG
TRACEPOINT_PROBE(syscalls, sys_enter_sendmmsg) {
    u64 pid_tgid = bpf_get_current_pid_tgid();
    struct syscall_pending_t *pend = pend_scratch_get();
    if (!pend) return 0;
    pend_from_common(pend, 307);
    pend->arg0 = args->fd;
    pend->arg1 = args->vlen;
    syscall_pending.update(&pid_tgid, pend);
    return 0;
}
TRACEPOINT_PROBE(syscalls, sys_exit_sendmmsg) {
    long ret = args->ret;
    u64 pid_tgid = bpf_get_current_pid_tgid();
    struct syscall_pending_t *pend = syscall_pending.lookup(&pid_tgid);
    if (pend) {
        submit_from_pending(pend, ret);
        syscall_pending.delete(&pid_tgid);
    }
    return 0;
}
#endif

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
