//go:build ignore
#include <linux/bpf.h>
#include <linux/ptrace.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include <bpf/bpf_core_read.h>

#define MAX_PATH_LEN 256

struct data_t {
    __u32 pid;
    __u32 uid;
    char comm[16];
    char fname[MAX_PATH_LEN];
};

struct {
    __uint(type, BPF_MAP_TYPE_PERF_EVENT_ARRAY);
    __uint(key_size, sizeof(__u32));
    __uint(value_size, sizeof(__u32));
} events SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(key_size, sizeof(__u32));
    __uint(value_size, sizeof(__u32));
    __uint(max_entries, 1024);
} blacklist_pids SEC(".maps");

/*
 DISABLED: Hooking sys_openat causes Fatal Kernel Panics on CRDroid due to high
 frequency event lock contention.
*/

struct sockaddr_in_v4 {
    unsigned short sin_family;
    unsigned short sin_port;
    unsigned int sin_addr;
};

SEC("kprobe/security_socket_connect")
int bpf_prog_connect(struct pt_regs *ctx)
{
    struct data_t data = {};

    data.pid = bpf_get_current_pid_tgid() >> 32;
    data.uid = bpf_get_current_uid_gid() & 0xFFFFFFFF;
    /*
     * Do not write to the traced process's user stack.  bpf_probe_write_user()
     * here corrupted the return path of blacklisted processes and could crash
     * them or corrupt unrelated application state.  Enforcement belongs in a
     * dedicated LSM/return-value hook; this probe is observation-only.
     */
    bpf_get_current_comm(&data.comm, sizeof(data.comm));

    // In security_socket_connect(struct socket *sock, struct sockaddr *address, int addrlen),
    // address is the second argument (x1) and is already copied to kernel space.
    struct sockaddr_in_v4 *address = (struct sockaddr_in_v4 *)PT_REGS_PARM2(ctx);

    struct sockaddr_in_v4 addr = {};
    bpf_probe_read(&addr, sizeof(addr), address);

    unsigned int ip = addr.sin_addr;
    unsigned short port = ((addr.sin_port & 0xFF) << 8) | ((addr.sin_port >> 8) & 0xFF);

    data.fname[0] = 'I'; data.fname[1] = 'P'; data.fname[2] = ':';
    data.fname[3] = ip & 0xFF;
    data.fname[4] = (ip >> 8) & 0xFF;
    data.fname[5] = (ip >> 16) & 0xFF;
    data.fname[6] = (ip >> 24) & 0xFF;
    data.fname[7] = port >> 8;
    data.fname[8] = port & 0xFF;
    data.fname[9] = 0;

    bpf_perf_event_output(ctx, &events, BPF_F_CURRENT_CPU, &data, sizeof(data));
    return 0;
}

SEC("kprobe/do_execve_file")
int bpf_prog_execve(struct pt_regs *ctx)
{
    struct data_t data = {};

    data.pid = bpf_get_current_pid_tgid() >> 32;
    data.uid = bpf_get_current_uid_gid() & 0xFFFFFFFF;
    /* Observation-only: never overwrite the traced task's user memory. */
    bpf_get_current_comm(&data.comm, sizeof(data.comm));

    // In do_execve_file(int fd, struct filename *filename, ...), filename is the second argument (x1).
    // struct filename contains the pointer to the copied kernel string as its first member.
    void *filename_struct = (void *)PT_REGS_PARM2(ctx);
    char *fname_ptr;
    bpf_probe_read(&fname_ptr, sizeof(fname_ptr), filename_struct);

    data.fname[0] = 'E'; data.fname[1] = 'X'; data.fname[2] = 'E'; data.fname[3] = 'C'; data.fname[4] = ':';
    bpf_probe_read_str(&data.fname[5], sizeof(data.fname) - 5, fname_ptr);

    bpf_perf_event_output(ctx, &events, BPF_F_CURRENT_CPU, &data, sizeof(data));
    return 0;
}

char _license[] SEC("license") = "GPL";
