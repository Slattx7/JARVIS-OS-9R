//go:build ignore
/*
 * coverage_test_v2.c - does the REAL DriftNet probe hooks actually see these?
 *
 * Supersedes the coverage_gaps.c from earlier in this conversation. That one guessed
 * at raw __arm64_sys_* syscall hooks (openat/connect/execve), matching the README's
 * diagram. The real bpf_probe.c hooks two different things instead:
 *   - kprobe/security_socket_connect  (an LSM hook, specific to the connect() path)
 *   - kprobe/do_execve_file           (an internal exec-handling function)
 * openat monitoring is fully disabled in the shipped code (its own comment says it
 * caused kernel panics on CRDroid) -- there's nothing live to test there, so this
 * drops the openat2 check entirely rather than probing a feature that doesn't exist.
 *
 * Also worth knowing before you run this: NEITHER live probe has the "skip uid <
 * 10000" filter you might expect from the whitepaper -- that filter only ever
 * existed inside the disabled openat block. The two probes that actually run watch
 * every UID, system daemons included. So this test doesn't need root or a special
 * UID to be SEEN; you only need root to read the dashboard/logs afterward and
 * confirm what showed up.
 *
 * Three checks, each printed with a unique COVTEST_ marker to grep for on the
 * driftnetd side afterward:
 *
 *   1. connect() on a plain TCP socket       -> SHOULD be caught (this is exactly
 *                                               the path security_socket_connect
 *                                               sits on)
 *   2. sendto() on an UNCONNECTED UDP socket -> the actual gap under test: this
 *                                               never calls connect(), so it should
 *                                               NOT traverse security_socket_connect
 *                                               at all if the hook is doing what its
 *                                               name implies
 *   3. execveat() instead of execve()        -> genuinely unknown without testing --
 *                                               do_execve_file may or may not be a
 *                                               shared choke point for both on this
 *                                               specific kernel/vendor fork
 *
 * Build (from wherever you already build anti_debug_arm64 / vulkan_engine_arm64--
 * same NDK toolchain):
 *   aarch64-linux-android30-clang -O2 -Wall -o coverage_test_v2 coverage_test_v2.c
 * Push and run (no root needed to GENERATE the traffic, only to read results after):
 *   adb push coverage_test_v2 /data/local/tmp/
 *   adb shell chmod 755 /data/local/tmp/coverage_test_v2
 *   adb shell /data/local/tmp/coverage_test_v2
 * Then, as root: grep driftnetd's stdout/log and the dashboard for "covtest".
 *
 * Both network targets below are Cloudflare's well-known public services (1.1.1.1),
 * chosen specifically because they're harmless, fast, and not going to look like
 * real exfiltration to anyone watching -- don't swap in anything else here.
 */
#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <stdio.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

static void banner(const char *label) {
    printf("\n[COVTEST_%s] attempting...\n", label);
    fflush(stdout);
}

int main(void) {
    printf("PID of this test process: %d (grep for this PID too, not just the markers)\n", getpid());

    /* 1. Baseline: plain connect() -- should be caught by security_socket_connect. */
    banner("CONNECT_BASELINE");
    {
        int s = socket(AF_INET, SOCK_STREAM, 0);
        struct sockaddr_in a = {0};
        a.sin_family = AF_INET;
        a.sin_port = htons(443);
        inet_pton(AF_INET, "1.1.1.1", &a.sin_addr);
        int rc = connect(s, (struct sockaddr *)&a, sizeof a);
        printf("[COVTEST_CONNECT_BASELINE] connect() returned %d (errno=%d) -- this one SHOULD show up.\n",
               rc, errno);
        close(s);
    }

    /* 2. The actual gap under test: sendto() on a socket that never called connect(). */
    banner("UDP_SENDTO_NO_CONNECT");
    {
        int s = socket(AF_INET, SOCK_DGRAM, 0);
        struct sockaddr_in a = {0};
        a.sin_family = AF_INET;
        a.sin_port = htons(53);
        inet_pton(AF_INET, "1.1.1.1", &a.sin_addr);
        const char msg[] = "covtest-udp-no-connect";
        ssize_t r = sendto(s, msg, sizeof msg - 1, 0, (struct sockaddr *)&a, sizeof a);
        printf("[COVTEST_UDP_SENDTO_NO_CONNECT] sendto() returned %zd (errno=%d) -- if the baseline "
               "above showed up but this one doesn't appear anywhere, that's the gap confirmed.\n",
               r, errno);
        close(s);
    }

    /* 3. execveat() instead of execve() -- genuinely unknown on this kernel until tested. */
    banner("EXECVEAT");
    {
        pid_t p = fork();
        if (p == 0) {
            char *const av[] = {"sh", "-c", "echo covtest-execveat-child-ran", NULL};
            char *const ev[] = {NULL};
            /* Bionic has exposed execveat() as a real libc function since API 21, so
             * this doesn't need a raw syscall() number -- deliberately avoided that
             * after realizing the numbers aren't safe to assume across architectures
             * without checking your own sysroot's <sys/syscall.h> first. */
            execveat(AT_FDCWD, "/system/bin/sh", av, ev, 0);
            _exit(127); /* only reached if execveat itself failed */
        }
        int st = 0;
        waitpid(p, &st, 0);
        printf("[COVTEST_EXECVEAT] child wait status: %d -- if CONNECT_BASELINE was caught but this "
               "never appears, execveat is a second real gap alongside plain UDP.\n", st);
    }

    printf("\nDone. As root: grep driftnetd's log / the dashboard for \"covtest\" and \"COVTEST\".\n");
    return 0;
}
