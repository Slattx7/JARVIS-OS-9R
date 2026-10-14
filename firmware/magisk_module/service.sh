#!/system/bin/sh
# Project OMNI: Magisk Late-Start Service
# This script runs during late-boot to silently start the OMNI pipeline.

# Ensure we are in a clean state
killall -9 bpf_loader driftnetd llama-server 2>/dev/null
sleep 10 # Wait for Android to finish settling

echo "0" > /data/local/ubuntu/root/llm_state

# 1. Mount the Ubuntu chroot dependencies and start DriftNet Backend + UI
mount -t proc proc /data/local/ubuntu/proc
mount -t sysfs sysfs /data/local/ubuntu/sys
mount -t bpf bpf /data/local/ubuntu/sys/fs/bpf
mount --bind /dev /data/local/ubuntu/dev
mount --bind /dev/pts /data/local/ubuntu/dev/pts

# DriftNet connects to port 11434 and serves on port 8787
# main.go's own comment warns against 0.0.0.0 ("bind to your Tailscale interface,
# not 0.0.0.0, unless you've locked down the tailnet ACLs deliberately") and its flag
# default is 127.0.0.1. This boot script was overriding that safe default and binding
# wide open with zero authentication on every endpoint (CheckOrigin also accepts any
# origin). Restored to loopback-only; adb reverse/forward still reaches it from a
# laptop. If you deliberately want tailnet-wide access, replace 127.0.0.1 with your
# Tailscale interface IP, not 0.0.0.0, per main.go's own guidance.
chroot /data/local/ubuntu /bin/su - root -c "cd /root/driftnet && nohup ./bin/driftnetd -addr 127.0.0.1:8787 -data ./data -ollama http://127.0.0.1:11434 -model tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf -web ./frontend/out > /root/driftnet_nohup.out 2>&1 &"

# 2. Start the eBPF Kernel Probe -> Go WebSocket Relay Pipeline
chroot /data/local/ubuntu /bin/su - root -c "cd /root/loader && nohup sh -c './bpf_loader bpf_probe.o | ./bpf_relay' > /root/bpf_nohup.out 2>&1 &"

# 3. Watchdog for Lazy AI (Thermal & Battery Orchestration)
# DriftNet will write "1" to /root/llm_state when an anomaly occurs.
# This native loop watches that file and spins up the GPU natively.
LLM_DIR="/data/local/tmp/llama"
STATE_FILE="/data/local/ubuntu/root/llm_state"

while true; do
    if [ -f "$STATE_FILE" ]; then
        STATE=$(cat $STATE_FILE)
        if [ "$STATE" = "1" ]; then
            if ! pgrep -f llama-server >/dev/null; then
                # Wake up the heavy AI model
                LD_LIBRARY_PATH=$LLM_DIR $LLM_DIR/llama-server -m $LLM_DIR/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf --host 127.0.0.1 --port 11434 -c 2048 -cb -np 4 > /dev/null 2>&1 &
            fi
        elif [ "$STATE" = "0" ]; then
            if pgrep -f llama-server >/dev/null; then
                # Put the AI model to sleep
                killall -9 llama-server
            fi
        fi
    fi
    sleep 2
done &
