//go:build linux
package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"log"
	"net/url"
	"os"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/cilium/ebpf"
	"github.com/gorilla/websocket"
)

type EventDetail struct {
	PID   int    `json:"pid"`
	UID   int    `json:"uid"`
	FName string `json:"fname"`
}

type Event struct {
	Device     string      `json:"device"`
	App        string      `json:"app"`
	Kind       string      `json:"kind"`
	Detail     EventDetail `json:"detail"`
	AppVersion string      `json:"app_version"`
}

type RelayClient struct {
	sync.Mutex
	conn *websocket.Conn
	url  string
}

func (r *RelayClient) Connect() {
	for {
		c, _, err := websocket.DefaultDialer.Dial(r.url, nil)
		if err == nil {
			r.Lock()
			r.conn = c
			r.Unlock()
			log.Printf("Connected to driftnetd at %s", r.url)
			go r.readLoop(c)
			return
		}
		log.Printf("wait for driftnetd at %s...", r.url)
		time.Sleep(2 * time.Second)
	}
}

func (r *RelayClient) Send(payload []byte) {
	for {
		r.Lock()
		c := r.conn
		r.Unlock()

		if c == nil {
			r.Connect()
			continue
		}

		err := c.WriteMessage(websocket.TextMessage, payload)
		if err == nil {
			return
		}

		log.Printf("Send error (%v), reconnecting...", err)
		r.Lock()
		if r.conn == c {
			c.Close()
			r.conn = nil
		}
		r.Unlock()
		time.Sleep(1 * time.Second)
		r.Connect()
	}
}

func (r *RelayClient) readLoop(c *websocket.Conn) {
	for {
		_, msg, err := c.ReadMessage()
		if err != nil {
			r.Lock()
			if r.conn == c {
				r.conn = nil
			}
			r.Unlock()
			c.Close()
			return
		}
		var resp struct {
			Action  string `json:"action"`
			Payload struct {
				PID int `json:"pid"`
			} `json:"payload"`
		}
		if err := json.Unmarshal(msg, &resp); err == nil && resp.Action == "BLOCK" {
			blacklistMap, err := ebpf.LoadPinnedMap("/sys/fs/bpf/blacklist_pids", nil)
			if err == nil {
				key := uint32(resp.Payload.PID)
				val := uint32(9) // SIGKILL
				blacklistMap.Put(key, val)
				blacklistMap.Close()
				syscall.Kill(resp.Payload.PID, syscall.SIGKILL)
				log.Printf("Ring-0 Kill Authorized: inserted PID %d into kernel blacklist map and delivered instant SIGKILL", key)
			} else {
				log.Printf("Failed to load pinned map: %v", err)
			}
		}
	}
}

func main() {
	targetURL := flag.String("url", "ws://127.0.0.1:8787/ws/ingest", "DriftNet ingest WebSocket URL")
	flag.Parse()

	rawURL := *targetURL
	if envURL := os.Getenv("DRIFTNET_URL"); envURL != "" {
		rawURL = envURL
	}

	u, err := url.Parse(rawURL)
	if err != nil {
		log.Fatalf("Invalid driftnet URL %q: %v", rawURL, err)
	}

	client := &RelayClient{url: u.String()}
	client.Connect()

	// PID: 123 | UID: 0 | COMM: test | FNAME: /etc/passwd
	re := regexp.MustCompile(`PID:\s*(\d+)\s*\|\s*UID:\s*(\d+)\s*\|\s*COMM:\s*(.*?)\s*\|\s*FNAME:\s*(.*)`)

	scanner := bufio.NewScanner(os.Stdin)
	for scanner.Scan() {
		line := scanner.Text()
		matches := re.FindStringSubmatch(line)
		if len(matches) == 5 {
			pid, _ := strconv.Atoi(matches[1])
			fname := strings.TrimSpace(matches[4])

			kind := "sys_openat"
			if strings.HasPrefix(fname, "IP:") {
				kind = "sys_connect"
			} else if strings.HasPrefix(fname, "EXEC:") {
				kind = "sys_execve"
				fname = strings.TrimPrefix(fname, "EXEC:")
			}

			comm := strings.TrimSpace(matches[3])
			uid, _ := strconv.Atoi(matches[2])
			ev := Event{
				Device:     "Android-Ring0",
				App:        comm,
				Kind:       kind,
				Detail:     EventDetail{PID: pid, UID: uid, FName: fname},
				AppVersion: "1.0",
			}

			payload, _ := json.Marshal(ev)
			client.Send(payload)
		}
	}
}
