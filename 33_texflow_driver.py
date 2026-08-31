#!/usr/bin/env python3
"""MCP stdio driver for texflow. Usage: import txflow; call Tool."""
import json
import subprocess
import threading
import queue

TEXFLOW = "/home/touhid/Documents/texflowmcp/texflow-mcp-main"
SERVER = ["uv", "--directory", TEXFLOW, "run", "texflow"]

class Tx:
    def __init__(self, workspace):
        self.proc = subprocess.Popen(
            SERVER + [workspace], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, bufsize=1)
        self.q = queue.Queue()
        self.err = []
        threading.Thread(target=self._pump, daemon=True).start()
        self._id = 0
        self._init()

    def _pump(self):
        for line in self.proc.stdout:
            self.q.put(line.strip())

    def _read(self, want_id, timeout=120):
        import time
        t0 = time.time()
        while time.time() - t0 < timeout:
            try:
                line = self.q.get(timeout=1)
            except queue.Empty:
                continue
            if not line:
                continue
            try:
                msg = json.loads(line)
            except Exception:
                continue
            if msg.get("id") == want_id:
                return msg
        raise TimeoutError(f"no response for id {want_id}")

    def _rpc(self, method, params=None):
        self._id += 1
        req = {"jsonrpc": "2.0", "id": self._id, "method": method}
        if params is not None:
            req["params"] = params
        self.proc.stdin.write(json.dumps(req) + "\n")
        self.proc.stdin.flush()
        return self._read(self._id)

    def _init(self):
        r = self._rpc("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "gridpower-builder", "version": "1.0"}})
        self.proc.stdin.write(json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}) + "\n")
        self.proc.stdin.flush()
        return r

    def call(self, tool, args):
        r = self._rpc("tools/call", {"name": tool, "arguments": args})
        if "error" in r:
            raise RuntimeError(f"{tool}: {r['error']}")
        content = r["result"].get("content", [])
        texts = [c.get("text", "") for c in content if c.get("type") == "text"]
        return "\n".join(texts)

    def close(self):
        self.proc.terminate()

if __name__ == "__main__":
    import sys
    ws = sys.argv[1] if len(sys.argv) > 1 else "/home/touhid/Documents/texflowmcp/workspace-gridpower"
    t = Tx(ws)
    tools = t._rpc("tools/list")
    names = [x["name"] for x in tools["result"]["tools"]]
    print("TOOLS:", names)
    print("--- create ---")
    print(t.call("document", {"action": "create", "document_class": "ieee-conference",
                              "title": "GridPowerAgent: A Reproducible Synthetic Benchmark Corpus for Evaluating Grid-Aware LLM Agents on IEEE 14/39/118 Test Systems",
                              "author": "Touhid Siddique Raj"}))
    print("--- outline ---")
    print(t.call("document", {"action": "outline"}))
    t.close()
