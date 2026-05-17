import { useLiveKit } from "../hooks/useLiveKit";

const STATUS_COLOR = {
  idle: "#888",
  connecting: "#f59e0b",
  connected: "#22c55e",
  failed: "#ef4444",
};

export default function Stream() {
  const { videoRef, status, start, stop } = useLiveKit();

  return (
    <div style={{ padding: 24, fontFamily: "sans-serif", flex: 1 }}>
      <h2 style={{ marginBottom: 16 }}>LiveKit stream</h2>

      <video
        ref={videoRef}
        autoPlay
        playsInline
        muted
        style={{ width: "100%", maxWidth: 720, background: "#000", borderRadius: 8, display: "block" }}
      />

      <div style={{ marginTop: 12, display: "flex", gap: 8, alignItems: "center" }}>
        <button onClick={start} disabled={status !== "idle" && status !== "failed"}>
          Start
        </button>
        <button onClick={stop} disabled={status === "idle"}>
          Stop
        </button>
        <span style={{ color: STATUS_COLOR[status] ?? "#888", fontSize: 13, fontWeight: 500 }}>
          ● {status}
        </span>
      </div>
    </div>
  );
}