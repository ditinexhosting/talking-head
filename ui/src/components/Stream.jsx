import { useWebRTC } from "../hooks/useWebRtc";

export default function Stream() {
  const { videoRef, status, start, stop } = useWebRTC();

  return (
    <div style={{ padding: 24, fontFamily: "sans-serif" }}>
      <h2>WebRTC stream</h2>

      <video
        ref={videoRef}
        autoPlay
        playsInline
        muted
        style={{ width: 640, background: "#000", borderRadius: 8 }}
      />

      <div style={{ marginTop: 12, display: "flex", gap: 8, alignItems: "center" }}>
        <button onClick={start} disabled={status !== "idle"}>
          Start
        </button>
        <button onClick={stop} disabled={status === "idle"}>
          Stop
        </button>
        <span style={{ color: "#888", fontSize: 13 }}>Status: {status}</span>
      </div>
    </div>
  );
}