import { useCallback, useRef, useState } from "react";
import { useWebSocket } from "../hooks/useWebSocket";

export default function VideoPanel() {
  const [text, setText] = useState("");
  const [sessionId, setSessionId] = useState(null);
  const bufferRef = useRef(null);
  const videoRef = useRef(null);
  const videoBlobUrlRef = useRef(null);
  const [videoUrl, setVideoUrl] = useState(null);

  const handleMessage = useCallback((msg) => {
    if (msg instanceof ArrayBuffer) {
      const state = bufferRef.current;
      if (!state || msg.byteLength < 1) return;
      const tag = new Uint8Array(msg, 0, 1)[0];
      const payload = new Uint8Array(msg, 1);
      if (tag === 0x01) {
        state.audioReceived += payload.byteLength;
      } else if (tag === 0x02) {
        state.videoChunks.push(payload);
        state.videoReceived += payload.byteLength;
      }
      const audioDone = state.audioReceived >= state.meta.audio.size;
      const videoDone = !state.meta.video || state.videoReceived >= state.meta.video.size;
      if (audioDone && videoDone) {
        if (state.videoChunks.length) {
          if (videoBlobUrlRef.current) URL.revokeObjectURL(videoBlobUrlRef.current);
          const url = URL.createObjectURL(new Blob(state.videoChunks, { type: "video/mp4" }));
          videoBlobUrlRef.current = url;
          setVideoUrl(url);
        }
        bufferRef.current = null;
      }
      return;
    }

    const data = JSON.parse(msg);
    switch (data.event) {
      case "connected":
        setSessionId(data.session_id);
        break;
      case "buffer":
        bufferRef.current = {
          meta: data,
          audioReceived: 0,
          videoChunks: [], videoReceived: 0,
        };
        break;
      default:
        console.log("ws message:", data);
    }
  }, []);

  const { send, status } = useWebSocket("ws://localhost:2000/audio", handleMessage);

  const handleSend = useCallback(() => {
    if (!text.trim()) return;
    send(JSON.stringify({ "text": text }));
    setText("");
  }, [send, text]);

  return (
    <div className="flex flex-1 flex-col gap-4 p-4">
      <div className="flex items-center gap-2">
        <span className={`h-2.5 w-2.5 rounded-full ${status === "open" ? "bg-green-500" :
          status === "closed" ? "bg-red-500" :
            "bg-yellow-400 animate-pulse"
          }`} />
        <span className={`text-xs font-medium ${status === "open" ? "text-green-600" :
          status === "closed" ? "text-red-500" :
            "text-yellow-500"
          }`}>
          {status === "open" ? "Connected" : status === "closed" ? "Disconnected" : "Connecting…"}
        </span>
        {status === "open" && sessionId && (
          <span className="text-xs text-gray-400 font-mono font-bold"> : {sessionId}</span>
        )}
      </div>

      <video
        ref={videoRef}
        src={videoUrl ?? undefined}
        className="max-w-full rounded border border-gray-300 bg-black"
        width={533}
        height={462}
        playsInline
        autoPlay
      />

      <textarea
        className="w-full resize-none rounded border border-gray-300 p-2 text-sm"
        rows={3}
        placeholder="Enter text..."
        value={text}
        onChange={(e) => setText(e.target.value)}
      />

      <div className="flex gap-2">
        <button
          type="button"
          className="flex-1 rounded bg-blue-600 px-4 py-2 text-sm text-white hover:bg-blue-700 disabled:opacity-50"
          disabled={status !== "open"}
          onClick={handleSend}
        >
          Send
        </button>
        <button
          type="button"
          className="rounded bg-gray-200 px-4 py-2 text-sm text-gray-700 hover:bg-gray-300"
        >
          Replay
        </button>
      </div>
    </div>
  );
}
