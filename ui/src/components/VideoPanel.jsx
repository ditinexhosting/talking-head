import { useEffect, useRef, useState } from "react";

const WS_URL = "ws://localhost:2000/audio";
const KEYFRAMES_URL = "http://localhost:3000/animate/keyframes";

function pcmBase64ToAudioBuffer(base64, sampleRate, audioCtx) {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);

  const samples = new Int16Array(bytes.buffer);
  const float32 = new Float32Array(samples.length);
  for (let i = 0; i < samples.length; i++) float32[i] = samples[i] / 32768;

  const buffer = audioCtx.createBuffer(1, float32.length, sampleRate);
  buffer.copyToChannel(float32, 0);
  return buffer;
}

async function fetchVideoBlob(visemes) {
  const res = await fetch(KEYFRAMES_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ visemes }),
  });
  if (!res.ok) throw new Error(`Keyframes API error: ${res.status}`);
  return res.blob();
}

async function playPair(audioBuffer, videoBlob, displayEl, audioCtx) {
  const url = URL.createObjectURL(videoBlob);

  // Use a detached video element for preloading/timing; swap into display when ready
  const videoEl = document.createElement("video");
  videoEl.src = url;
  videoEl.muted = true;
  videoEl.preload = "auto";

  await new Promise((resolve, reject) => {
    videoEl.oncanplaythrough = resolve;
    videoEl.onerror = reject;
    videoEl.load();
  });

  // Show this video in the display element
  displayEl.src = url;
  displayEl.muted = true;
  await displayEl.load();

  const startAt = audioCtx.currentTime + 0.1;

  const source = audioCtx.createBufferSource();
  source.buffer = audioBuffer;
  source.connect(audioCtx.destination);
  source.start(startAt);

  const delay = (startAt - audioCtx.currentTime) * 1000;
  await new Promise((resolve) => setTimeout(resolve, delay));
  displayEl.play().catch(console.error);

  // Video is master clock — resolve when video ends (includes buffer frames)
  await new Promise((resolve, reject) => {
    displayEl.onended = resolve;
    displayEl.onerror = reject;
  });

  URL.revokeObjectURL(url);
}

export default function VideoPanel() {
  const wsRef = useRef(null);
  const audioCtxRef = useRef(null);
  const videoRef = useRef(null);
  const playChainRef = useRef(Promise.resolve());
  const lastPairsRef = useRef([]); // saved pairs from last run for replay
  const [status, setStatus] = useState("disconnected");
  const [text, setText] = useState("");
  const [loading, setLoading] = useState(false);
  const [replaying, setReplaying] = useState(false);
  const [hasPairs, setHasPairs] = useState(false);

  useEffect(() => {
    const ws = new WebSocket(WS_URL);
    wsRef.current = ws;

    ws.onopen = () => setStatus("connected");
    ws.onclose = () => setStatus("disconnected");
    ws.onerror = () => setStatus("error");

    ws.onmessage = (e) => {
      try {
        const msg = JSON.parse(e.data);

        if (msg.event === "audio") {
          const audioCtx = audioCtxRef.current;
          if (!audioCtx) return;

          const pairPromise = Promise.all([
            Promise.resolve(pcmBase64ToAudioBuffer(msg.data, msg.sample_rate, audioCtx)),
            fetchVideoBlob(msg.visemes),
          ]).then(([audioBuffer, videoBlob]) => ({ audioBuffer, videoBlob }));

          playChainRef.current = playChainRef.current
            .then(() => pairPromise)
            .then((pair) => {
              lastPairsRef.current.push(pair);
              setHasPairs(true);
              return playPair(pair.audioBuffer, pair.videoBlob, videoRef.current, audioCtxRef.current);
            })
            .catch(console.error);
        }

        if (msg.event === "done") {
          setLoading(false);
        }
      } catch {
        // ignore non-JSON messages
      }
    };

    return () => ws.close();
  }, []);

  function handleSend() {
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) return;

    if (!audioCtxRef.current) {
      audioCtxRef.current = new AudioContext();
    } else {
      audioCtxRef.current.resume();
    }

    playChainRef.current = Promise.resolve();
    lastPairsRef.current = [];
    setHasPairs(false);
    ws.send(JSON.stringify({ text }));
    setText("");
    setLoading(true);
  }

  function handleReplay() {
    const pairs = lastPairsRef.current;
    if (!pairs.length || replaying) return;

    audioCtxRef.current?.resume();
    setReplaying(true);

    let chain = Promise.resolve();
    for (const pair of pairs) {
      chain = chain.then(() =>
        playPair(pair.audioBuffer, pair.videoBlob, videoRef.current, audioCtxRef.current)
      );
    }
    chain.finally(() => setReplaying(false));
  }

  return (
    <div className="flex-1 flex flex-col gap-2">
      <p className="text-sm text-gray-400">
        Socket:{" "}
        <span
          className={
            status === "connected"
              ? "text-green-500"
              : status === "error"
                ? "text-red-500"
                : "text-yellow-500"
          }
        >
          {status}
        </span>
      </p>

      <video
        ref={videoRef}
        className="w-full rounded bg-black"
        muted
        playsInline
      />

      <div className="mt-auto flex flex-col gap-2">
        <textarea
          className="w-full resize-none rounded border border-gray-300 p-2 text-sm"
          rows={3}
          placeholder="Enter text..."
          value={text}
          onChange={(e) => setText(e.target.value)}
          disabled={loading}
        />
        <div className="flex gap-2">
        <button
          type="button"
          className="flex flex-1 items-center justify-center gap-2 rounded bg-blue-600 px-4 py-2 text-sm text-white hover:bg-blue-700 disabled:opacity-50"
          disabled={status !== "connected" || !text.trim() || loading || replaying}
          onClick={handleSend}
        >
          {loading && (
            <svg className="h-4 w-4 animate-spin" viewBox="0 0 24 24" fill="none">
              <circle
                className="opacity-25"
                cx="12"
                cy="12"
                r="10"
                stroke="currentColor"
                strokeWidth="4"
              />
              <path
                className="opacity-75"
                fill="currentColor"
                d="M4 12a8 8 0 018-8v4l3-3-3-3v4a8 8 0 00-8 8h4z"
              />
            </svg>
          )}
          {loading ? "Sending..." : "Send"}
        </button>
        <button
          type="button"
          className="rounded bg-gray-200 px-4 py-2 text-sm text-gray-700 hover:bg-gray-300 disabled:opacity-50"
          disabled={!hasPairs || loading || replaying}
          onClick={handleReplay}
        >
          {replaying ? "Replaying..." : "Replay"}
        </button>
        </div>
      </div>
    </div>
  );
}
