import { useCallback, useEffect, useRef, useState } from "react";
import { useWebSocket } from "./useWebSocket";

const CODEC = 'video/mp4; codecs="avc1.42E01E"';
const START_BUFFER_SECONDS = 4.0;
const BUFFER_EVICT_KEEP = 3.5;
const STATS_INTERVAL_MS = 1000;

export const useTalkingHead = (url) => {
  const [sessionId, setSessionId] = useState(null);
  const [canReplay, setCanReplay] = useState(false);

  const videoRef = useRef(null);
  const mediaSourceRef = useRef(null);
  const sourceBufferRef = useRef(null);
  const queueRef = useRef([]);
  const objectUrlRef = useRef(null);
  const hasDataRef = useRef(false);
  const playbackStartedRef = useRef(false);
  const streamEndedRef = useRef(false);
  const statsRef = useRef({
    chunkCount: 0,
    byteCount: 0,
    prevBufferedEnd: 0,
    waitStartMs: 0,
  });

  const evaluatePlayback = useCallback(() => {
    const v = videoRef.current;
    const sb = sourceBufferRef.current;
    if (!v || !sb || sb.buffered.length === 0) return;
    const end = sb.buffered.end(sb.buffered.length - 1);
    const ahead = end - v.currentTime;
    const ended = streamEndedRef.current;

    if (!playbackStartedRef.current && ahead >= START_BUFFER_SECONDS) {
      playbackStartedRef.current = true;
      console.log(`[mse] starting playback buffer=${ahead.toFixed(2)}s ended=${ended}`);
      v.play().catch(() => {});
    }
  }, []);

  const pump = useCallback(() => {
    const sb = sourceBufferRef.current;
    if (!sb || sb.updating || queueRef.current.length === 0) return;
    const chunk = queueRef.current[0];
    try {
      sb.appendBuffer(chunk);
      queueRef.current.shift();
    } catch (err) {
      if (err && err.name === "QuotaExceededError") {
        const v = videoRef.current;
        const keepFrom = Math.max(0, (v?.currentTime ?? 0) - BUFFER_EVICT_KEEP);
        if (keepFrom > 0 && !sb.updating) {
          try {
            sb.remove(0, keepFrom);
          } catch (removeErr) {
            console.error("[mse] remove failed:", removeErr);
          }
        }
      } else {
        console.error("[mse] appendBuffer failed:", err);
        queueRef.current.shift();
      }
    }
  }, []);

  const setupMediaSource = useCallback(
    (video) => {
      if (!MediaSource.isTypeSupported(CODEC)) {
        console.error("[mse] codec not supported:", CODEC);
        return;
      }
      const mediaSource = new MediaSource();
      mediaSourceRef.current = mediaSource;
      const objectUrl = URL.createObjectURL(mediaSource);
      objectUrlRef.current = objectUrl;
      video.autoplay = false;
      video.src = objectUrl;

      video.addEventListener("waiting", () => {
        statsRef.current.waitStartMs = performance.now();
        const buf = video.buffered;
        console.warn(
          `[stream] starved: currentTime=${video.currentTime.toFixed(2)}s ` +
            `bufferedEnd=${(buf.length ? buf.end(buf.length - 1) : 0).toFixed(2)}s ` +
            `queue=${queueRef.current.length}`
        );
      });
      video.addEventListener("playing", () => {
        if (statsRef.current.waitStartMs) {
          const gap = performance.now() - statsRef.current.waitStartMs;
          console.log(`[stream] resumed after ${gap.toFixed(0)}ms wait`);
          statsRef.current.waitStartMs = 0;
        }
      });
      mediaSource.addEventListener("sourceopen", () => {
        if (mediaSourceRef.current !== mediaSource || mediaSource.readyState !== "open") {
          return;
        }
        let sb;
        try {
          sb = mediaSource.addSourceBuffer(CODEC);
        } catch (err) {
          console.warn("[mse] addSourceBuffer skipped:", err);
          return;
        }
        sb.mode = "sequence";
        sourceBufferRef.current = sb;
        sb.addEventListener("updateend", () => {
          if (!hasDataRef.current) {
            hasDataRef.current = true;
            setCanReplay(true);
          }
          evaluatePlayback();
          pump();
        });
        pump();
      });
    },
    [pump, evaluatePlayback]
  );

  const teardownMediaSource = useCallback(() => {
    const ms = mediaSourceRef.current;
    const sb = sourceBufferRef.current;
    mediaSourceRef.current = null;
    sourceBufferRef.current = null;
    if (sb && ms) {
      try {
        if (sb.updating) sb.abort();
        ms.removeSourceBuffer(sb);
      } catch {
        /* sb already removed or ms not open */
      }
    }
    if (ms && ms.readyState === "open") {
      try {
        ms.endOfStream();
      } catch {
        /* ms transitioned before we got here */
      }
    }
    if (objectUrlRef.current) {
      URL.revokeObjectURL(objectUrlRef.current);
      objectUrlRef.current = null;
    }
    queueRef.current = [];
    hasDataRef.current = false;
    playbackStartedRef.current = false;
    streamEndedRef.current = false;
    setCanReplay(false);
  }, []);

  const attachVideo = useCallback(
    (node) => {
      if (videoRef.current === node) return;
      if (videoRef.current) teardownMediaSource();
      videoRef.current = node;
      if (node) setupMediaSource(node);
    },
    [setupMediaSource, teardownMediaSource]
  );

  const handleMessage = useCallback(
    (msg) => {
      if (msg instanceof ArrayBuffer) {
        statsRef.current.chunkCount += 1;
        statsRef.current.byteCount += msg.byteLength;
        queueRef.current.push(new Uint8Array(msg));
        pump();
        return;
      }

      try {
        const data = JSON.parse(msg);
        switch (data.event) {
          case "connected":
            setSessionId(data.session_id);
            break;
          case "frame_chunk_start":
            streamEndedRef.current = false;
            console.log("[stream] frame chunk started");
            break;
          case "done":
          case "end":
          case "stream_end":
          case "complete":
          case "frame_chunk_end":
            console.log("[stream] end-of-stream signal:", data.event);
            streamEndedRef.current = true;
            evaluatePlayback();
            break;
          default:
            break;
        }
      } catch (err) {
        console.error("[ws] bad text msg:", err);
      }
    },
    [pump, evaluatePlayback]
  );

  const { send, status } = useWebSocket(url, handleMessage);

  useEffect(() => {
    if (status === "closed") {
      streamEndedRef.current = true;
      evaluatePlayback();
    }
  }, [status, evaluatePlayback]);

  useEffect(() => {
    let prevWall = performance.now();
    const id = setInterval(() => {
      const now = performance.now();
      const wallDelta = (now - prevWall) / 1000;
      prevWall = now;
      if (wallDelta <= 0) return;

      const s = statsRef.current;
      const fps = s.chunkCount / wallDelta;
      const kbps = (s.byteCount * 8) / 1000 / wallDelta;
      s.chunkCount = 0;
      s.byteCount = 0;

      const v = videoRef.current;
      let bufferAhead = 0;
      let videoRate = 0;
      if (v && v.buffered.length > 0) {
        const end = v.buffered.end(v.buffered.length - 1);
        bufferAhead = end - v.currentTime;
        videoRate = (end - s.prevBufferedEnd) / wallDelta;
        s.prevBufferedEnd = end;
      }

      console.log(
        `[stream] rx=${fps.toFixed(1)} chunks/s ${kbps.toFixed(0)} kbps | ` +
          `video=${videoRate.toFixed(2)}x realtime | ` +
          `bufferAhead=${bufferAhead.toFixed(2)}s queue=${queueRef.current.length}`
      );
    }, STATS_INTERVAL_MS);
    return () => clearInterval(id);
  }, []);

  const replay = useCallback(() => {
    const v = videoRef.current;
    if (!v || v.buffered.length === 0) return;
    v.currentTime = v.buffered.start(0);
    v.play().catch(() => {});
  }, []);

  const sendText = useCallback(
    (text) => {
      const trimmed = text.trim();
      if (!trimmed) return false;
      streamEndedRef.current = false;
      send(JSON.stringify({ visemes: [] }));
      return true;
    },
    [send]
  );

  return {
    status,
    sessionId,
    sendText,
    attachVideo,
    replay,
    canReplay,
  };
};
