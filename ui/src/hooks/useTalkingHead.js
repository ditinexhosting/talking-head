import { useCallback, useEffect, useRef, useState } from "react";
import { useWebSocket } from "./useWebSocket";

const decodePcmS16LE = (buf, sampleRate, ctx) => {
  const view = new DataView(buf);
  const len = buf.byteLength / 2;
  const audioBuffer = ctx.createBuffer(1, len, sampleRate);
  const channel = audioBuffer.getChannelData(0);
  for (let i = 0; i < len; i++) channel[i] = view.getInt16(i * 2, true) / 32768;
  return audioBuffer;
};

export const useTalkingHead = (url) => {
  const [sessionId, setSessionId] = useState(null);
  const [hasSentences, setHasSentences] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);

  const currentRef = useRef(null);
  const sentencesRef = useRef([]);
  const playIndexRef = useRef(0);
  const playingRef = useRef(false);
  const activeSourceRef = useRef(null);
  const audioCtxRef = useRef(null);
  const videoUrlsRef = useRef({ a: null, b: null });
  const videoARef = useRef(null);
  const videoBRef = useRef(null);
  const activeAbRef = useRef("a");
  const playNextRef = useRef(null);

  const ensureAudioCtx = useCallback(() => {
    if (!audioCtxRef.current) {
      audioCtxRef.current = new (window.AudioContext || window.webkitAudioContext)();
    }
    if (audioCtxRef.current.state === "suspended") audioCtxRef.current.resume();
    return audioCtxRef.current;
  }, []);

  const playNext = useCallback(async () => {
    if (playingRef.current) return;
    const sentence = sentencesRef.current[playIndexRef.current];
    if (!sentence) {
      setIsSpeaking(false);
      return;
    }
    playIndexRef.current += 1;
    playingRef.current = true;
    setIsSpeaking(true);

    const ctx = ensureAudioCtx();
    const audioBuffer = decodePcmS16LE(sentence.audio, sentence.audioMeta.sample_rate, ctx);

    const source = ctx.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(ctx.destination);
    activeSourceRef.current = source;

    let audioEnded = false;
    let videoEnded = false;

    const finish = () => {
      if (audioEnded && videoEnded) {
        playingRef.current = false;
        activeSourceRef.current = null;
        playNextRef.current?.();
      }
    };

    source.onended = () => { audioEnded = true; finish(); };

    const nextKey = activeAbRef.current === "a" ? "b" : "a";
    const back = nextKey === "a" ? videoARef.current : videoBRef.current;
    const front = nextKey === "a" ? videoBRef.current : videoARef.current;
    if (sentence.videoBlob && back) {
      if (videoUrlsRef.current[nextKey]) URL.revokeObjectURL(videoUrlsRef.current[nextKey]);
      const url = URL.createObjectURL(sentence.videoBlob);
      videoUrlsRef.current[nextKey] = url;
      back.onended = () => { videoEnded = true; finish(); };
      back.src = url;
      await new Promise((resolve) => {
        const onReady = () => { back.removeEventListener("loadeddata", onReady); resolve(); };
        back.addEventListener("loadeddata", onReady, { once: true });
      });
      back.style.opacity = "1";
      if (front) front.style.opacity = "0";
      activeAbRef.current = nextKey;
      try {
        await back.play();
      } catch {
        videoEnded = true;
      }
    } else {
      videoEnded = true;
    }
    // 8f/25fps audio delay — video needs a few frames to start.
    source.start(ctx.currentTime + 8 / 25);
    if (audioEnded || videoEnded) finish();
  }, [ensureAudioCtx]);

  useEffect(() => {
    playNextRef.current = playNext;
  }, [playNext]);

  const replay = useCallback(() => {
    if (!sentencesRef.current.length) return;
    ensureAudioCtx();
    if (activeSourceRef.current) {
      try { activeSourceRef.current.onended = null; activeSourceRef.current.stop(); } catch { /* noop */ }
      activeSourceRef.current = null;
    }
    for (const v of [videoARef.current, videoBRef.current]) {
      if (v) { v.onended = null; v.pause(); }
    }
    playingRef.current = false;
    playIndexRef.current = 0;
    playNext();
  }, [ensureAudioCtx, playNext]);

  const handleMessage = useCallback((msg) => {
    if (msg instanceof ArrayBuffer) {
      const state = currentRef.current;
      if (!state || msg.byteLength < 1) return;
      const tag = new Uint8Array(msg, 0, 1)[0];
      const payload = new Uint8Array(msg.slice(1));
      if (tag === 0x01) {
        state.audioChunks.push(payload);
        state.audioReceived += payload.byteLength;
      } else if (tag === 0x02) {
        state.videoChunks.push(payload);
        state.videoReceived += payload.byteLength;
      }
      return;
    }

    const data = JSON.parse(msg);
    switch (data.event) {
      case "connected":
        setSessionId(data.session_id);
        break;
      case "chunk_start":
        currentRef.current = {
          meta: data,
          audioChunks: [], audioReceived: 0,
          videoChunks: [], videoReceived: 0,
        };
        break;
      case "chunk_end": {
        const s = currentRef.current;
        currentRef.current = null;
        if (!s) break;
        const audioBlob = new Blob(s.audioChunks);
        const videoBlob = s.videoChunks.length ? new Blob(s.videoChunks, { type: "video/mp4" }) : null;
        audioBlob.arrayBuffer().then((ab) => {
          sentencesRef.current.push({
            audio: ab,
            audioMeta: s.meta.audio,
            videoBlob,
          });
          setHasSentences(true);
          playNextRef.current?.();
        });
        break;
      }
      default:
        console.log("ws message:", data);
    }
  }, []);

  const { send, status } = useWebSocket(url, handleMessage);

  const sendText = useCallback((text) => {
    const trimmed = text.trim();
    if (!trimmed) return false;
    ensureAudioCtx();
    send(JSON.stringify({ text: trimmed }));
    playingRef.current = false;
    activeSourceRef.current = null;
    sentencesRef.current = [];
    playIndexRef.current = 0;
    return true;
  }, [send, ensureAudioCtx]);

  return {
    status,
    sessionId,
    isSpeaking,
    hasSentences,
    videoARef,
    videoBRef,
    sendText,
    replay,
  };
};
