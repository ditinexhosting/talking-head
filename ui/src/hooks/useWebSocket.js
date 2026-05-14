import { useEffect, useRef, useState, useCallback } from "react";

export const useWebSocket = (url, onMessage) => {
  const wsRef = useRef(null);
  const onMessageRef = useRef(onMessage);
  const lastArrivalRef = useRef(0);
  const [status, setStatus] = useState("connecting");

  useEffect(() => { onMessageRef.current = onMessage; }, [onMessage]);

  useEffect(() => {
    const ws = new WebSocket(url);
    ws.binaryType = "arraybuffer";
    wsRef.current = ws;

    ws.onopen = () => setStatus("open");
    ws.onclose = () => setStatus("closed");
    ws.onmessage = (e) => {
      const now = performance.now();
      const gap = lastArrivalRef.current ? now - lastArrivalRef.current : 0;
      lastArrivalRef.current = now;
      const isBin = e.data instanceof ArrayBuffer;
      const size = isBin ? e.data.byteLength : e.data.length;
      console.log(`[ws] ${isBin ? "bin" : "txt"} gap=${gap.toFixed(1)}ms size=${size}`);
      onMessageRef.current(e.data);
    };

    return () => ws.close();
  }, [url]);

  const send = useCallback((data) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(data);
    }
  }, []);

  return { send, status };
};
