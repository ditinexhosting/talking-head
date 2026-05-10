import { useEffect, useRef, useState, useCallback } from "react";

export const useWebSocket = (url, onMessage) => {
  const wsRef = useRef(null);
  const onMessageRef = useRef(onMessage);
  const [status, setStatus] = useState("connecting");

  useEffect(() => { onMessageRef.current = onMessage; }, [onMessage]);

  useEffect(() => {
    const ws = new WebSocket(url);
    ws.binaryType = "arraybuffer";
    wsRef.current = ws;

    ws.onopen = () => setStatus("open");
    ws.onclose = () => setStatus("closed");
    ws.onmessage = (e) => onMessageRef.current(e.data);

    return () => ws.close();
  }, [url]);

  const send = useCallback((data) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(data);
    }
  }, []);

  return { send, status };
};
