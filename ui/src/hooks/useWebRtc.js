import { useRef, useState, useCallback } from "react";

const BACKEND = "https://cargo.ditinex.com";

export function useWebRTC() {
  const videoRef = useRef(null);
  const pcRef = useRef(null);
  const [status, setStatus] = useState("idle"); // idle | connecting | connected | failed

  const start = useCallback(async () => {
    setStatus("connecting");

    const pc = new RTCPeerConnection({
      iceServers: [
        { urls: "stun:stun.l.google.com:19302" },
        {
          urls: "turn:51.178.52.75:3478",
          username: "webrtc",
          credential: "password",
        },
      ],
    });
    pcRef.current = pc;

    // When a track arrives, attach it to the <video>
    pc.ontrack = (e) => {
      if (videoRef.current) {
        videoRef.current.srcObject = e.streams[0];
      }
    };

    pc.onconnectionstatechange = () => {
      setStatus(pc.connectionState);
    };

    // Tell the server we want to receive video
    pc.addTransceiver("video", { direction: "recvonly" });

    // Create offer and send to FastAPI
    const offer = await pc.createOffer();
    await pc.setLocalDescription(offer);

    const res = await fetch(`${BACKEND}/offer`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sdp: offer.sdp, type: offer.type }),
    });

    const answer = await res.json();
    await pc.setRemoteDescription(answer);
  }, []);

  const stop = useCallback(() => {
    pcRef.current?.close();
    pcRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    setStatus("idle");
  }, []);

  return { videoRef, status, start, stop };
}
