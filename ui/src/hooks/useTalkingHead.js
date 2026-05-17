import { useCallback, useEffect, useRef, useState } from "react";

const OFFER_URL = "/api/webrtc/offer";

const buildIceServers = () => {
  const servers = [];
  const stun = import.meta.env.VITE_STUN_URL || "stun:stun.l.google.com:19302";
  if (stun) servers.push({ urls: stun });

  const turnUrl = import.meta.env.VITE_TURN_URL;
  if (turnUrl) {
    servers.push({
      urls: turnUrl,
      username: import.meta.env.VITE_TURN_USERNAME,
      credential: import.meta.env.VITE_TURN_CREDENTIAL,
    });
  }
  return servers;
};

export const useTalkingHead = () => {
  const [status, setStatus] = useState("idle");
  const videoRef = useRef(null);
  const pcRef = useRef(null);

  const closePc = useCallback(() => {
    const pc = pcRef.current;
    if (!pc) return;
    pcRef.current = null;
    try {
      pc.getReceivers().forEach((r) => {
        try {
          r.track?.stop();
        } catch {
          /* track already stopped */
        }
      });
    } catch {
      /* receivers unavailable */
    }
    try {
      pc.close();
    } catch {
      /* pc already closed */
    }
    const v = videoRef.current;
    if (v) {
      try {
        v.srcObject = null;
      } catch {
        /* srcObject unsupported */
      }
    }
  }, []);

  const attachVideo = useCallback((node) => {
    videoRef.current = node;
  }, []);

  const waitForIceGathering = (pc) =>
    new Promise((resolve) => {
      if (pc.iceGatheringState === "complete") return resolve();
      const onChange = () => {
        if (pc.iceGatheringState === "complete") {
          pc.removeEventListener("icegatheringstatechange", onChange);
          resolve();
        }
      };
      pc.addEventListener("icegatheringstatechange", onChange);
    });

  const sendText = useCallback(
    (text) => {
      const trimmed = text.trim();
      if (!trimmed) return false;
      const videoEl = videoRef.current;
      if (!videoEl) return false;

      closePc();

      const iceServers = buildIceServers();
      console.log("[webrtc] iceServers:", iceServers);
      const pc = new RTCPeerConnection({ iceServers });
      pcRef.current = pc;
      setStatus("connecting");

      pc.addTransceiver("video", { direction: "recvonly" });

      pc.ontrack = (event) => {
        if (pcRef.current !== pc) return;
        console.log("[webrtc] ontrack:", event.track.kind);
        videoEl.srcObject = event.streams[0];
        videoEl.play().catch(() => {});
      };

      pc.onicecandidate = (e) => {
        if (e.candidate) {
          console.log("[webrtc] local candidate:", e.candidate.candidate);
        }
      };
      pc.onicecandidateerror = (e) => {
        console.warn(
          "[webrtc] candidate error:",
          e.errorCode,
          e.errorText,
          e.url
        );
      };
      pc.oniceconnectionstatechange = () => {
        console.log("[webrtc] iceConnectionState:", pc.iceConnectionState);
      };
      pc.onconnectionstatechange = () => {
        if (pcRef.current !== pc) return;
        const s = pc.connectionState;
        console.log("[webrtc] connectionState:", s);
        if (s === "connected") setStatus("open");
        else if (s === "failed" || s === "disconnected" || s === "closed") {
          setStatus("closed");
        }
      };

      (async () => {
        try {
          const offer = await pc.createOffer();
          await pc.setLocalDescription(offer);
          await waitForIceGathering(pc);

          const res = await fetch(OFFER_URL, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              sdp: pc.localDescription.sdp,
              type: pc.localDescription.type,
            }),
          });
          if (!res.ok) throw new Error(`offer http ${res.status}`);
          const answer = await res.json();
          console.log("[webrtc] remote SDP:\n", answer.sdp);
          if (pcRef.current !== pc) return;
          await pc.setRemoteDescription({
            sdp: answer.sdp,
            type: answer.type,
          });
        } catch (err) {
          console.error("[webrtc] negotiation failed:", err);
          if (pcRef.current === pc) {
            setStatus("closed");
            closePc();
          }
        }
      })();

      return true;
    },
    [closePc]
  );

  useEffect(() => closePc, [closePc]);

  return {
    status,
    sendText,
    attachVideo,
  };
};
