import { useRef, useState, useCallback } from "react";
import { Room, RoomEvent, Track } from "livekit-client";

const VPS_URL = "https://cargo.ditinex.com";

function randomUserId() {
  return "user-" + Math.random().toString(36).slice(2, 10);
}

export function useLiveKit() {
  const videoRef = useRef(null);
  const roomRef = useRef(null);
  const [status, setStatus] = useState("idle"); // idle | connecting | connected | failed

  const attachExistingTracks = useCallback((room) => {
    for (const participant of room.remoteParticipants.values()) {
      for (const pub of participant.trackPublications.values()) {
        if (pub.track?.kind === Track.Kind.Video && videoRef.current) {
          pub.track.attach(videoRef.current);
        }
      }
    }
  }, []);

  const start = useCallback(async () => {
    if (roomRef.current) return;
    setStatus("connecting");

    try {
      const res = await fetch(`${VPS_URL}/session`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ user_id: randomUserId() }),
      });

      if (!res.ok) throw new Error(`Session failed: ${res.status}`);
      const { token, livekit_url } = await res.json();

      const room = new Room();
      roomRef.current = room;

      room.on(RoomEvent.TrackSubscribed, (track) => {
        if (track.kind === Track.Kind.Video && videoRef.current) {
          track.attach(videoRef.current);
        }
      });

      room.on(RoomEvent.TrackUnsubscribed, (track) => {
        track.detach();
      });

      room.on(RoomEvent.Connected, () => {
        setStatus("connected");
        attachExistingTracks(room);
      });

      room.on(RoomEvent.Disconnected, () => {
        setStatus("idle");
        roomRef.current = null;
      });

      room.on(RoomEvent.ConnectionStateChanged, (state) => {
        console.log("[livekit] connection state:", state);
      });

      await room.connect(livekit_url, token);
    } catch (err) {
      console.error("[livekit] failed:", err);
      setStatus("failed");
      roomRef.current = null;
    }
  }, [attachExistingTracks]);

  const stop = useCallback(async () => {
    const room = roomRef.current;
    if (room) {
      await room.disconnect();
      roomRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setStatus("idle");
  }, []);

  return { videoRef, status, start, stop };
}
