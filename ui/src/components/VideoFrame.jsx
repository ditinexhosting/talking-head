import aibg from "../assets/aibg.jpg";

export default function VideoFrame({ videoARef, videoBRef, isSpeaking }) {
  return (
    <div
      className="flex-1 min-w-0 flex items-end justify-center bg-zinc-900 bg-cover bg-center"
      style={{ backgroundImage: `url(${aibg})` }}
    >
      <div className="relative aspect-square w-full max-w-105 overflow-hidden bg-black shadow-2xl">
        <video
          ref={videoARef}
          className="absolute inset-0 h-full w-full object-cover"
          style={{ opacity: 1 }}
          playsInline
          muted
        />
        <video
          ref={videoBRef}
          className="absolute inset-0 h-full w-full object-cover"
          style={{ opacity: 0 }}
          playsInline
          muted
        />

        <div className="absolute top-3 left-3 flex items-center gap-2 px-2.5 py-1 rounded-full bg-black/50 backdrop-blur-sm text-xs text-white">
          <span className={`h-1.5 w-1.5 rounded-full ${isSpeaking ? "bg-emerald-400 animate-pulse" : "bg-zinc-500"}`} />
          {isSpeaking ? "Speaking" : "Idle"}
        </div>
      </div>
    </div>
  );
}
