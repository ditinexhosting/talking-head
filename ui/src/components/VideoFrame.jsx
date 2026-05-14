import aibg from "../assets/bg4.png";

export default function VideoFrame({ attachVideo }) {
  return (
    <div
      className="relative flex-1 min-w-0 flex items-center justify-center bg-zinc-900 bg-cover bg-center overflow-hidden"
      style={{ backgroundImage: `url(${aibg})` }}
    >
      <video
        ref={attachVideo}
        muted
        playsInline
        className="relative max-h-full max-w-full object-contain"
      />
    </div>
  );
}
