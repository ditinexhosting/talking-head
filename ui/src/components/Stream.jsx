export default function Stream({ videoRef }) {
  return (
    <div className="relative flex-1 min-w-0 flex items-center justify-center bg-black overflow-hidden">
      <video
        ref={videoRef}
        autoPlay
        muted
        playsInline
        className="relative max-h-full max-w-full object-contain"
      />
    </div>
  );
}
