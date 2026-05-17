export default function Header({ status }) {
  const dot =
    status === "open" ? "bg-emerald-400" :
      status === "closed" ? "bg-red-500" :
        status === "connecting" ? "bg-amber-400 animate-pulse" :
          "bg-zinc-500";

  const label =
    status === "open" ? "Streaming" :
      status === "closed" ? "Disconnected" :
        status === "connecting" ? "Connecting…" :
          "Idle";

  return (
    <header className="h-[8vh] min-h-14 flex items-center justify-between px-6 border-b border-zinc-800 bg-zinc-950 text-zinc-100">
      <div className="flex items-center gap-3">
        <div className="h-8 w-8 rounded-md bg-gradient-to-br from-violet-500 to-fuchsia-500 grid place-items-center text-sm font-bold">
          TH
        </div>
        <div className="leading-tight">
          <div className="text-sm font-semibold">Talking Head</div>
          <div className="text-[11px] text-zinc-400">Real-time avatar</div>
        </div>
      </div>

      <div className="flex items-center gap-3 text-xs">
        <div className="flex items-center gap-2">
          <span className={`h-2 w-2 rounded-full ${dot}`} />
          <span className="text-zinc-300">{label}</span>
        </div>
      </div>
    </header>
  );
}
