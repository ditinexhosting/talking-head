import { useEffect, useRef, useState } from "react";

export default function ChatPanel({
  status,
  isSpeaking,
  hasSentences,
  onSend,
  onReplay,
}) {
  const [text, setText] = useState("");
  const [messages, setMessages] = useState([]);
  const scrollRef = useRef(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages, isSpeaking]);

  const handleSend = () => {
    const trimmed = text.trim();
    if (!trimmed || status !== "open") return;
    if (onSend(trimmed)) {
      setMessages((prev) => [
        ...prev,
        { role: "user", text: trimmed, id: Date.now() },
      ]);
      setText("");
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <aside className="w-full max-w-md border-l border-zinc-800 bg-zinc-950 text-zinc-100 flex flex-col">
      <div className="px-4 py-3 border-b border-zinc-800 flex items-center justify-between">
        <div className="text-sm font-semibold">Conversation</div>
        <button
          type="button"
          className="text-xs px-2 py-1 rounded text-zinc-300 hover:text-white hover:bg-zinc-800 disabled:opacity-40 disabled:hover:bg-transparent"
          disabled={!hasSentences}
          onClick={onReplay}
        >
          Replay
        </button>
      </div>

      <div ref={scrollRef} className="flex-1 overflow-y-auto px-4 py-4 space-y-3">
        {messages.length === 0 && (
          <div className="h-full flex items-center justify-center text-center text-sm text-zinc-500">
            <div>
              <div className="mb-1 text-zinc-300">Start a conversation</div>
              <div className="text-xs">Type something for the avatar to speak.</div>
            </div>
          </div>
        )}

        {messages.map((m) => (
          <div
            key={m.id}
            className={`flex ${m.role === "user" ? "justify-end" : "justify-start"}`}
          >
            <div
              className={`max-w-[85%] rounded-2xl px-3.5 py-2 text-sm leading-relaxed whitespace-pre-wrap ${
                m.role === "user"
                  ? "bg-violet-600 text-white rounded-br-sm"
                  : "bg-zinc-800 text-zinc-100 rounded-bl-sm"
              }`}
            >
              {m.text}
            </div>
          </div>
        ))}

        {isSpeaking && (
          <div className="flex justify-start">
            <div className="bg-zinc-800 rounded-2xl rounded-bl-sm px-3.5 py-2 text-sm text-zinc-300 flex items-center gap-1.5">
              <span className="h-1.5 w-1.5 rounded-full bg-zinc-400 animate-bounce" style={{ animationDelay: "0ms" }} />
              <span className="h-1.5 w-1.5 rounded-full bg-zinc-400 animate-bounce" style={{ animationDelay: "120ms" }} />
              <span className="h-1.5 w-1.5 rounded-full bg-zinc-400 animate-bounce" style={{ animationDelay: "240ms" }} />
            </div>
          </div>
        )}
      </div>

      <div className="p-3 border-t border-zinc-800">
        <div className="flex items-end gap-2 rounded-xl bg-zinc-900 ring-1 ring-zinc-800 focus-within:ring-violet-500 px-3 py-2">
          <textarea
            className="flex-1 resize-none bg-transparent outline-none text-sm placeholder:text-zinc-500 max-h-32"
            rows={1}
            placeholder={status === "open" ? "Type a message…" : "Connecting…"}
            value={text}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={status !== "open"}
          />
          <button
            type="button"
            onClick={handleSend}
            disabled={status !== "open" || !text.trim()}
            className="shrink-0 h-8 px-3 rounded-lg bg-violet-600 text-white text-sm font-medium hover:bg-violet-500 disabled:opacity-40 disabled:hover:bg-violet-600"
          >
            Send
          </button>
        </div>
      </div>
    </aside>
  );
}
