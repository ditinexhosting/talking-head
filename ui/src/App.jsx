import Header from "./components/Header";
import ChatPanel from "./components/ChatPanel";
import VideoFrame from "./components/VideoFrame";
import { useTalkingHead } from "./hooks/useTalkingHead";
import "./App.css";

function App() {
  const { status, sessionId, sendText, attachVideo, replay, canReplay } =
    useTalkingHead("ws://localhost:2000/video");

  return (
    <div className="h-screen w-screen flex flex-col bg-zinc-900 text-zinc-100 overflow-hidden">
      <Header status={status} sessionId={sessionId} />
      <main className="flex-1 min-h-0 flex flex-row">
        <VideoFrame attachVideo={attachVideo} />
        <ChatPanel
          status={status}
          onSend={sendText}
          onReplay={replay}
          canReplay={canReplay}
        />
      </main>
    </div>
  );
}

export default App;
