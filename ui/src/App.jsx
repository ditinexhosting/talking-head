import Header from "./components/Header";
import VideoFrame from "./components/VideoFrame";
import ChatPanel from "./components/ChatPanel";
import { useTalkingHead } from "./hooks/useTalkingHead";
import "./App.css";

function App() {
  const {
    status,
    sessionId,
    isSpeaking,
    hasSentences,
    videoARef,
    videoBRef,
    sendText,
    replay,
  } = useTalkingHead("ws://localhost:2000/audio");

  return (
    <div className="h-screen w-screen flex flex-col bg-zinc-900 text-zinc-100 overflow-hidden">
      <Header status={status} sessionId={sessionId} />
      <main className="flex-1 min-h-0 flex flex-row">
        <VideoFrame
          videoARef={videoARef}
          videoBRef={videoBRef}
          isSpeaking={isSpeaking}
        />
        <ChatPanel
          status={status}
          isSpeaking={isSpeaking}
          hasSentences={hasSentences}
          onSend={sendText}
          onReplay={replay}
        />
      </main>
    </div>
  );
}

export default App;
