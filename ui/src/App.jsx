import Header from "./components/Header";
import ChatPanel from "./components/ChatPanel";
import VideoFrame from "./components/VideoFrame";
import Stream from "./components/Stream";
import { useTalkingHead } from "./hooks/useTalkingHead";
import "./App.css";

function App() {
  const { status, sendText, attachVideo } = useTalkingHead();

  return (
    <div className="h-screen w-screen flex flex-col bg-zinc-900 text-zinc-100 overflow-hidden">
      <Header status={status} />
      <main className="flex-1 min-h-0 flex flex-row">
        {/* <VideoFrame attachVideo={attachVideo} /> */}
        <Stream />
        <ChatPanel status={status} onSend={sendText} />
      </main>
    </div>
  );
}

export default App;
