import Header from "./components/Header";
import ChatPanel from "./components/ChatPanel";
import Stream from "./components/Stream";
import { useLiveKit } from "./hooks/useLiveKit";
import "./App.css";

function App() {
  const { videoRef, status, sendText } = useLiveKit();

  return (
    <div className="h-screen w-screen flex flex-col bg-zinc-900 text-zinc-100 overflow-hidden">
      <Header status={status} />
      <main className="flex-1 min-h-0 flex flex-row">
        <Stream videoRef={videoRef} />
        <ChatPanel status={status} onSend={sendText} />
      </main>
    </div>
  );
}

export default App;
