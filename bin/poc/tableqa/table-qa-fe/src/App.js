import React, { useState } from "react";
import { EventSourcePolyfill } from "event-source-polyfill";

const App = () => {
  const [inputText, setInputText] = useState("");
  const [instruction, setInstruction] = useState("");
  const [response, setResponse] = useState("");

  const handleSubmit = () => {
    const url = "https://your-llm-api-endpoint.com"; // Replace with your API URL
    const eventSource = new EventSourcePolyfill(url, {
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ input_text: inputText, instruction }),
      method: "POST",
    });

    setResponse(""); // Clear previous response

    eventSource.onmessage = (event) => {
      setResponse((prev) => prev + event.data);
    };

    eventSource.onerror = () => {
      eventSource.close();
      console.error("Error with EventSource.");
    };
  };

  return (
    <div style={{ display: "flex", height: "100vh" }}>
      <div style={{ flex: 1, padding: "1rem", borderRight: "1px solid #ccc" }}>
        <h3>LLM Input</h3>
        <div>
          <label>Input Text:</label>
          <textarea
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            style={{ width: "100%", height: "100px" }}
          />
        </div>
        <div>
          <label>Instruction:</label>
          <textarea
            value={instruction}
            onChange={(e) => setInstruction(e.target.value)}
            style={{ width: "100%", height: "100px" }}
          />
        </div>
        <button onClick={handleSubmit} style={{ marginTop: "1rem" }}>
          Submit
        </button>
      </div>
      <div style={{ flex: 2, padding: "1rem" }}>
        <h3>LLM Output</h3>
        <div
          style={{
            width: "100%",
            height: "calc(100% - 2rem)",
            border: "1px solid #ccc",
            padding: "1rem",
            overflowY: "auto",
          }}
        >
          {response}
        </div>
      </div>
    </div>
  );
};

export default App;

