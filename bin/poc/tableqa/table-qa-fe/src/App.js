// App.js
// Prompt
// https://g.co/gemini/share/728ad3025014
//
// PORT=18088 npm start

import React, { useState } from 'react';
import './App.css';

export default function App() {
  const [inText, setInText] = useState('');
  const [instruction, setInstruction] = useState('');
  const [output, setOutput] = useState('');
  const [code, setCode] = useState('');      
  const [loading, setLoading] = useState(false);
  const endpoint = 'https://037e184c5f2a.ngrok.app/tableqa/codeact';

  const handleSubmit = async () => {
    setLoading(true);
    try {
      const payload = {
        in_text: tryParseJSON(inText),
        instruction,
      };
      const response = await fetch(
        endpoint, 
        {
          method: 'POST',
          headers: {'Content-Type': 'application/json',},
          body: JSON.stringify(payload),
        }
      );
      const data = await response.json();
      console.log(data);
      const lastMsg =
        data.msgs && data.msgs.length > 0
          ? data.msgs[data.msgs.length - 1].content
          : '';
      setOutput(lastMsg);
    } catch (err) {
      console.error(err);
      setOutput('Error: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  const tryParseJSON = (str) => {
    try {
      return JSON.parse(str);
    } catch {
      return str;
    }
  };

  return (
    <div className="app-container">
      <div className="panel input-panel">
        <h2>Input</h2>
        <textarea
          value={inText}
          onChange={(e) => setInText(e.target.value)}
          placeholder="Enter in_text (JSON or string)"
        />
        <input
          type="text"
          value={instruction}
          onChange={(e) => setInstruction(e.target.value)}
          placeholder="Enter instruction"
        />
        <button onClick={handleSubmit} disabled={loading}>
          {loading ? 'Loading...' : 'Submit'}
        </button>
      </div>

      <div className="panel output-panel">
        <h2>Output</h2>
        <pre>{output}</pre>
      </div>
    </div>
  );
}
