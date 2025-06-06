// App.js
// Prompt
// Initial implementation generation:
// https://chatgpt.com/share/682ac8e1-941c-800d-a2e1-9dcdd6cbedec
// Support model selection via dropdown menu"
// https://chatgpt.com/share/682c2e64-04d0-800d-b736-67363d5924f1
// Add markdown rendering:
// https://chatgpt.com/canvas/shared/682c3c0767888191bb5a0834815cdb82
//
// PORT=18088 npm start --verbose

import React, { useState, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeHighlight from 'rehype-highlight'; // for syntax highlighting
import 'highlight.js/styles/github.css';      // choose a style or customize

import './App.css';

export default function App() {
  const [inText, setInText] = useState('');
  const [instruction, setInstruction] = useState('');
  const [models, setModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState('');
  const [output, setOutput] = useState('');
  const [code, setCode] = useState('');      
  const [loading, setLoading] = useState(false);
  const api_host = 'https://037e184c5f2a.ngrok.app';
  const codeact_endpoint = `${api_host}/tableqa/codeact`;
  const get_models_endpoint = `${api_host}/get_models`;

  // Fetch available models on mount
  useEffect(() => {
    const fetchModels = async () => {
      try {
        const res = await fetch(get_models_endpoint);
        const list = await res.json();
        setModels(list);
        if (list.length > 0) setSelectedModel(list[0]);
      } catch (err) {
        console.error('Failed to load models:', err);
      }
    };
    fetchModels();
  }, []);

  const handleSubmit = async () => {
    setLoading(true);
    try {
      const payload = {
        llm: selectedModel,
        in_text: tryParseJSON(inText),
        instruction,
      };
      const response = await fetch(
        codeact_endpoint, 
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
      const code = data.code;
      setCode(code);
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
        <h2>Model Selection</h2>
        {/* Model Dropdown */}
        <label htmlFor="model-select">Supported LLMs:</label>
        <select
          id="model-select"
          value={selectedModel}
          onChange={(e) => setSelectedModel(e.target.value)}
        >
          {models.map((m) => (
            <option key={m} value={m}>
              {m}
            </option>
          ))}
        </select>
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
        <div className="markdown-output">
          <ReactMarkdown remarkPlugins={[remarkGfm]}>
            {output}
          </ReactMarkdown>
        </div>
        <h2>Code</h2>
        <div className="markdown-output">
          <ReactMarkdown rehypePlugins={[rehypeHighlight]}>
            {code === '' ? '' : `\`\`\`python${code}\`\`\``}
          </ReactMarkdown>
        </div>
      </div>
    </div>
  );
}
