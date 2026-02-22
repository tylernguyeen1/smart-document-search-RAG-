import { useMemo, useState } from "react";

const API_BASE = "http://127.0.0.1:8000";

function App() {
  const [file, setFile] = useState(null);
  const [isDragOver, setIsDragOver] = useState(false);
  const [uploadStatus, setUploadStatus] = useState("");
  const [question, setQuestion] = useState("");
  const [answerFormat, setAnswerFormat] = useState("paragraph");
  const [summary, setSummary] = useState("");
  const [results, setResults] = useState([]);
  const [loadingUpload, setLoadingUpload] = useState(false);
  const [loadingAsk, setLoadingAsk] = useState(false);

  const fileLabel = useMemo(() => (file ? file.name : "Drop a PDF here"), [file]);
  const formatExcerpt = (text, maxChars = 700) => {
    const compact = (text || "").replace(/\s+/g, " ").trim();
    if (compact.length <= maxChars) return compact;
    return `${compact.slice(0, maxChars)}...`;
  };

  const onDrop = (event) => {
    event.preventDefault();
    setIsDragOver(false);
    const dropped = event.dataTransfer.files?.[0];
    if (!dropped) return;
    if (!dropped.name.toLowerCase().endsWith(".pdf")) {
      setUploadStatus("Only PDF files are allowed in this UI.");
      return;
    }
    setFile(dropped);
    setUploadStatus("");
  };

  const onFileInput = (event) => {
    const selected = event.target.files?.[0];
    if (!selected) return;
    if (!selected.name.toLowerCase().endsWith(".pdf")) {
      setUploadStatus("Only PDF files are allowed in this UI.");
      return;
    }
    setFile(selected);
    setUploadStatus("");
  };

  const uploadAndIndex = async () => {
    if (!file) {
      setUploadStatus("Select a PDF first.");
      return;
    }
    setLoadingUpload(true);
    setUploadStatus("Uploading and building index...");
    try {
      const formData = new FormData();
      formData.append("file", file);
      const res = await fetch(`${API_BASE}/upload-and-index`, {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.detail || "Upload failed.");
      }
      setUploadStatus(
        `Indexed ${data.metadata.count} chunks from ${data.file_name}.`
      );
    } catch (error) {
      setUploadStatus(error.message);
    } finally {
      setLoadingUpload(false);
    }
  };

  const askQuestion = async () => {
    if (!question.trim()) return;
    setLoadingAsk(true);
    setSummary("");
    setResults([]);
    try {
      const res = await fetch(`${API_BASE}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: question.trim(),
          index_dir: "data/index",
          top_k: 5,
          answer_format: answerFormat,
        }),
      });
      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.detail || "Query failed.");
      }
      setSummary(data.summary || "");
      setResults(data.results || []);
    } catch (error) {
      setSummary(error.message);
    } finally {
      setLoadingAsk(false);
    }
  };

  return (
    <main className="page">
      <h1>Smart Document Search</h1>

      <section className="card">
        <h2>1. Upload PDF</h2>
        <div
          className={`dropzone ${isDragOver ? "drag" : ""}`}
          onDragOver={(e) => {
            e.preventDefault();
            setIsDragOver(true);
          }}
          onDragLeave={() => setIsDragOver(false)}
          onDrop={onDrop}
        >
          <p>{fileLabel}</p>
          <input type="file" accept=".pdf" onChange={onFileInput} />
        </div>
        <button onClick={uploadAndIndex} disabled={loadingUpload}>
          {loadingUpload ? "Building..." : "Upload and Build Index"}
        </button>
        {uploadStatus ? <p className="status">{uploadStatus}</p> : null}
      </section>

      <section className="card">
        <h2>2. Ask Question</h2>
        <textarea
          rows={4}
          placeholder="Ask a question about the uploaded PDF..."
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
        />
        <label className="fieldLabel" htmlFor="answerFormat">
          Answer format
        </label>
        <select
          id="answerFormat"
          value={answerFormat}
          onChange={(e) => setAnswerFormat(e.target.value)}
          className="selectField"
        >
          <option value="paragraph">Summary paragraph</option>
          <option value="bullets">Bullet points</option>
        </select>
        <button onClick={askQuestion} disabled={loadingAsk}>
          {loadingAsk ? "Searching..." : "Ask"}
        </button>
      </section>

      <section className="card">
        <h2>Summary</h2>
        <p className="summaryText">{summary || "No answer yet."}</p>
      </section>

      <section className="card">
        <h2>Citations</h2>
        {results.length === 0 ? (
          <p>No citations yet.</p>
        ) : (
          results.map((item) => (
            <article key={item.chunk_id} className="citation">
              <strong>
                {item.file_name} ({item.chunk_id})
              </strong>
              <p>{formatExcerpt(item.text)}</p>
            </article>
          ))
        )}
      </section>
    </main>
  );
}

export default App;
