import { useState, useRef, useEffect } from 'react'
import ReactMarkdown from 'react-markdown'
import './App.css'

const SYSTEM_PROMPT = "You are a helpful AI assistant powered by Llama 3.3 70B. Be concise but thorough."

function App() {
  const [messages, setMessages] = useState([])  // Each message can have { role, content, metrics? }
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [serverStatus, setServerStatus] = useState('checking')
  const [modelInfo, setModelInfo] = useState(null)
  const [showMetrics, setShowMetrics] = useState(true)  // Toggle metrics display
  const messagesEndRef = useRef(null)
  const inputRef = useRef(null)

  // Check server status on mount
  useEffect(() => {
    checkServerStatus()
  }, [])

  // Auto-scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  // Focus input after loading
  useEffect(() => {
    if (!isLoading) {
      inputRef.current?.focus()
    }
  }, [isLoading])

  const checkServerStatus = async () => {
    try {
      const response = await fetch('/api/models')
      if (response.ok) {
        const data = await response.json()
        setModelInfo(data)
        setServerStatus('connected')
      } else {
        setServerStatus('error')
      }
    } catch (error) {
      setServerStatus('disconnected')
    }
  }

  const sendMessage = async (e) => {
    e.preventDefault()
    if (!input.trim() || isLoading) return

    const userMessage = { role: 'user', content: input.trim() }
    setMessages(prev => [...prev, userMessage])
    setInput('')
    setIsLoading(true)

    // Prepare messages with system prompt
    const allMessages = [
      { role: 'system', content: SYSTEM_PROMPT },
      ...messages,
      userMessage,
    ]

    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          messages: allMessages,
          max_tokens: 1024,
          temperature: 0.7,
          stream: true,
        }),
      })

      if (!response.ok) throw new Error('Failed to get response')

      // Handle SSE streaming
      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let assistantMessage = ''
      let messageMetrics = null

      // Add empty assistant message
      setMessages(prev => [...prev, { role: 'assistant', content: '' }])

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        const chunk = decoder.decode(value)
        const lines = chunk.split('\n')

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6))
              if (data.token) {
                assistantMessage += data.token
                setMessages(prev => {
                  const newMessages = [...prev]
                  newMessages[newMessages.length - 1] = {
                    role: 'assistant',
                    content: assistantMessage,
                  }
                  return newMessages
                })
              }
              // Capture metrics when done
              if (data.done && data.metrics) {
                messageMetrics = data.metrics
                setMessages(prev => {
                  const newMessages = [...prev]
                  newMessages[newMessages.length - 1] = {
                    role: 'assistant',
                    content: assistantMessage,
                    metrics: messageMetrics,
                  }
                  return newMessages
                })
              }
              if (data.error) {
                throw new Error(data.error)
              }
            } catch (parseError) {
              // Ignore parse errors for incomplete chunks
            }
          }
        }
      }
    } catch (error) {
      console.error('Error:', error)
      setMessages(prev => [
        ...prev,
        {
          role: 'assistant',
          content: `⚠️ Error: ${error.message}. Make sure the backend server is running.`,
        },
      ])
    } finally {
      setIsLoading(false)
    }
  }

  const clearChat = () => {
    setMessages([])
  }

  return (
    <div className="app">
      {/* Header */}
      <header className="header">
        <div className="header-left">
          <span className="logo">🦙</span>
          <div className="header-title">
            <h1>Llama 3.3 70B</h1>
            <span className="header-subtitle">
              {modelInfo ? `${modelInfo.current} quantization` : 'Local AI Chat'}
            </span>
          </div>
        </div>
        <div className="header-right">
          <div className={`status-badge status-${serverStatus}`}>
            <span className="status-dot"></span>
            {serverStatus === 'connected' ? 'Connected' : 
             serverStatus === 'checking' ? 'Checking...' : 'Disconnected'}
          </div>
          <button 
            className={`toggle-btn ${showMetrics ? 'active' : ''}`} 
            onClick={() => setShowMetrics(!showMetrics)}
            title={showMetrics ? 'Hide performance metrics' : 'Show performance metrics'}
          >
            📊
          </button>
          {messages.length > 0 && (
            <button className="clear-btn" onClick={clearChat}>
              Clear Chat
            </button>
          )}
        </div>
      </header>

      {/* Messages */}
      <main className="messages-container">
        {messages.length === 0 ? (
          <div className="welcome">
            <div className="welcome-icon">🦙</div>
            <h2>Welcome to Llama Chat</h2>
            <p>
              Chat with Meta's Llama 3.3 70B model running locally on your Mac.
              {modelInfo && ` Using ${modelInfo.current} quantization.`}
            </p>
            <div className="welcome-suggestions">
              <button onClick={() => setInput('Explain quantum computing in simple terms')}>
                Explain quantum computing
              </button>
              <button onClick={() => setInput('Write a Python function to sort a list')}>
                Write a sorting function
              </button>
              <button onClick={() => setInput('What are the benefits of meditation?')}>
                Benefits of meditation
              </button>
            </div>
          </div>
        ) : (
          <div className="messages">
            {messages.map((message, index) => (
              <div key={index} className={`message message-${message.role}`}>
                <div className="message-avatar">
                  {message.role === 'user' ? '👤' : '🦙'}
                </div>
                <div className="message-content">
                  <div className="message-role">
                    {message.role === 'user' ? 'You' : 'Llama'}
                  </div>
                  <div className="message-text">
                    {message.role === 'assistant' ? (
                      <ReactMarkdown>{message.content || '...'}</ReactMarkdown>
                    ) : (
                      message.content
                    )}
                  </div>
                  {/* Performance metrics */}
                  {message.metrics && showMetrics && (
                    <div className="message-metrics">
                      <span className="metric">
                        <span className="metric-icon">⚡</span>
                        {message.metrics.tokens_per_second} tok/s
                      </span>
                      <span className="metric">
                        <span className="metric-icon">📝</span>
                        {message.metrics.completion_tokens} tokens
                      </span>
                      <span className="metric">
                        <span className="metric-icon">⏱️</span>
                        {(message.metrics.total_time_ms / 1000).toFixed(1)}s
                      </span>
                      <span className="metric metric-detail">
                        prompt: {(message.metrics.prompt_eval_time_ms / 1000).toFixed(1)}s
                      </span>
                      <span className="metric metric-detail">
                        gen: {(message.metrics.completion_time_ms / 1000).toFixed(1)}s
                      </span>
                    </div>
                  )}
                </div>
              </div>
            ))}
            {isLoading && messages[messages.length - 1]?.role !== 'assistant' && (
              <div className="message message-assistant">
                <div className="message-avatar">🦙</div>
                <div className="message-content">
                  <div className="message-role">Llama</div>
                  <div className="typing-indicator">
                    <span></span><span></span><span></span>
                  </div>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>
        )}
      </main>

      {/* Input */}
      <footer className="input-container">
        <form onSubmit={sendMessage} className="input-form">
          <input
            ref={inputRef}
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder={serverStatus === 'connected' ? 'Ask Llama anything...' : 'Waiting for server...'}
            disabled={isLoading || serverStatus !== 'connected'}
            className="input-field"
          />
          <button
            type="submit"
            disabled={!input.trim() || isLoading || serverStatus !== 'connected'}
            className="send-btn"
          >
            {isLoading ? (
              <span className="loading-spinner"></span>
            ) : (
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M22 2L11 13M22 2l-7 20-4-9-9-4 20-7z"/>
              </svg>
            )}
          </button>
        </form>
        <div className="input-footer">
          <span>Powered by llama.cpp • Running locally on Apple Silicon</span>
        </div>
      </footer>
    </div>
  )
}

export default App
