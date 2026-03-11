import { useState, useEffect, useRef } from 'react';
import LandingScreen from './components/LandingScreen';
import UploadScreen from './components/UploadScreen';
import ProcessingScreen from './components/ProcessingScreen';
import ResultsScreen from './components/ResultsScreen';
import HistoryScreen from './components/HistoryScreen';
import ErrorScreen from './components/ErrorScreen';
import AuthScreen from './components/AuthScreen';
import SettingsScreen from './components/SettingsScreen';
import { api, setAuthToken } from './services/api';
import { Settings, LogOut } from 'lucide-react';
import './index.css';

function App() {
  const [currentView, setCurrentView] = useState('landing'); // Landing is first

  // User State
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [userEmail, setUserEmail] = useState('');

  // Upload State
  const [selectedFile, setSelectedFile] = useState(null);
  const [selectedLanguage, setSelectedLanguage] = useState('');
  const [jobId, setJobId] = useState(null);

  // Result State
  const [result, setResult] = useState({
    status: '',
    text: '',
    confidence: null,
    wordCount: 0,
    processingTimeMs: 0,
    error: null
  });

  // History State
  const [history, setHistory] = useState({
    jobs: [],
    currentPage: 1,
    totalPages: 1
  });

  // Global Error
  const [errorMsg, setErrorMsg] = useState('');

  // Polling Cancellation
  const pollingActive = useRef(false);

  // Automatically check token
  useEffect(() => {
    const token = localStorage.getItem('access_token');
    if (token) {
      setIsLoggedIn(true);
      setAuthToken(token);
    }
  }, []);

  const handleLogout = () => {
    localStorage.removeItem('access_token');
    setAuthToken(null);
    setIsLoggedIn(false);
    resetFlow();
    setCurrentView('auth');
  };

  const handleError = (msg) => {
    setErrorMsg(msg);
    setCurrentView('error');
  };

  // === View Handlers ===
  const handleStart = () => {
    if (isLoggedIn) {
      setCurrentView('upload');
    } else {
      setCurrentView('auth');
    }
  };

  const handleUploadSubmit = async (file, language) => {
    try {
      setSelectedFile(file);
      setSelectedLanguage(language);
      setCurrentView('processing');
      pollingActive.current = true;

      const res = await api.uploadDocument(file, language);
      setJobId(res.job_id);

      // Start polling
      pollResult(res.job_id);
    } catch (err) {
      handleError(err.message || 'Failed to upload document.');
    }
  };

  const cancelProcessing = () => {
    pollingActive.current = false;
    resetFlow();
  };

  const pollResult = async (id) => {
    try {
      let currentStatus = 'processing';
      while (currentStatus === 'processing' && pollingActive.current) {
        const res = await api.pollResult(id);
        currentStatus = res.status;

        if (currentStatus === 'completed') {
          setResult({
            status: res.status,
            text: res.text,
            confidence: res.confidence,
            wordCount: res.word_count || res.text.split(' ').length,
            processingTimeMs: res.processing_time_ms || 0,
            error: null
          });
          setCurrentView('result');
          return;
        } else if (currentStatus === 'failed') {
          throw new Error(res.error || 'Processing failed');
        }

        // Wait 2 seconds
        await new Promise(resolve => setTimeout(resolve, 2000));
      }
    } catch (err) {
      handleError(err.message || 'Error checking result status.');
    }
  };

  const resetFlow = () => {
    setSelectedFile(null);
    setJobId(null);
    setResult({ status: '', text: '', confidence: null, wordCount: 0, processingTimeMs: 0, error: null });
    setCurrentView('upload');
  };

  const navigateToHistory = () => setCurrentView('history');
  const navigateToSettings = () => setCurrentView('settings');

  return (
    <div className="app-container">
      {/* Top Header / Navigation could go here */}
      <header className="flex justify-between items-center" style={{ padding: 'var(--space-2) var(--space-4)', borderBottom: '1px solid var(--color-border)' }}>
        <h2 style={{ color: 'var(--color-accent)', cursor: 'pointer' }} onClick={() => isLoggedIn ? setCurrentView('landing') : null}>
          Akshara OCR
        </h2>
        {currentView !== 'auth' && (
          <div className="flex gap-2">
            <button className="btn-secondary" onClick={() => setCurrentView('upload')}>New Extract</button>
            <button className="btn-secondary" onClick={navigateToHistory}>History</button>
            <button className="btn-secondary" onClick={navigateToSettings} style={{ padding: '8px' }} aria-label="Settings">
              <Settings size={20} />
            </button>
          </div>
        )}
      </header>

      <main>
        {currentView === 'landing' && <LandingScreen onStart={handleStart} />}
        {currentView === 'auth' && <AuthScreen onLoginSuccess={() => { setIsLoggedIn(true); setCurrentView('upload'); }} />}
        {currentView === 'upload' && <UploadScreen onUploadSubmit={handleUploadSubmit} />}
        {currentView === 'processing' && <ProcessingScreen onCancel={cancelProcessing} />}
        {currentView === 'result' && <ResultsScreen result={result} selectedLanguage={selectedLanguage} selectedFile={selectedFile} onBack={resetFlow} />}
        {currentView === 'history' && <HistoryScreen />}
        {currentView === 'settings' && <SettingsScreen onLogout={handleLogout} />}
        {currentView === 'error' && <ErrorScreen message={errorMsg} onRetry={resetFlow} />}
      </main>
    </div>
  );
}

export default App;
