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
import { demoCases } from './data/demoCases';
import { Settings } from 'lucide-react';
import './index.css';

function App() {
  const [currentView, setCurrentView] = useState('landing'); // Landing is first

  // User State
  const [isLoggedIn, setIsLoggedIn] = useState(false);

  // Upload State
  const [selectedFile, setSelectedFile] = useState(null);
  const [selectedLanguage, setSelectedLanguage] = useState('');

  // Result State
  const [result, setResult] = useState({
    status: '',
    text: '',
    confidence: null,
    wordCount: 0,
    processingTimeMs: 0,
    error: null
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
      while ((currentStatus === 'processing' || currentStatus === 'queued') && pollingActive.current) {
        const res = await api.pollResult(id);
        currentStatus = res.status;

        if (currentStatus === 'done') {
          setResult({
            status: res.status,
            text: res.text,
            confidence: res.confidence,
            wordCount: res.word_count || (res.text ? res.text.split(' ').length : 0),
            processingTimeMs: res.processing_time_ms || 0,
            error: null
          });
          setCurrentView('result');
          return;
        } else if (currentStatus === 'error') {
          throw new Error(res.text || res.message || 'Processing failed');
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
        {currentView === 'landing' && <LandingScreen onStart={handleStart} demoCases={demoCases} />}
        {currentView === 'auth' && <AuthScreen onLoginSuccess={() => { setIsLoggedIn(true); setCurrentView('upload'); }} />}
        {currentView === 'upload' && <UploadScreen onUploadSubmit={handleUploadSubmit} demoCases={demoCases} />}
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
