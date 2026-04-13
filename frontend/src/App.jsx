import { useState, useEffect, useRef } from 'react';
import LandingScreen from './components/LandingScreen';
import UploadScreen from './components/UploadScreen';
import ProcessingScreen from './components/ProcessingScreen';
import ResultsScreen from './components/ResultsScreen';
import HistoryScreen from './components/HistoryScreen';
import ErrorScreen from './components/ErrorScreen';
import SettingsScreen from './components/SettingsScreen';
import { localHistory } from './services/localHistory';
import { demoCases } from './data/demoCases';
import { Settings, History, PlusSquare } from 'lucide-react';
import { runLocalInference } from './services/onnxInference';
import './index.css';

function App() {
  const [currentView, setCurrentView] = useState('landing');
  const [selectedFile, setSelectedFile] = useState(null);
  const [selectedLanguage, setSelectedLanguage] = useState('hin');
  const [errorMsg, setErrorMsg] = useState('');
  
  // OCR Result State
  const [result, setResult] = useState({
    status: '',
    text: '',
    confidence: null,
    wordCount: 0,
    processingTimeMs: 0,
    error: null
  });

  const pollingActive = useRef(false);

  const handleError = (msg) => {
    setErrorMsg(msg);
    setCurrentView('error');
  };

  // === View Handlers ===
  const handleStart = () => {
    setCurrentView('upload');
  };

  const handleUploadSubmit = async (file) => {
    try {
      setSelectedFile(file);
      // Hardcoded Hindi focus
      setSelectedLanguage('hin');
      setCurrentView('processing');
      pollingActive.current = true;

      // EXECUTING CAPACITOR/REACT NATIVE ONNX ENGINE OFFLINE (WebGPU enabled)
      const res = await runLocalInference(file, '/vocab.json', '/model.onnx'); 
      
      if (res.status === 'done') {
          setResult({
            status: res.status,
            text: res.text,
            confidence: res.confidence,
            wordCount: res.text ? res.text.trim().split(/\s+/).length : 0,
            processingTimeMs: res.processingTimeMs,
            error: null
          });

          // Save to local history automatically (Hardcoded 'hin')
          await localHistory.saveResult(res, file, 'hin');
          
          setCurrentView('result');
      }
    } catch (err) {
      handleError(err.message || 'Failed to extract text offline.');
    }
  };

  const cancelProcessing = () => {
    pollingActive.current = false;
    resetFlow();
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
      <header className="flex justify-between items-center" style={{ padding: 'var(--space-2) var(--space-4)', borderBottom: '1px solid var(--color-border)' }}>
        <h2 style={{ color: 'var(--color-accent)', cursor: 'pointer' }} onClick={() => setCurrentView('landing')}>
          Akshara OCR
        </h2>
        
        <div className="flex gap-2">
          <button className="icon-btn" onClick={() => setCurrentView('upload')} title="New Extractions">
            <PlusSquare size={20} />
          </button>
          <button className="icon-btn" onClick={navigateToHistory} title="Local History">
            <History size={20} />
          </button>
          <button className="icon-btn" onClick={navigateToSettings} title="Settings">
            <Settings size={20} />
          </button>
        </div>
      </header>

      <main>
        {currentView === 'landing' && <LandingScreen onStart={handleStart} demoCases={demoCases} />}
        {currentView === 'upload' && <UploadScreen onUploadSubmit={handleUploadSubmit} demoCases={demoCases} />}
        {currentView === 'processing' && <ProcessingScreen onCancel={cancelProcessing} />}
        {currentView === 'result' && (
          <ResultsScreen 
            result={result} 
            selectedLanguage={selectedLanguage} 
            selectedFile={selectedFile} 
            onBack={resetFlow} 
          />
        )}
        {currentView === 'history' && <HistoryScreen />}
        {currentView === 'settings' && <SettingsScreen />}
        {currentView === 'error' && <ErrorScreen message={errorMsg} onRetry={resetFlow} />}
      </main>
    </div>
  );
}

export default App;
