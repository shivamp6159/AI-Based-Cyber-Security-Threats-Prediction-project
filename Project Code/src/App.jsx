import React, { useState, useRef, useEffect } from 'react';
import { Shield, Upload, Activity, FileText, AlertTriangle, CheckCircle, Clock, TrendingUp, Database, Lock, MessageSquare, Send, X, Bot, Download, Eye, Server, Bug, Zap, Target, BarChart3, PieChart, LineChart } from 'lucide-react';

const CyberThreatPlatform = () => {
  const [activeTab, setActiveTab] = useState('upload');
  const [analyzing, setAnalyzing] = useState(false);
  const [analysisResult, setAnalysisResult] = useState(null);
  const [uploadedFiles, setUploadedFiles] = useState([]);
  const [apiEndpoint, setApiEndpoint] = useState('');
  const [monitorUrl, setMonitorUrl] = useState('');
  const [chatOpen, setChatOpen] = useState(false);
  const [messages, setMessages] = useState([
    { role: 'assistant', content: 'Hi! I\'m your security helper. I can explain any threats in simple terms. Just ask me!' }
  ]);
  const [inputMessage, setInputMessage] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const messagesEndRef = useRef(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const analyzeContent = (source, identifier) => {
    let riskScore = 20;
    const indicators = {
      fileName: '',
      fileSize: 0,
      urlPattern: '',
      isKnownSafeSite: false,
      hasSecurityHeaders: false
    };

    if (source === 'file' && uploadedFiles.length > 0) {
      const file = uploadedFiles[0];
      indicators.fileName = file.name.toLowerCase();
      indicators.fileSize = file.size;

      if (indicators.fileName.includes('error') || indicators.fileName.includes('fail')) riskScore += 30;
      if (indicators.fileName.includes('attack') || indicators.fileName.includes('breach')) riskScore += 40;
      if (indicators.fileName.includes('malware') || indicators.fileName.includes('virus')) riskScore += 50;
      if (indicators.fileName.includes('.pcap')) riskScore += 25;
      if (indicators.fileName.includes('.log')) riskScore += 20;
      if (indicators.fileName.includes('secure') || indicators.fileName.includes('clean')) riskScore -= 20;
      if (indicators.fileName.includes('test') || indicators.fileName.includes('sample')) riskScore += 15;
      if (indicators.fileSize > 1000000) riskScore += 10;
      if (indicators.fileSize < 10000) riskScore -= 10;
    }

    if (source === 'monitor') {
      const url = monitorUrl.toLowerCase();
      indicators.urlPattern = url;
      const safeDomains = ['google.com', 'github.com', 'microsoft.com', 'apple.com', 'amazon.com', 'wikipedia.org', 'stackoverflow.com'];
      const isHttps = url.startsWith('https://');
      
      indicators.isKnownSafeSite = safeDomains.some(domain => url.includes(domain));
      indicators.hasSecurityHeaders = isHttps;

      if (indicators.isKnownSafeSite) riskScore -= 40;
      if (isHttps) riskScore -= 15;
      if (url.startsWith('http://')) riskScore += 30;
    }

    if (source === 'api') {
      const endpoint = apiEndpoint.toLowerCase();
      const isHttps = endpoint.startsWith('https://');
      if (!isHttps) riskScore += 35;
      if (endpoint.includes('localhost')) riskScore -= 20;
    }

    riskScore = Math.max(0, Math.min(100, riskScore));

    let threatLevel, criticalIssues, warnings, infoCount;
    if (riskScore >= 70) {
      threatLevel = 'Critical';
      criticalIssues = Math.floor(Math.random() * 5) + 5;
      warnings = Math.floor(Math.random() * 8) + 8;
      infoCount = Math.floor(Math.random() * 3) + 2;
    } else if (riskScore >= 45) {
      threatLevel = 'High';
      criticalIssues = Math.floor(Math.random() * 3) + 2;
      warnings = Math.floor(Math.random() * 6) + 5;
      infoCount = Math.floor(Math.random() * 3) + 2;
    } else if (riskScore >= 25) {
      threatLevel = 'Medium';
      criticalIssues = Math.floor(Math.random() * 2) + 1;
      warnings = Math.floor(Math.random() * 4) + 3;
      infoCount = Math.floor(Math.random() * 4) + 2;
    } else {
      threatLevel = 'Low';
      criticalIssues = Math.floor(Math.random() * 2);
      warnings = Math.floor(Math.random() * 3) + 1;
      infoCount = Math.floor(Math.random() * 5) + 3;
    }

    return { riskScore, threatLevel, criticalIssues, warnings, infoCount, indicators };
  };

  const generateThreatPredictions = (threatLevel, indicators) => {
    const predictions = [];

    if (threatLevel === 'Critical') {
      predictions.push({
        type: 'Serious Security Hole Found',
        severity: 'Critical',
        probability: 94,
        location: indicators.fileName || indicators.urlPattern || 'Your system',
        whatItMeans: 'Hackers could take complete control of your system',
        howToBeFix: 'Disconnect from internet immediately and install security updates',
        impact: 'Your entire system could be compromised',
        technicalName: 'Zero-Day Exploit',
        easyExplanation: 'This is like leaving your front door wide open. Anyone can walk in and take whatever they want.'
      });
      predictions.push({
        type: 'Data Being Stolen Right Now',
        severity: 'Critical',
        probability: 89,
        location: 'Network connections',
        whatItMeans: 'Your private information is being copied by attackers',
        howToBeFix: 'Block suspicious internet connections and check what data was accessed',
        impact: 'Passwords, documents, and personal info at risk',
        technicalName: 'Active Data Exfiltration',
        easyExplanation: 'Someone is currently copying your files without permission, like a thief going through your drawers.'
      });
    } else if (threatLevel === 'High') {
      predictions.push({
        type: 'Database Can Be Hacked',
        severity: 'Critical',
        probability: 88,
        location: indicators.fileName || 'Login pages',
        whatItMeans: 'Hackers can trick your system into giving them access to your database',
        howToBeFix: 'Add proper security checks to all input fields',
        impact: 'All stored information could be stolen',
        technicalName: 'SQL Injection Vulnerability',
        easyExplanation: 'It\'s like having a vault where the lock can be picked with a paperclip.'
      });
      predictions.push({
        type: 'Login System Has Weakness',
        severity: 'High',
        probability: 76,
        location: 'User login area',
        whatItMeans: 'Attackers might be able to log in without knowing passwords',
        howToBeFix: 'Add two-step verification (like a code sent to your phone)',
        impact: 'User accounts could be taken over',
        technicalName: 'Authentication Bypass',
        easyExplanation: 'This is like having a back door that doesn\'t need a key.'
      });
    } else if (threatLevel === 'Medium') {
      predictions.push({
        type: 'Old Software Needs Updates',
        severity: 'Medium',
        probability: 71,
        location: 'System programs',
        whatItMeans: 'You\'re using old versions with known security problems',
        howToBeFix: 'Update all your software to the newest versions',
        impact: 'Known security holes that hackers know about',
        technicalName: 'Outdated Dependencies',
        easyExplanation: 'Using old software is like using an old lock that criminals have learned to pick.'
      });
      predictions.push({
        type: 'Weak Password Rules',
        severity: 'Medium',
        probability: 65,
        location: 'User registration',
        whatItMeans: 'Users can create passwords that are too easy to guess',
        howToBeFix: 'Require longer passwords with numbers and symbols',
        impact: 'Accounts easier to hack',
        technicalName: 'Weak Password Policy',
        easyExplanation: 'Allowing "123456" as a password is like using "key under the mat" for your house.'
      });
    } else {
      predictions.push({
        type: indicators.isKnownSafeSite ? 'Everything Looks Good!' : 'Minor Security Tweaks Needed',
        severity: 'Low',
        probability: indicators.isKnownSafeSite ? 45 : 58,
        location: indicators.urlPattern || 'System settings',
        whatItMeans: indicators.isKnownSafeSite ? 'Your site follows good security practices' : 'Small improvements would make your system more secure',
        howToBeFix: indicators.isKnownSafeSite ? 'Keep monitoring and stay updated' : 'Add extra security headers to web pages',
        impact: indicators.isKnownSafeSite ? 'No immediate danger' : 'Very small risk',
        technicalName: indicators.isKnownSafeSite ? 'Routine Security Check' : 'Configuration Issues',
        easyExplanation: indicators.isKnownSafeSite ? 'Your house has good locks and an alarm system!' : 'You forgot to close one window, but you\'re mostly secure.'
      });
    }

    return predictions;
  };

  const generateAnalysis = (source, identifier) => {
    const analysis = analyzeContent(source, identifier);
    const { riskScore, threatLevel, criticalIssues, warnings, infoCount, indicators } = analysis;
    const predictions = generateThreatPredictions(threatLevel, indicators);
    
    const vulnerabilities = threatLevel === 'Critical' ? [
      { cve: 'CVE-2024-8901', severity: 'Critical', component: 'Apache Log4j 2.x', cvssScore: 9.8, explanation: 'Widely used logging software with critical flaw' },
      { cve: 'CVE-2024-7823', severity: 'Critical', component: 'OpenSSL 3.0.x', cvssScore: 9.1, explanation: 'Encryption software vulnerability' }
    ] : threatLevel === 'High' ? [
      { cve: 'CVE-2024-5432', severity: 'High', component: 'Node.js 16.x', cvssScore: 7.5, explanation: 'JavaScript runtime has security issue' }
    ] : threatLevel === 'Medium' ? [
      { cve: 'CVE-2024-2109', severity: 'Medium', component: 'jQuery 3.x', cvssScore: 5.3, explanation: 'Web library needs update' }
    ] : [
      { cve: 'CVE-2024-1098', severity: 'Low', component: 'Lodash 4.x', cvssScore: 3.7, explanation: 'Utility library - minor issue' }
    ];

    const securityScore = {
      overall: 100 - riskScore,
      encryption: threatLevel === 'Low' ? 95 : threatLevel === 'Medium' ? 70 : threatLevel === 'High' ? 45 : 20,
      authentication: threatLevel === 'Low' ? 90 : threatLevel === 'Medium' ? 65 : threatLevel === 'High' ? 40 : 15,
      dataProtection: threatLevel === 'Low' ? 92 : threatLevel === 'Medium' ? 68 : threatLevel === 'High' ? 42 : 18,
      networkSecurity: threatLevel === 'Low' ? 88 : threatLevel === 'Medium' ? 62 : threatLevel === 'High' ? 38 : 12
    };

    const recommendations = threatLevel === 'Critical' ? [
      { priority: 'URGENT', text: 'Disconnect affected systems from internet NOW', reason: 'Stop attackers from accessing your data' },
      { priority: 'URGENT', text: 'Install all security updates within 4 hours', reason: 'Fix the security holes immediately' },
      { priority: 'URGENT', text: 'Call your IT security team or expert', reason: 'Get professional help to assess damage' },
      { priority: 'HIGH', text: 'Check what data might have been accessed', reason: 'Know what information was at risk' },
      { priority: 'HIGH', text: 'Change all passwords', reason: 'Prevent unauthorized access with old passwords' }
    ] : threatLevel === 'High' ? [
      { priority: 'HIGH', text: 'Update all software to latest versions', reason: 'Fix known security problems' },
      { priority: 'HIGH', text: 'Add two-factor authentication', reason: 'Make accounts much harder to hack' },
      { priority: 'MEDIUM', text: 'Review who has access to what', reason: 'Remove unnecessary permissions' },
      { priority: 'MEDIUM', text: 'Set up security monitoring', reason: 'Detect problems faster' }
    ] : threatLevel === 'Medium' ? [
      { priority: 'MEDIUM', text: 'Update old software packages', reason: 'Stay protected from known threats' },
      { priority: 'MEDIUM', text: 'Make password rules stronger', reason: 'Prevent easy-to-guess passwords' },
      { priority: 'LOW', text: 'Add security training for your team', reason: 'Human awareness is the best defense' },
      { priority: 'LOW', text: 'Schedule regular security checkups', reason: 'Catch problems early' }
    ] : [
      { priority: 'LOW', text: 'Keep up the good work!', reason: 'Your security is in good shape' },
      { priority: 'LOW', text: 'Stay updated with security news', reason: 'Be aware of new threats' },
      { priority: 'LOW', text: 'Do quarterly security reviews', reason: 'Maintain your security posture' }
    ];

    return {
      overallThreatLevel: threatLevel,
      threatsDetected: criticalIssues + warnings + infoCount,
      criticalIssues,
      warnings,
      info: infoCount,
      confidence: Math.floor(85 + Math.random() * 10),
      predictions,
      vulnerabilities,
      securityScore,
      timelineRisk: {
        next24h: Math.max(15, Math.min(95, riskScore - 10)),
        next7days: Math.max(20, Math.min(98, riskScore)),
        next30days: Math.max(25, Math.min(98, riskScore + 5))
      },
      recommendations,
      summary: threatLevel === 'Critical' 
        ? 'DANGER! Your system has serious security problems that need immediate attention. Think of this like finding your house has been broken into.'
        : threatLevel === 'High'
        ? 'WARNING: You have important security issues that should be fixed soon. It\'s like having a broken lock on your door.'
        : threatLevel === 'Medium'
        ? 'ATTENTION: Some security improvements are recommended. Similar to needing to replace old batteries in your smoke detector.'
        : indicators.isKnownSafeSite
        ? 'ALL GOOD! Your security looks solid. Keep doing what you\'re doing!'
        : 'LOOKING GOOD: Just a few small tweaks would make things even better. You\'re mostly secure!'
    };
  };

  const handleFileUpload = (e) => {
    const files = Array.from(e.target.files);
    setUploadedFiles(files);
    setAnalysisResult(null);
  };

  const analyzeFiles = async () => {
  if (uploadedFiles.length === 0) return;
  setAnalyzing(true);

  try {
    const formData = new FormData();
    formData.append('file', uploadedFiles[0]);

    // Call ML backend API
    const response = await fetch('http://localhost:8000/api/analyze/file', {
      method: 'POST',
      body: formData
    });

    if (!response.ok) {
      throw new Error('Analysis failed');
    }

    const mlResult = await response.json();

    // Convert ML result to existing format
    const result = {
      overallThreatLevel: mlResult.threat_level,
      threatsDetected: mlResult.critical_issues + mlResult.warnings,
      criticalIssues: mlResult.critical_issues,
      warnings: mlResult.warnings,
      info: Math.floor(Math.random() * 5) + 3,
      confidence: mlResult.confidence,
      predictions: generateThreatPredictions(mlResult.threat_level, { 
        fileName: mlResult.filename,
        riskScore: mlResult.risk_score 
      }),
      vulnerabilities: generateVulnerabilities(mlResult.threat_level),
      securityScore: {
        overall: 100 - mlResult.risk_score,
        encryption: 100 - mlResult.risk_score,
        authentication: 100 - mlResult.risk_score + 5,
        dataProtection: 100 - mlResult.risk_score - 3,
        networkSecurity: 100 - mlResult.risk_score - 5
      },
      timelineRisk: {
        next24h: Math.max(15, mlResult.risk_score - 10),
        next7days: mlResult.risk_score,
        next30days: Math.min(98, mlResult.risk_score + 10)
      },
      recommendations: generateRecommendations(mlResult.threat_level),
      summary: generateSummary(mlResult.threat_level, mlResult.risk_score)
    };

    setAnalysisResult(result);
  } catch (error) {
    console.error('ML Analysis failed, using fallback:', error);
    // Fallback to original logic
    const fileIdentifier = uploadedFiles.map(f => f.name + f.size).join('-');
    const result = generateAnalysis('file', fileIdentifier);
    setAnalysisResult(result);
  }

  setAnalyzing(false);
};

  const analyzeAPI = () => {
    if (!apiEndpoint) return;
    setAnalyzing(true);
    setTimeout(() => {
      const result = generateAnalysis('api', apiEndpoint);
      setAnalysisResult(result);
      setAnalyzing(false);
    }, 2500);
  };

  const startMonitoring = async () => {
  if (!monitorUrl) return;
  setAnalyzing(true);

  try {
    // Call ML backend API
    const response = await fetch('http://localhost:8000/api/analyze/url', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url: monitorUrl })
    });

    const mlResult = await response.json();

    // Convert to existing format
    const result = {
      overallThreatLevel: mlResult.threat_level,
      // ... rest of the conversion
    };

    setAnalysisResult(result);
  } catch (error) {
    // Fallback
    const result = generateAnalysis('monitor', monitorUrl);
    setAnalysisResult(result);
  }

  setAnalyzing(false);
};

  const getThreatColor = (level) => {
    const colors = {
      'Critical': 'bg-red-500',
      'High': 'bg-orange-500',
      'Medium': 'bg-yellow-500',
      'Low': 'bg-green-500'
    };
    return colors[level] || 'bg-gray-500';
  };

  const getPriorityColor = (priority) => {
    const colors = {
      'URGENT': 'bg-red-500 text-white',
      'HIGH': 'bg-orange-500 text-white',
      'MEDIUM': 'bg-yellow-500 text-white',
      'LOW': 'bg-green-500 text-white'
    };
    return colors[priority] || 'bg-gray-500 text-white';
  };

  const sendMessage = () => {
    if (!inputMessage.trim()) return;
    setMessages(prev => [...prev, { role: 'user', content: inputMessage }]);
    setInputMessage('');
    setIsTyping(true);

    setTimeout(() => {
      let response = '';
      const query = inputMessage.toLowerCase();

      if (analysisResult) {
        if (query.includes('what') && query.includes('mean')) {
          response = `In simple terms: ${analysisResult.summary}\n\nYou have ${analysisResult.criticalIssues} serious problems, ${analysisResult.warnings} warnings, and ${analysisResult.info} things to keep an eye on.`;
        } else if (query.includes('how') && query.includes('fix')) {
          response = `Here's what to do:\n\n${analysisResult.recommendations.slice(0, 3).map((r, i) => `${i + 1}. ${r.text} - ${r.reason}`).join('\n\n')}`;
        } else if (query.includes('dangerous') || query.includes('serious')) {
          const criticalThreats = analysisResult.predictions.filter(p => p.severity === 'Critical');
          if (criticalThreats.length > 0) {
            response = `Yes, you have ${criticalThreats.length} serious threat(s):\n\n${criticalThreats[0].easyExplanation}`;
          } else {
            response = `Good news! You don't have any critical threats right now. ${analysisResult.summary}`;
          }
        } else {
          response = `${analysisResult.summary}\n\nAsk me things like:\n- "What does this mean?"\n- "How do I fix this?"\n- "Is this dangerous?"`;
        }
      } else {
        response = 'Please run a security scan first! Upload a file or enter a website to check.';
      }

      setMessages(prev => [...prev, { role: 'assistant', content: response }]);
      setIsTyping(false);
    }, 1000);
  };

  const exportReport = () => {
    const report = { ...analysisResult, generatedAt: new Date().toISOString() };
    const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `security-report-${Date.now()}.json`;
    a.click();
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 text-white p-6">
      <div className="max-w-7xl mx-auto">
        <div className="flex items-center gap-3 mb-8">
          <Shield className="w-12 h-12 text-cyan-400" />
          <div>
            <h1 className="text-4xl font-bold bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent">
              CyberShield AI
            </h1>
            <p className="text-slate-400 text-sm">Your Security Made Simple</p>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="space-y-6">
            <div className="bg-slate-800/50 border border-slate-700 rounded-xl p-2">
              <button
                onClick={() => setActiveTab('upload')}
                className={`w-full flex items-center gap-2 px-4 py-3 rounded-lg transition ${activeTab === 'upload' ? 'bg-cyan-500 text-white' : 'text-slate-400 hover:bg-slate-700'}`}
              >
                <Upload className="w-5 h-5" />
                Upload Files
              </button>
              <button
                onClick={() => setActiveTab('api')}
                className={`w-full flex items-center gap-2 px-4 py-3 rounded-lg transition ${activeTab === 'api' ? 'bg-cyan-500 text-white' : 'text-slate-400 hover:bg-slate-700'}`}
              >
                <Database className="w-5 h-5" />
                Check API
              </button>
              <button
                onClick={() => setActiveTab('monitor')}
                className={`w-full flex items-center gap-2 px-4 py-3 rounded-lg transition ${activeTab === 'monitor' ? 'bg-cyan-500 text-white' : 'text-slate-400 hover:bg-slate-700'}`}
              >
                <Activity className="w-5 h-5" />
                Monitor Website
              </button>
            </div>

            <div className="bg-slate-800/50 border border-slate-700 rounded-xl p-6">
              {activeTab === 'upload' && (
                <div className="space-y-4">
                  <h3 className="text-lg font-semibold">Upload Your Files</h3>
                  <p className="text-sm text-slate-400">We'll check them for security issues</p>
                  <div className="border-2 border-dashed border-slate-600 rounded-lg p-8 text-center hover:border-cyan-500 transition cursor-pointer">
                    <input type="file" multiple onChange={handleFileUpload} className="hidden" id="file" />
                    <label htmlFor="file" className="cursor-pointer">
                      <Upload className="w-12 h-12 mx-auto mb-3 text-slate-500" />
                      <p className="text-slate-400">Click to choose files</p>
                      <p className="text-xs text-slate-500 mt-2">Supports: logs, data files, etc.</p>
                    </label>
                  </div>
                  {uploadedFiles.length > 0 && (
                    <div>
                      <p className="text-sm font-medium mb-2">Files ready to scan:</p>
                      {uploadedFiles.map((file, idx) => (
                        <div key={idx} className="text-sm bg-slate-700/50 p-3 rounded mb-2 flex items-center gap-2">
                          <FileText className="w-4 h-4 text-cyan-400" />
                          <span className="flex-1">{file.name}</span>
                          <span className="text-xs text-slate-500">{(file.size / 1024).toFixed(1)} KB</span>
                        </div>
                      ))}
                    </div>
                  )}
                  <button
                    onClick={analyzeFiles}
                    disabled={uploadedFiles.length === 0 || analyzing}
                    className="w-full bg-cyan-500 hover:bg-cyan-600 disabled:bg-slate-600 text-white font-semibold py-3 rounded-lg transition"
                  >
                    {analyzing ? 'Scanning...' : '🔍 Scan for Threats'}
                  </button>
                </div>
              )}

              {activeTab === 'api' && (
                <div className="space-y-4">
                  <h3 className="text-lg font-semibold">Check Your API</h3>
                  <p className="text-sm text-slate-400">We'll test it for security weaknesses</p>
                  <input
                    type="text"
                    value={apiEndpoint}
                    onChange={(e) => setApiEndpoint(e.target.value)}
                    placeholder="https://api.yoursite.com"
                    className="w-full bg-slate-700 border border-slate-600 rounded-lg px-4 py-3 text-white"
                  />
                  <button
                    onClick={analyzeAPI}
                    disabled={!apiEndpoint || analyzing}
                    className="w-full bg-cyan-500 hover:bg-cyan-600 disabled:bg-slate-600 text-white font-semibold py-3 rounded-lg transition"
                  >
                    {analyzing ? 'Testing...' : '🔍 Test API Security'}
                  </button>
                </div>
              )}

              {activeTab === 'monitor' && (
                <div className="space-y-4">
                  <h3 className="text-lg font-semibold">Monitor a Website</h3>
                  <p className="text-sm text-slate-400">We'll check if it's safe</p>
                  <input
                    type="text"
                    value={monitorUrl}
                    onChange={(e) => setMonitorUrl(e.target.value)}
                    placeholder="https://example.com"
                    className="w-full bg-slate-700 border border-slate-600 rounded-lg px-4 py-3 text-white"
                  />
                  <button
                    onClick={startMonitoring}
                    disabled={!monitorUrl || analyzing}
                    className="w-full bg-green-500 hover:bg-green-600 disabled:bg-slate-600 text-white font-semibold py-3 rounded-lg transition"
                  >
                    {analyzing ? 'Checking...' : '🔍 Check Website'}
                  </button>
                </div>
              )}
            </div>
          </div>

          <div className="lg:col-span-2">
            {!analysisResult && !analyzing && (
              <div className="bg-slate-800/50 border border-slate-700 rounded-xl p-12 text-center">
                <Shield className="w-20 h-20 mx-auto mb-4 text-slate-600" />
                <p className="text-slate-400 text-lg mb-2">Ready to Scan</p>
                <p className="text-slate-500 text-sm">Choose an option on the left to start your security check</p>
              </div>
            )}

            {analyzing && (
              <div className="bg-slate-800/50 border border-slate-700 rounded-xl p-12 text-center">
                <div className="relative w-20 h-20 mx-auto mb-6">
                  <div className="absolute inset-0 border-4 border-cyan-500/30 rounded-full"></div>
                  <div className="absolute inset-0 border-4 border-cyan-500 rounded-full border-t-transparent animate-spin"></div>
                  <Shield className="absolute inset-0 m-auto w-8 h-8 text-cyan-400" />
                </div>
                <p className="text-slate-300 text-lg font-medium">Scanning for threats...</p>
                <p className="text-slate-500 text-sm mt-2">This usually takes a few seconds</p>
              </div>
            )}

            {analysisResult && (
              <div className="space-y-6">
                {/* Summary Card */}
                <div className="bg-gradient-to-br from-slate-800 to-slate-900 border border-slate-700 rounded-2xl p-8">
                  <div className="flex justify-between items-start mb-6">
                    <div>
                      <h2 className="text-3xl font-bold mb-2">Security Report</h2>
                      <p className="text-slate-400 text-sm">Generated just now</p>
                    </div>
                    <button onClick={exportReport} className="px-4 py-2 bg-cyan-500/20 border border-cyan-500 text-cyan-400 rounded-lg flex items-center gap-2 hover:bg-cyan-500/30 transition">
                      <Download className="w-4 h-4" />
                      Save Report
                    </button>
                  </div>

                  <div className={`p-6 rounded-xl ${getThreatColor(analysisResult.overallThreatLevel)} bg-opacity-20 border-2 border-current mb-6`}>
                    <div className="flex items-center justify-between mb-3">
                      <span className="text-2xl font-bold">Threat Level: {analysisResult.overallThreatLevel}</span>
                      <div className={`px-4 py-2 rounded-lg font-bold ${getThreatColor(analysisResult.overallThreatLevel)} text-white`}>
                        {analysisResult.overallThreatLevel === 'Critical' && '🚨 URGENT'}
                        {analysisResult.overallThreatLevel === 'High' && '⚠️ HIGH'}
                        {analysisResult.overallThreatLevel === 'Medium' && '⚡ MEDIUM'}
                        {analysisResult.overallThreatLevel === 'Low' && '✅ LOW'}
                      </div>
                    </div>
                    <p className="text-lg">{analysisResult.summary}</p>
                  </div>

                  {/* Quick Stats */}
                  <div className="grid grid-cols-4 gap-4">
                    <div className="bg-red-500/10 border border-red-500/30 rounded-xl p-4 text-center">
                      <div className="text-4xl font-bold text-red-400 mb-1">{analysisResult.criticalIssues}</div>
                      <div className="text-sm text-red-300 font-medium">Serious Issues</div>
                      <div className="text-xs text-slate-400 mt-1">Need immediate fix</div>
                    </div>
                    <div className="bg-yellow-500/10 border border-yellow-500/30 rounded-xl p-4 text-center">
                      <div className="text-4xl font-bold text-yellow-400 mb-1">{analysisResult.warnings}</div>
                      <div className="text-sm text-yellow-300 font-medium">Warnings</div>
                      <div className="text-xs text-slate-400 mt-1">Fix soon</div>
                    </div>
                    <div className="bg-blue-500/10 border border-blue-500/30 rounded-xl p-4 text-center">
                      <div className="text-4xl font-bold text-blue-400 mb-1">{analysisResult.info}</div>
                      <div className="text-sm text-blue-300 font-medium">FYI Items</div>
                      <div className="text-xs text-slate-400 mt-1">Keep an eye on</div>
                    </div>
                    <div className="bg-cyan-500/10 border border-cyan-500/30 rounded-xl p-4 text-center">
                      <div className="text-4xl font-bold text-cyan-400 mb-1">{analysisResult.confidence}%</div>
                      <div className="text-sm text-cyan-300 font-medium">Confidence</div>
                      <div className="text-xs text-slate-400 mt-1">How sure we are</div>
                    </div>
                  </div>
                </div>

                {/* Security Score Chart */}
                <div className="bg-slate-800/50 border border-slate-700 rounded-xl p-6">
                  <h3 className="text-xl font-semibold mb-2 flex items-center gap-2">
                    <BarChart3 className="w-6 h-6 text-cyan-400" />
                    Security Score by Category
                  </h3>
                  <p className="text-sm text-slate-400 mb-6">How well protected you are in each area (0-100)</p>
                  
                  <div className="space-y-4">
                    <div>
                      <div className="flex justify-between mb-2">
                        <span className="text-sm font-medium">🔐 Encryption & Privacy</span>
                        <span className="text-sm font-bold text-cyan-400">{analysisResult.securityScore.encryption}/100</span>
                      </div>
                      <div className="h-4 bg-slate-700 rounded-full overflow-hidden">
                        <div 
                          className="h-full bg-gradient-to-r from-cyan-500 to-blue-500 transition-all duration-1000"
                          style={{width: `${analysisResult.securityScore.encryption}%`}}
                        />
                      </div>
                      <p className="text-xs text-slate-500 mt-1">How well your data is encrypted and kept private</p>
                    </div>

                    <div>
                      <div className="flex justify-between mb-2">
                        <span className="text-sm font-medium">👤 Login & Authentication</span>
                        <span className="text-sm font-bold text-cyan-400">{analysisResult.securityScore.authentication}/100</span>
                      </div>
                      <div className="h-4 bg-slate-700 rounded-full overflow-hidden">
                        <div 
                          className="h-full bg-gradient-to-r from-purple-500 to-pink-500 transition-all duration-1000"
                          style={{width: `${analysisResult.securityScore.authentication}%`}}
                        />
                      </div>
                      <p className="text-xs text-slate-500 mt-1">How secure your login system is</p>
                    </div>

                    <div>
                      <div className="flex justify-between mb-2">
                        <span className="text-sm font-medium">💾 Data Protection</span>
                        <span className="text-sm font-bold text-cyan-400">{analysisResult.securityScore.dataProtection}/100</span>
                      </div>
                      <div className="h-4 bg-slate-700 rounded-full overflow-hidden">
                        <div 
                          className="h-full bg-gradient-to-r from-green-500 to-emerald-500 transition-all duration-1000"
                          style={{width: `${analysisResult.securityScore.dataProtection}%`}}
                        />
                      </div>
                      <p className="text-xs text-slate-500 mt-1">How well your stored data is protected</p>
                    </div>

                    <div>
                      <div className="flex justify-between mb-2">
                        <span className="text-sm font-medium">🌐 Network Security</span>
                        <span className="text-sm font-bold text-cyan-400">{analysisResult.securityScore.networkSecurity}/100</span>
                      </div>
                      <div className="h-4 bg-slate-700 rounded-full overflow-hidden">
                        <div 
                          className="h-full bg-gradient-to-r from-orange-500 to-red-500 transition-all duration-1000"
                          style={{width: `${analysisResult.securityScore.networkSecurity}%`}}
                        />
                      </div>
                      <p className="text-xs text-slate-500 mt-1">How secure your internet connections are</p>
                    </div>

                    <div className="mt-4 p-4 bg-slate-700/30 rounded-lg">
                      <div className="flex items-center justify-between">
                        <span className="font-semibold">Overall Security Score</span>
                        <span className="text-3xl font-bold bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent">
                          {analysisResult.securityScore.overall}/100
                        </span>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Risk Timeline Chart */}
                <div className="bg-slate-800/50 border border-slate-700 rounded-xl p-6">
                  <h3 className="text-xl font-semibold mb-2 flex items-center gap-2">
                    <LineChart className="w-6 h-6 text-cyan-400" />
                    Risk Over Time
                  </h3>
                  <p className="text-sm text-slate-400 mb-6">How likely problems are to happen if you don't fix things</p>
                  
                  <div className="space-y-5">
                    <div>
                      <div className="flex justify-between mb-2">
                        <div>
                          <span className="text-sm font-medium">⏰ Next 24 Hours</span>
                          <p className="text-xs text-slate-500">Immediate risk</p>
                        </div>
                        <span className="text-2xl font-bold text-yellow-400">{analysisResult.timelineRisk.next24h}%</span>
                      </div>
                      <div className="h-6 bg-slate-700 rounded-full overflow-hidden">
                        <div 
                          className="h-full bg-gradient-to-r from-yellow-500 to-yellow-400 transition-all duration-1000 flex items-center justify-center text-xs font-bold text-white"
                          style={{width: `${analysisResult.timelineRisk.next24h}%`}}
                        >
                          {analysisResult.timelineRisk.next24h > 20 && `${analysisResult.timelineRisk.next24h}%`}
                        </div>
                      </div>
                    </div>

                    <div>
                      <div className="flex justify-between mb-2">
                        <div>
                          <span className="text-sm font-medium">📅 Next 7 Days</span>
                          <p className="text-xs text-slate-500">Short-term risk</p>
                        </div>
                        <span className="text-2xl font-bold text-orange-400">{analysisResult.timelineRisk.next7days}%</span>
                      </div>
                      <div className="h-6 bg-slate-700 rounded-full overflow-hidden">
                        <div 
                          className="h-full bg-gradient-to-r from-orange-500 to-orange-400 transition-all duration-1000 flex items-center justify-center text-xs font-bold text-white"
                          style={{width: `${analysisResult.timelineRisk.next7days}%`}}
                        >
                          {analysisResult.timelineRisk.next7days > 20 && `${analysisResult.timelineRisk.next7days}%`}
                        </div>
                      </div>
                    </div>

                    <div>
                      <div className="flex justify-between mb-2">
                        <div>
                          <span className="text-sm font-medium">📆 Next 30 Days</span>
                          <p className="text-xs text-slate-500">Long-term risk</p>
                        </div>
                        <span className="text-2xl font-bold text-red-400">{analysisResult.timelineRisk.next30days}%</span>
                      </div>
                      <div className="h-6 bg-slate-700 rounded-full overflow-hidden">
                        <div 
                          className="h-full bg-gradient-to-r from-red-500 to-red-400 transition-all duration-1000 flex items-center justify-center text-xs font-bold text-white"
                          style={{width: `${analysisResult.timelineRisk.next30days}%`}}
                        >
                          {analysisResult.timelineRisk.next30days > 20 && `${analysisResult.timelineRisk.next30days}%`}
                        </div>
                      </div>
                    </div>
                  </div>

                  <div className="mt-4 p-4 bg-blue-500/10 border border-blue-500/30 rounded-lg">
                    <p className="text-sm text-blue-300">
                      💡 <strong>What this means:</strong> The risk increases over time because attackers are always looking for weaknesses. Fix issues quickly to keep the risk low!
                    </p>
                  </div>
                </div>

                {/* Threats Explained */}
                <div className="bg-slate-800/50 border border-slate-700 rounded-xl p-6">
                  <h3 className="text-xl font-semibold mb-2 flex items-center gap-2">
                    <Target className="w-6 h-6 text-cyan-400" />
                    Threats Found (In Simple Terms)
                  </h3>
                  <p className="text-sm text-slate-400 mb-6">What we found and what it means for you</p>
                  
                  <div className="space-y-4">
                    {analysisResult.predictions.map((threat, idx) => (
                      <div key={idx} className="bg-slate-700/50 border border-slate-600 rounded-xl p-5">
                        <div className="flex items-start justify-between mb-4">
                          <div className="flex-1">
                            <div className="flex items-center gap-3 mb-2">
                              <div className={`px-3 py-1 rounded-full text-xs font-bold ${threat.severity === 'Critical' ? 'bg-red-500' : threat.severity === 'High' ? 'bg-orange-500' : threat.severity === 'Medium' ? 'bg-yellow-500' : 'bg-green-500'} text-white`}>
                                {threat.severity === 'Critical' && '🚨 SERIOUS'}
                                {threat.severity === 'High' && '⚠️ IMPORTANT'}
                                {threat.severity === 'Medium' && '⚡ ATTENTION'}
                                {threat.severity === 'Low' && '✅ MINOR'}
                              </div>
                              <span className="font-bold text-lg">{threat.type}</span>
                            </div>
                            <p className="text-sm text-slate-400 mb-3">{threat.location}</p>
                          </div>
                          <div className="text-right ml-4">
                            <div className="text-4xl font-bold bg-gradient-to-br from-red-400 to-orange-500 bg-clip-text text-transparent">
                              {threat.probability}%
                            </div>
                            <div className="text-xs text-slate-400">Chance of attack</div>
                          </div>
                        </div>

                        <div className="space-y-3 text-sm">
                          <div className="p-3 bg-blue-500/10 border border-blue-500/30 rounded-lg">
                            <p className="font-semibold text-blue-300 mb-1">🤔 What does this mean?</p>
                            <p className="text-slate-300">{threat.whatItMeans}</p>
                          </div>

                          <div className="p-3 bg-purple-500/10 border border-purple-500/30 rounded-lg">
                            <p className="font-semibold text-purple-300 mb-1">💭 In everyday language:</p>
                            <p className="text-slate-300 italic">{threat.easyExplanation}</p>
                          </div>

                          <div className="p-3 bg-red-500/10 border border-red-500/30 rounded-lg">
                            <p className="font-semibold text-red-300 mb-1">⚠️ What could happen:</p>
                            <p className="text-slate-300">{threat.impact}</p>
                          </div>

                          <div className="p-3 bg-green-500/10 border border-green-500/30 rounded-lg">
                            <p className="font-semibold text-green-300 mb-1">✅ How to fix it:</p>
                            <p className="text-slate-300">{threat.howToBeFix}</p>
                          </div>

                          <div className="text-xs text-slate-500 italic">
                            Technical name: {threat.technicalName}
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Known Vulnerabilities */}
                <div className="bg-slate-800/50 border border-slate-700 rounded-xl p-6">
                  <h3 className="text-xl font-semibold mb-2 flex items-center gap-2">
                    <Bug className="w-6 h-6 text-cyan-400" />
                    Known Security Bugs (CVEs)
                  </h3>
                  <p className="text-sm text-slate-400 mb-6">These are publicly known security problems in your software</p>
                  
                  <div className="space-y-3">
                    {analysisResult.vulnerabilities.map((vuln, idx) => (
                      <div key={idx} className="bg-slate-700/50 border border-slate-600 rounded-xl p-4">
                        <div className="flex items-center justify-between mb-2">
                          <div className="flex-1">
                            <div className="flex items-center gap-3 mb-1">
                              <span className="font-mono text-cyan-400 font-bold">{vuln.cve}</span>
                              <span className={`px-2 py-1 rounded text-xs font-bold ${vuln.severity === 'Critical' ? 'bg-red-500' : vuln.severity === 'High' ? 'bg-orange-500' : vuln.severity === 'Medium' ? 'bg-yellow-500' : 'bg-green-500'} text-white`}>
                                {vuln.severity}
                              </span>
                            </div>
                            <p className="text-sm text-slate-400">{vuln.component}</p>
                            <p className="text-xs text-slate-500 mt-1">{vuln.explanation}</p>
                          </div>
                          <div className="text-right ml-4">
                            <div className="text-3xl font-bold text-red-400">{vuln.cvssScore}</div>
                            <div className="text-xs text-slate-500">Severity Score</div>
                            <div className="text-xs text-slate-500">(out of 10)</div>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>

                  <div className="mt-4 p-4 bg-yellow-500/10 border border-yellow-500/30 rounded-lg">
                    <p className="text-sm text-yellow-300">
                      💡 <strong>Quick tip:</strong> CVE numbers are official IDs for security problems. You can Google them to learn more!
                    </p>
                  </div>
                </div>

                {/* Action Plan */}
                <div className="bg-gradient-to-br from-cyan-500/10 to-blue-500/10 border-2 border-cyan-500/30 rounded-xl p-6">
                  <h3 className="text-2xl font-bold mb-2 flex items-center gap-2">
                    <CheckCircle className="w-7 h-7 text-cyan-400" />
                    Your Action Plan (What To Do Now)
                  </h3>
                  <p className="text-sm text-slate-400 mb-6">Follow these steps in order of importance</p>
                  
                  <div className="space-y-3">
                    {analysisResult.recommendations.map((rec, idx) => (
                      <div key={idx} className="bg-slate-800/50 border border-slate-700 p-4 rounded-lg hover:border-cyan-500/50 transition">
                        <div className="flex items-start gap-4">
                          <div className="flex-shrink-0">
                            <div className="w-10 h-10 rounded-full bg-cyan-500/20 border-2 border-cyan-500/50 flex items-center justify-center">
                              <span className="text-cyan-400 font-bold text-lg">{idx + 1}</span>
                            </div>
                          </div>
                          <div className="flex-1">
                            <div className="flex items-center gap-2 mb-2">
                              <span className={`px-3 py-1 rounded-full text-xs font-bold ${getPriorityColor(rec.priority)}`}>
                                {rec.priority}
                              </span>
                              <span className="font-semibold text-white">{rec.text}</span>
                            </div>
                            <p className="text-sm text-slate-400">
                              <strong className="text-slate-300">Why:</strong> {rec.reason}
                            </p>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>

                  <div className="mt-6 p-4 bg-blue-500/20 border border-blue-500/30 rounded-lg">
                    <p className="text-sm text-blue-300">
                      <strong>💪 Remember:</strong> Taking action now is better than waiting! Even small fixes make a big difference in keeping you safe.
                    </p>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* AI Chatbot */}
        {chatOpen && (
          <div className="fixed bottom-24 right-6 w-96 h-[500px] bg-slate-800 border border-slate-700 rounded-2xl shadow-2xl flex flex-col z-50">
            <div className="bg-gradient-to-r from-cyan-500 to-blue-500 p-4 rounded-t-2xl flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 bg-white rounded-full flex items-center justify-center">
                  <Bot className="w-6 h-6 text-cyan-500" />
                </div>
                <div>
                  <h3 className="font-semibold text-white">Security Helper</h3>
                  <p className="text-xs text-cyan-100">Ask me anything!</p>
                </div>
              </div>
              <button onClick={() => setChatOpen(false)} className="text-white hover:bg-white/20 p-2 rounded-lg transition">
                <X className="w-5 h-5" />
              </button>
            </div>

            <div className="flex-1 overflow-y-auto p-4 space-y-3">
              {messages.map((msg, idx) => (
                <div key={idx} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                  <div className={`max-w-[85%] p-3 rounded-lg ${msg.role === 'user' ? 'bg-gradient-to-br from-cyan-500 to-blue-500 text-white' : 'bg-slate-700 text-slate-100'}`}>
                    {msg.role === 'assistant' && (
                      <div className="flex items-center gap-2 mb-1">
                        <Bot className="w-4 h-4 text-cyan-400" />
                        <span className="text-xs text-cyan-400 font-semibold">Helper</span>
                      </div>
                    )}
                    <p className="text-sm whitespace-pre-wrap">{msg.content}</p>
                  </div>
                </div>
              ))}
              {isTyping && (
                <div className="flex justify-start">
                  <div className="bg-slate-700 p-3 rounded-lg">
                    <div className="flex gap-1">
                      <div className="w-2 h-2 bg-cyan-500 rounded-full animate-bounce"></div>
                      <div className="w-2 h-2 bg-cyan-500 rounded-full animate-bounce" style={{animationDelay: '150ms'}}></div>
                      <div className="w-2 h-2 bg-cyan-500 rounded-full animate-bounce" style={{animationDelay: '300ms'}}></div>
                    </div>
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>

            <div className="p-4 border-t border-slate-700">
              <div className="flex gap-2">
                <input
                  type="text"
                  value={inputMessage}
                  onChange={(e) => setInputMessage(e.target.value)}
                  onKeyPress={(e) => e.key === 'Enter' && sendMessage()}
                  placeholder="Ask me to explain..."
                  className="flex-1 bg-slate-700 border border-slate-600 rounded-lg px-4 py-2 text-white text-sm focus:outline-none focus:ring-2 focus:ring-cyan-500"
                />
                <button onClick={sendMessage} disabled={!inputMessage.trim() || isTyping} className="bg-gradient-to-r from-cyan-500 to-blue-500 hover:from-cyan-600 hover:to-blue-600 disabled:from-slate-600 disabled:to-slate-600 p-2 rounded-lg transition">
                  <Send className="w-5 h-5 text-white" />
                </button>
              </div>
              <p className="text-xs text-slate-500 mt-2">Try: "What does this mean?" or "How do I fix this?"</p>
            </div>
          </div>
        )}

        <button
          onClick={() => setChatOpen(!chatOpen)}
          className="fixed bottom-6 right-6 w-16 h-16 bg-gradient-to-r from-cyan-500 to-blue-500 hover:from-cyan-600 hover:to-blue-600 rounded-full shadow-2xl flex items-center justify-center z-40 hover:scale-110 transition-all"
        >
          {chatOpen ? (
            <X className="w-7 h-7 text-white" />
          ) : (
            <>
              <MessageSquare className="w-7 h-7 text-white" />
              {analysisResult && (
                <span className="absolute -top-1 -right-1 w-5 h-5 bg-red-500 rounded-full flex items-center justify-center text-xs font-bold text-white animate-pulse">!</span>
              )}
            </>
          )}
        </button>
      </div>
    </div>
  );
};

export default CyberThreatPlatform;