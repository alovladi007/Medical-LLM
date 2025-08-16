import React, { useState, useEffect, useCallback } from 'react';
import { LineChart, Line, BarChart, Bar, PieChart, Pie, Cell, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar } from 'recharts';
import { Camera, Heart, Brain, FileText, Users, BookOpen, Activity, AlertCircle, TrendingUp, Shield, Zap, Database, Clock, Search, Bell, Settings, ChevronRight, Home, BarChart3, Stethoscope, FlaskConical, GraduationCap, MessageSquare, Upload, Download, ChevronDown } from 'lucide-react';

// Main Application Component
const MedAGIPlatform = () => {
  const [activeTab, setActiveTab] = useState('dashboard');
  const [userRole, setUserRole] = useState('provider');
  const [notifications, setNotifications] = useState(5);
  const [patientData, setPatientData] = useState(null);
  const [aiInsights, setAiInsights] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  // Simulated real-time data
  const [vitals, setVitals] = useState({
    heartRate: 72,
    bloodPressure: '120/80',
    temperature: 98.6,
    o2Saturation: 98,
    respiratoryRate: 16
  });

  // Simulate real-time updates
  useEffect(() => {
    const interval = setInterval(() => {
      setVitals(prev => ({
        ...prev,
        heartRate: Math.floor(70 + Math.random() * 10),
        o2Saturation: Math.floor(96 + Math.random() * 4)
      }));
    }, 3000);
    return () => clearInterval(interval);
  }, []);

  // Navigation Component
  const Navigation = () => (
    <nav className="bg-gradient-to-r from-blue-600 to-purple-600 text-white shadow-lg">
      <div className="px-6 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-8">
            <div className="flex items-center space-x-3">
              <div className="bg-white/20 p-2 rounded-lg">
                <Brain className="h-8 w-8" />
              </div>
              <div>
                <h1 className="text-2xl font-bold">Med-AGI</h1>
                <p className="text-xs opacity-90">Medical Intelligence Platform</p>
              </div>
            </div>
            
            <div className="flex space-x-6">
              <button
                onClick={() => setActiveTab('dashboard')}
                className={`px-4 py-2 rounded-lg transition ${
                  activeTab === 'dashboard' ? 'bg-white/20' : 'hover:bg-white/10'
                }`}
              >
                <Home className="h-5 w-5 inline mr-2" />
                Dashboard
              </button>
              <button
                onClick={() => setActiveTab('diagnostics')}
                className={`px-4 py-2 rounded-lg transition ${
                  activeTab === 'diagnostics' ? 'bg-white/20' : 'hover:bg-white/10'
                }`}
              >
                <Stethoscope className="h-5 w-5 inline mr-2" />
                Diagnostics
              </button>
              <button
                onClick={() => setActiveTab('research')}
                className={`px-4 py-2 rounded-lg transition ${
                  activeTab === 'research' ? 'bg-white/20' : 'hover:bg-white/10'
                }`}
              >
                <FlaskConical className="h-5 w-5 inline mr-2" />
                Research
              </button>
              <button
                onClick={() => setActiveTab('education')}
                className={`px-4 py-2 rounded-lg transition ${
                  activeTab === 'education' ? 'bg-white/20' : 'hover:bg-white/10'
                }`}
              >
                <GraduationCap className="h-5 w-5 inline mr-2" />
                Education
              </button>
              <button
                onClick={() => setActiveTab('ai-assistant')}
                className={`px-4 py-2 rounded-lg transition ${
                  activeTab === 'ai-assistant' ? 'bg-white/20' : 'hover:bg-white/10'
                }`}
              >
                <MessageSquare className="h-5 w-5 inline mr-2" />
                AI Assistant
              </button>
            </div>
          </div>
          
          <div className="flex items-center space-x-4">
            <div className="relative">
              <Bell className="h-6 w-6 cursor-pointer hover:scale-110 transition" />
              {notifications > 0 && (
                <span className="absolute -top-2 -right-2 bg-red-500 text-xs rounded-full h-5 w-5 flex items-center justify-center">
                  {notifications}
                </span>
              )}
            </div>
            <Settings className="h-6 w-6 cursor-pointer hover:scale-110 transition" />
            <div className="flex items-center space-x-2 bg-white/20 px-3 py-1 rounded-lg">
              <div className="w-8 h-8 bg-white/30 rounded-full flex items-center justify-center">
                <Users className="h-5 w-5" />
              </div>
              <select 
                value={userRole} 
                onChange={(e) => setUserRole(e.target.value)}
                className="bg-transparent outline-none text-sm"
              >
                <option value="provider" className="text-gray-800">Provider</option>
                <option value="researcher" className="text-gray-800">Researcher</option>
                <option value="student" className="text-gray-800">Student</option>
              </select>
            </div>
          </div>
        </div>
      </div>
    </nav>
  );

  // Dashboard Component
  const Dashboard = () => {
    const performanceData = [
      { month: 'Jan', accuracy: 94, cases: 120 },
      { month: 'Feb', accuracy: 95, cases: 145 },
      { month: 'Mar', accuracy: 96, cases: 168 },
      { month: 'Apr', accuracy: 97, cases: 192 },
      { month: 'May', accuracy: 96, cases: 210 },
      { month: 'Jun', accuracy: 98, cases: 235 }
    ];

    const specialtyData = [
      { specialty: 'Cardiology', value: 35, color: '#3B82F6' },
      { specialty: 'Radiology', value: 28, color: '#8B5CF6' },
      { specialty: 'Neurology', value: 20, color: '#10B981' },
      { specialty: 'Oncology', value: 17, color: '#F59E0B' }
    ];

    return (
      <div className="p-6 space-y-6">
        {/* Key Metrics */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
          <MetricCard
            title="Active Patients"
            value="1,847"
            change="+12%"
            icon={<Users className="h-6 w-6" />}
            color="bg-blue-500"
          />
          <MetricCard
            title="AI Diagnoses"
            value="98.2%"
            subtitle="Accuracy"
            icon={<Brain className="h-6 w-6" />}
            color="bg-purple-500"
          />
          <MetricCard
            title="Research Studies"
            value="24"
            subtitle="Active"
            icon={<FlaskConical className="h-6 w-6" />}
            color="bg-green-500"
          />
          <MetricCard
            title="Student Progress"
            value="87%"
            subtitle="Avg Score"
            icon={<GraduationCap className="h-6 w-6" />}
            color="bg-yellow-500"
          />
        </div>

        {/* Real-time Patient Monitor */}
        <div className="bg-white rounded-xl shadow-lg p-6">
          <h2 className="text-xl font-bold mb-4 flex items-center">
            <Activity className="mr-2 text-red-500" />
            Real-time Patient Monitoring
          </h2>
          <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
            <VitalSign label="Heart Rate" value={`${vitals.heartRate} bpm`} status="normal" />
            <VitalSign label="Blood Pressure" value={vitals.bloodPressure} status="normal" />
            <VitalSign label="Temperature" value={`${vitals.temperature}°F`} status="normal" />
            <VitalSign label="O₂ Saturation" value={`${vitals.o2Saturation}%`} status="normal" />
            <VitalSign label="Respiratory Rate" value={`${vitals.respiratoryRate}/min`} status="normal" />
          </div>
        </div>

        {/* Charts */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="bg-white rounded-xl shadow-lg p-6">
            <h3 className="text-lg font-bold mb-4">Diagnostic Performance</h3>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={performanceData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="month" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Line type="monotone" dataKey="accuracy" stroke="#3B82F6" name="Accuracy %" />
                <Line type="monotone" dataKey="cases" stroke="#10B981" name="Cases" />
              </LineChart>
            </ResponsiveContainer>
          </div>

          <div className="bg-white rounded-xl shadow-lg p-6">
            <h3 className="text-lg font-bold mb-4">Cases by Specialty</h3>
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie
                  data={specialtyData}
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  label={({ specialty, value }) => `${specialty}: ${value}%`}
                  outerRadius={80}
                  fill="#8884d8"
                  dataKey="value"
                >
                  {specialtyData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Recent Activity */}
        <div className="bg-white rounded-xl shadow-lg p-6">
          <h3 className="text-lg font-bold mb-4">Recent AI Insights</h3>
          <div className="space-y-3">
            <InsightItem
              type="diagnosis"
              message="Potential early-stage pneumonia detected in Patient #4821 chest X-ray"
              confidence={92}
              time="2 minutes ago"
            />
            <InsightItem
              type="alert"
              message="Abnormal EKG pattern detected - possible atrial fibrillation"
              confidence={88}
              time="5 minutes ago"
            />
            <InsightItem
              type="research"
              message="New correlation found between biomarkers in diabetes study cohort"
              confidence={95}
              time="1 hour ago"
            />
          </div>
        </div>
      </div>
    );
  };

  // Diagnostics Component
  const Diagnostics = () => {
    const [selectedImage, setSelectedImage] = useState(null);
    const [analysisResult, setAnalysisResult] = useState(null);

    const handleImageUpload = (e) => {
      const file = e.target.files[0];
      if (file) {
        setSelectedImage(URL.createObjectURL(file));
        simulateAnalysis();
      }
    };

    const simulateAnalysis = () => {
      setIsLoading(true);
      setTimeout(() => {
        setAnalysisResult({
          findings: [
            { condition: 'Cardiomegaly', probability: 0.78 },
            { condition: 'Pleural Effusion', probability: 0.65 },
            { condition: 'Pneumonia', probability: 0.42 },
            { condition: 'No Finding', probability: 0.15 }
          ],
          urgency: 'moderate',
          recommendations: [
            'Consider echocardiogram for cardiac evaluation',
            'Recommend follow-up chest X-ray in 2 weeks',
            'Evaluate for signs of heart failure'
          ]
        });
        setIsLoading(false);
      }, 2000);
    };

    return (
      <div className="p-6 space-y-6">
        <div className="bg-white rounded-xl shadow-lg p-6">
          <h2 className="text-2xl font-bold mb-6">AI-Powered Diagnostic Assistant</h2>
          
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Image Upload Section */}
            <div>
              <div className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center">
                {selectedImage ? (
                  <div>
                    <img src={selectedImage} alt="Medical scan" className="mx-auto max-h-96 rounded" />
                    <button 
                      onClick={() => {setSelectedImage(null); setAnalysisResult(null);}}
                      className="mt-4 text-red-500 hover:text-red-700"
                    >
                      Remove Image
                    </button>
                  </div>
                ) : (
                  <div>
                    <Camera className="mx-auto h-16 w-16 text-gray-400 mb-4" />
                    <label className="cursor-pointer">
                      <span className="bg-blue-500 text-white px-6 py-3 rounded-lg inline-block hover:bg-blue-600 transition">
                        Upload Medical Image
                      </span>
                      <input
                        type="file"
                        className="hidden"
                        accept="image/*"
                        onChange={handleImageUpload}
                      />
                    </label>
                    <p className="text-gray-500 mt-3">Supports X-ray, CT, MRI formats</p>
                  </div>
                )}
              </div>

              {/* Patient Context */}
              <div className="mt-6 space-y-4">
                <h3 className="font-semibold">Patient Context</h3>
                <input
                  type="text"
                  placeholder="Chief complaint"
                  className="w-full px-4 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
                <textarea
                  placeholder="Clinical history..."
                  className="w-full px-4 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                  rows="3"
                />
              </div>
            </div>

            {/* Analysis Results */}
            <div>
              {isLoading ? (
                <div className="flex items-center justify-center h-full">
                  <div className="text-center">
                    <div className="animate-spin rounded-full h-16 w-16 border-b-2 border-blue-500 mx-auto"></div>
                    <p className="mt-4 text-gray-600">Analyzing image with AI models...</p>
                  </div>
                </div>
              ) : analysisResult ? (
                <div className="space-y-6">
                  <div>
                    <h3 className="font-semibold mb-3">Detected Conditions</h3>
                    {analysisResult.findings.map((finding, idx) => (
                      <div key={idx} className="mb-3">
                        <div className="flex justify-between items-center mb-1">
                          <span className="text-sm font-medium">{finding.condition}</span>
                          <span className="text-sm text-gray-600">
                            {(finding.probability * 100).toFixed(1)}%
                          </span>
                        </div>
                        <div className="w-full bg-gray-200 rounded-full h-2">
                          <div
                            className={`h-2 rounded-full ${
                              finding.probability > 0.7 ? 'bg-red-500' :
                              finding.probability > 0.4 ? 'bg-yellow-500' : 'bg-green-500'
                            }`}
                            style={{ width: `${finding.probability * 100}%` }}
                          />
                        </div>
                      </div>
                    ))}
                  </div>

                  <div>
                    <h3 className="font-semibold mb-3">Clinical Recommendations</h3>
                    <ul className="space-y-2">
                      {analysisResult.recommendations.map((rec, idx) => (
                        <li key={idx} className="flex items-start">
                          <ChevronRight className="h-5 w-5 text-blue-500 mt-0.5 mr-2 flex-shrink-0" />
                          <span className="text-sm">{rec}</span>
                        </li>
                      ))}
                    </ul>
                  </div>

                  <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
                    <div className="flex items-center">
                      <AlertCircle className="h-5 w-5 text-yellow-600 mr-2" />
                      <span className="font-semibold text-yellow-800">Urgency: Moderate</span>
                    </div>
                    <p className="text-sm text-yellow-700 mt-2">
                      Recommend clinical correlation and follow-up within 48 hours
                    </p>
                  </div>
                </div>
              ) : (
                <div className="text-center text-gray-500 mt-20">
                  <FileText className="h-16 w-16 mx-auto mb-4 text-gray-300" />
                  <p>Upload an image to start AI analysis</p>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Differential Diagnosis Tool */}
        <DifferentialDiagnosis />
      </div>
    );
  };

  // Differential Diagnosis Component
  const DifferentialDiagnosis = () => {
    const [symptoms, setSymptoms] = useState('');
    const [differentials, setDifferentials] = useState(null);

    const generateDifferentials = () => {
      if (!symptoms) return;
      
      setIsLoading(true);
      setTimeout(() => {
        setDifferentials([
          { diagnosis: 'Acute Myocardial Infarction', probability: 0.85, evidence: ['Chest pain', 'Elevated troponins', 'EKG changes'] },
          { diagnosis: 'Pulmonary Embolism', probability: 0.72, evidence: ['Chest pain', 'Dyspnea', 'Tachycardia'] },
          { diagnosis: 'Aortic Dissection', probability: 0.45, evidence: ['Chest pain', 'Hypertension history'] },
          { diagnosis: 'Pneumothorax', probability: 0.38, evidence: ['Chest pain', 'Dyspnea'] },
          { diagnosis: 'GERD', probability: 0.25, evidence: ['Chest pain', 'Post-prandial timing'] }
        ]);
        setIsLoading(false);
      }, 1500);
    };

    return (
      <div className="bg-white rounded-xl shadow-lg p-6">
        <h3 className="text-xl font-bold mb-4">Differential Diagnosis Generator</h3>
        
        <div className="flex space-x-4 mb-6">
          <input
            type="text"
            value={symptoms}
            onChange={(e) => setSymptoms(e.target.value)}
            placeholder="Enter symptoms (e.g., chest pain, dyspnea, fever)"
            className="flex-1 px-4 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
          <button
            onClick={generateDifferentials}
            disabled={!symptoms || isLoading}
            className="bg-blue-500 text-white px-6 py-2 rounded-lg hover:bg-blue-600 transition disabled:opacity-50"
          >
            {isLoading ? 'Analyzing...' : 'Generate'}
          </button>
        </div>

        {differentials && (
          <div className="space-y-4">
            {differentials.map((dx, idx) => (
              <div key={idx} className="border rounded-lg p-4 hover:shadow-md transition">
                <div className="flex justify-between items-start mb-2">
                  <h4 className="font-semibold text-lg">{dx.diagnosis}</h4>
                  <span className={`px-3 py-1 rounded-full text-sm font-medium ${
                    dx.probability > 0.7 ? 'bg-red-100 text-red-700' :
                    dx.probability > 0.4 ? 'bg-yellow-100 text-yellow-700' :
                    'bg-green-100 text-green-700'
                  }`}>
                    {(dx.probability * 100).toFixed(0)}% probability
                  </span>
                </div>
                <div>
                  <span className="text-sm text-gray-600">Supporting evidence:</span>
                  <div className="flex flex-wrap gap-2 mt-1">
                    {dx.evidence.map((item, i) => (
                      <span key={i} className="bg-gray-100 px-2 py-1 rounded text-sm">
                        {item}
                      </span>
                    ))}
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    );
  };

  // Research Hub Component
  const ResearchHub = () => {
    const studyData = [
      { name: 'Diabetes Biomarkers', enrollment: 324, completion: 78 },
      { name: 'COVID Long-term Effects', enrollment: 892, completion: 45 },
      { name: 'AI Diagnostic Accuracy', enrollment: 1250, completion: 92 },
      { name: 'Precision Oncology', enrollment: 156, completion: 33 }
    ];

    return (
      <div className="p-6 space-y-6">
        <div className="bg-white rounded-xl shadow-lg p-6">
          <h2 className="text-2xl font-bold mb-6">Research Analytics Hub</h2>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
            <div className="bg-gradient-to-r from-purple-500 to-pink-500 rounded-lg p-6 text-white">
              <Database className="h-8 w-8 mb-3" />
              <h3 className="text-3xl font-bold">2.3M</h3>
              <p className="text-sm opacity-90">Patient Records</p>
            </div>
            <div className="bg-gradient-to-r from-blue-500 to-cyan-500 rounded-lg p-6 text-white">
              <BarChart3 className="h-8 w-8 mb-3" />
              <h3 className="text-3xl font-bold">156</h3>
              <p className="text-sm opacity-90">Active Studies</p>
            </div>
            <div className="bg-gradient-to-r from-green-500 to-teal-500 rounded-lg p-6 text-white">
              <TrendingUp className="h-8 w-8 mb-3" />
              <h3 className="text-3xl font-bold">94%</h3>
              <p className="text-sm opacity-90">Data Quality Score</p>
            </div>
          </div>

          <div>
            <h3 className="text-lg font-semibold mb-4">Active Clinical Studies</h3>
            <div className="space-y-3">
              {studyData.map((study, idx) => (
                <div key={idx} className="border rounded-lg p-4">
                  <div className="flex justify-between items-center mb-2">
                    <h4 className="font-medium">{study.name}</h4>
                    <span className="text-sm text-gray-600">{study.enrollment} participants</span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div
                      className="bg-blue-500 h-2 rounded-full"
                      style={{ width: `${study.completion}%` }}
                    />
                  </div>
                  <span className="text-xs text-gray-500 mt-1">{study.completion}% complete</span>
                </div>
              ))}
            </div>
          </div>
        </div>

        <HypothesisGenerator />
      </div>
    );
  };

  // Hypothesis Generator Component
  const HypothesisGenerator = () => {
    const [researchQuestion, setResearchQuestion] = useState('');
    const [hypotheses, setHypotheses] = useState(null);

    const generateHypotheses = () => {
      if (!researchQuestion) return;
      
      setIsLoading(true);
      setTimeout(() => {
        setHypotheses([
          {
            hypothesis: "There is a significant correlation between sleep duration and glycemic control in Type 2 diabetes patients",
            confidence: 0.82,
            supportingLiterature: 45,
            suggestedMethodology: "Prospective cohort study with continuous glucose monitoring"
          },
          {
            hypothesis: "Circadian rhythm disruption mediates the relationship between shift work and diabetes risk",
            confidence: 0.75,
            supportingLiterature: 32,
            suggestedMethodology: "Cross-sectional study with actigraphy and metabolic profiling"
          },
          {
            hypothesis: "Sleep quality improvements can reduce HbA1c levels independent of weight loss",
            confidence: 0.68,
            supportingLiterature: 28,
            suggestedMethodology: "Randomized controlled trial with sleep intervention"
          }
        ]);
        setIsLoading(false);
      }, 2000);
    };

    return (
      <div className="bg-white rounded-xl shadow-lg p-6">
        <h3 className="text-xl font-bold mb-4">AI Hypothesis Generator</h3>
        
        <div className="flex space-x-4 mb-6">
          <input
            type="text"
            value={researchQuestion}
            onChange={(e) => setResearchQuestion(e.target.value)}
            placeholder="Enter research question (e.g., relationship between sleep and diabetes)"
            className="flex-1 px-4 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500"
          />
          <button
            onClick={generateHypotheses}
            disabled={!researchQuestion || isLoading}
            className="bg-purple-500 text-white px-6 py-2 rounded-lg hover:bg-purple-600 transition disabled:opacity-50"
          >
            {isLoading ? 'Generating...' : 'Generate'}
          </button>
        </div>

        {hypotheses && (
          <div className="space-y-4">
            {hypotheses.map((hyp, idx) => (
              <div key={idx} className="border rounded-lg p-4 bg-purple-50">
                <h4 className="font-semibold mb-2">{hyp.hypothesis}</h4>
                <div className="grid grid-cols-3 gap-4 text-sm">
                  <div>
                    <span className="text-gray-600">Confidence:</span>
                    <div className="font-medium text-purple-600">
                      {(hyp.confidence * 100).toFixed(0)}%
                    </div>
                  </div>
                  <div>
                    <span className="text-gray-600">Literature Support:</span>
                    <div className="font-medium text-purple-600">
                      {hyp.supportingLiterature} papers
                    </div>
                  </div>
                  <div>
                    <span className="text-gray-600">Methodology:</span>
                    <div className="font-medium text-purple-600 text-xs">
                      {hyp.suggestedMethodology}
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    );
  };

  // Education Portal Component
  const EducationPortal = () => {
    const courses = [
      { name: 'Clinical Reasoning', progress: 75, nextLesson: 'Differential Diagnosis Workshop' },
      { name: 'Medical Imaging Interpretation', progress: 60, nextLesson: 'Chest X-ray Patterns' },
      { name: 'Evidence-Based Medicine', progress: 85, nextLesson: 'Meta-analysis Techniques' },
      { name: 'Patient Communication', progress: 92, nextLesson: 'Breaking Bad News' }
    ];

    const skillsData = [
      { skill: 'Diagnosis', score: 85 },
      { skill: 'Treatment', score: 78 },
      { skill: 'Communication', score: 92 },
      { skill: 'Research', score: 70 },
      { skill: 'Procedures', score: 65 },
      { skill: 'Documentation', score: 88 }
    ];

    return (
      <div className="p-6 space-y-6">
        <div className="bg-white rounded-xl shadow-lg p-6">
          <h2 className="text-2xl font-bold mb-6">Medical Education Hub</h2>
          
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Learning Progress */}
            <div>
              <h3 className="text-lg font-semibold mb-4">Your Courses</h3>
              <div className="space-y-3">
                {courses.map((course, idx) => (
                  <div key={idx} className="border rounded-lg p-4 hover:shadow-md transition cursor-pointer">
                    <div className="flex justify-between items-center mb-2">
                      <h4 className="font-medium">{course.name}</h4>
                      <span className="text-sm text-gray-600">{course.progress}%</span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-2 mb-2">
                      <div
                        className="bg-green-500 h-2 rounded-full"
                        style={{ width: `${course.progress}%` }}
                      />
                    </div>
                    <p className="text-xs text-gray-600">Next: {course.nextLesson}</p>
                  </div>
                ))}
              </div>
            </div>

            {/* Skills Assessment */}
            <div>
              <h3 className="text-lg font-semibold mb-4">Skills Assessment</h3>
              <ResponsiveContainer width="100%" height={300}>
                <RadarChart data={skillsData}>
                  <PolarGrid />
                  <PolarAngleAxis dataKey="skill" />
                  <PolarRadiusAxis angle={90} domain={[0, 100]} />
                  <Radar name="Score" dataKey="score" stroke="#8B5CF6" fill="#8B5CF6" fillOpacity={0.6} />
                </RadarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>

        <VirtualPatientSimulator />
      </div>
    );
  };

  // Virtual Patient Simulator Component
  const VirtualPatientSimulator = () => {
    const [simulationStarted, setSimulationStarted] = useState(false);
    const [currentStep, setCurrentStep] = useState(0);

    const patientCase = {
      presentation: "45-year-old male presents with acute chest pain, shortness of breath, and diaphoresis",
      vitals: "BP: 150/95, HR: 110, RR: 22, O2: 94%, Temp: 98.6°F",
      history: "Hypertension, Type 2 Diabetes, 20-pack year smoking history",
      options: [
        "Order EKG and troponins",
        "Administer aspirin and nitroglycerin",
        "Order chest X-ray",
        "Start oxygen therapy"
      ]
    };

    return (
      <div className="bg-white rounded-xl shadow-lg p-6">
        <h3 className="text-xl font-bold mb-4">Virtual Patient Simulator</h3>
        
        {!simulationStarted ? (
          <div className="text-center py-12">
            <Heart className="h-16 w-16 text-red-500 mx-auto mb-4" />
            <h4 className="text-lg font-semibold mb-2">Emergency Department Case</h4>
            <p className="text-gray-600 mb-4">Practice clinical decision-making with AI feedback</p>
            <button
              onClick={() => setSimulationStarted(true)}
              className="bg-red-500 text-white px-6 py-3 rounded-lg hover:bg-red-600 transition"
            >
              Start Simulation
            </button>
          </div>
        ) : (
          <div className="space-y-4">
            <div className="bg-red-50 border border-red-200 rounded-lg p-4">
              <h4 className="font-semibold mb-2">Patient Presentation</h4>
              <p className="text-sm">{patientCase.presentation}</p>
              <div className="mt-2 grid grid-cols-2 gap-2 text-sm">
                <div><strong>Vitals:</strong> {patientCase.vitals}</div>
                <div><strong>History:</strong> {patientCase.history}</div>
              </div>
            </div>

            <div>
              <h4 className="font-semibold mb-3">What is your next action?</h4>
              <div className="space-y-2">
                {patientCase.options.map((option, idx) => (
                  <button
                    key={idx}
                    onClick={() => setCurrentStep(currentStep + 1)}
                    className="w-full text-left px-4 py-3 border rounded-lg hover:bg-blue-50 hover:border-blue-300 transition"
                  >
                    {idx + 1}. {option}
                  </button>
                ))}
              </div>
            </div>

            {currentStep > 0 && (
              <div className="bg-green-50 border border-green-200 rounded-lg p-4">
                <div className="flex items-center mb-2">
                  <Shield className="h-5 w-5 text-green-600 mr-2" />
                  <span className="font-semibold text-green-800">AI Feedback</span>
                </div>
                <p className="text-sm text-green-700">
                  Good choice! Ordering EKG and troponins is appropriate for suspected acute coronary syndrome.
                  Consider also administering aspirin if no contraindications.
                </p>
              </div>
            )}
          </div>
        )}
      </div>
    );
  };

  // AI Assistant Component
  const AIAssistant = () => {
    const [message, setMessage] = useState('');
    const [conversation, setConversation] = useState([
      { role: 'assistant', content: 'Hello! I\'m your medical AI assistant. How can I help you today?' }
    ]);

    const sendMessage = () => {
      if (!message.trim()) return;
      
      setConversation([...conversation, { role: 'user', content: message }]);
      setMessage('');
      
      // Simulate AI response
      setTimeout(() => {
        setConversation(prev => [...prev, {
          role: 'assistant',
          content: 'Based on the symptoms you\'ve described, I recommend considering the following differential diagnoses: 1) Acute coronary syndrome, 2) Pulmonary embolism, 3) Aortic dissection. Would you like me to elaborate on any of these conditions or suggest appropriate diagnostic workup?'
        }]);
      }, 1000);
    };

    return (
      <div className="p-6">
        <div className="bg-white rounded-xl shadow-lg p-6 h-[600px] flex flex-col">
          <div className="flex items-center mb-4 pb-4 border-b">
            <Brain className="h-8 w-8 text-purple-500 mr-3" />
            <div>
              <h2 className="text-xl font-bold">Medical AI Assistant</h2>
              <p className="text-sm text-gray-600">Powered by advanced medical language models</p>
            </div>
          </div>

          <div className="flex-1 overflow-y-auto mb-4 space-y-4">
            {conversation.map((msg, idx) => (
              <div key={idx} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                <div className={`max-w-[70%] px-4 py-3 rounded-lg ${
                  msg.role === 'user' 
                    ? 'bg-blue-500 text-white' 
                    : 'bg-gray-100 text-gray-800'
                }`}>
                  <p className="text-sm">{msg.content}</p>
                </div>
              </div>
            ))}
          </div>

          <div className="flex space-x-2">
            <input
              type="text"
              value={message}
              onChange={(e) => setMessage(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && sendMessage()}
              placeholder="Ask a medical question..."
              className="flex-1 px-4 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500"
            />
            <button
              onClick={sendMessage}
              className="bg-purple-500 text-white px-6 py-2 rounded-lg hover:bg-purple-600 transition"
            >
              Send
            </button>
          </div>
        </div>
      </div>
    );
  };

  // Helper Components
  const MetricCard = ({ title, value, change, subtitle, icon, color }) => (
    <div className="bg-white rounded-xl shadow-lg p-6">
      <div className="flex items-center justify-between mb-4">
        <div className={`${color} text-white p-3 rounded-lg`}>
          {icon}
        </div>
        {change && (
          <span className={`text-sm font-semibold ${
            change.startsWith('+') ? 'text-green-600' : 'text-red-600'
          }`}>
            {change}
          </span>
        )}
      </div>
      <h3 className="text-2xl font-bold text-gray-800">{value}</h3>
      <p className="text-sm text-gray-600 mt-1">{subtitle || title}</p>
    </div>
  );

  const VitalSign = ({ label, value, status }) => (
    <div className="bg-gray-50 rounded-lg p-3">
      <p className="text-xs text-gray-600 mb-1">{label}</p>
      <p className="text-lg font-semibold">{value}</p>
      <div className={`h-1 w-full rounded mt-2 ${
        status === 'normal' ? 'bg-green-500' : 
        status === 'warning' ? 'bg-yellow-500' : 'bg-red-500'
      }`} />
    </div>
  );

  const InsightItem = ({ type, message, confidence, time }) => (
    <div className="flex items-start space-x-3 p-3 bg-gray-50 rounded-lg">
      <div className={`p-2 rounded-lg ${
        type === 'diagnosis' ? 'bg-blue-100' :
        type === 'alert' ? 'bg-red-100' : 'bg-purple-100'
      }`}>
        {type === 'diagnosis' ? <Stethoscope className="h-5 w-5 text-blue-600" /> :
         type === 'alert' ? <AlertCircle className="h-5 w-5 text-red-600" /> :
         <FlaskConical className="h-5 w-5 text-purple-600" />}
      </div>
      <div className="flex-1">
        <p className="text-sm font-medium">{message}</p>
        <div className="flex items-center mt-1 text-xs text-gray-500">
          <span>{confidence}% confidence</span>
          <span className="mx-2">•</span>
          <span>{time}</span>
        </div>
      </div>
    </div>
  );

  // Main Render
  return (
    <div className="min-h-screen bg-gray-50">
      <Navigation />
      
      <div className="flex">
        {/* Sidebar */}
        <div className="w-64 bg-white shadow-lg h-[calc(100vh-80px)] p-4">
          <div className="space-y-2">
            <button className="w-full text-left px-4 py-2 rounded-lg hover:bg-gray-100 flex items-center">
              <Clock className="h-4 w-4 mr-3 text-gray-600" />
              <span className="text-sm">Recent Patients</span>
            </button>
            <button className="w-full text-left px-4 py-2 rounded-lg hover:bg-gray-100 flex items-center">
              <FileText className="h-4 w-4 mr-3 text-gray-600" />
              <span className="text-sm">Reports</span>
            </button>
            <button className="w-full text-left px-4 py-2 rounded-lg hover:bg-gray-100 flex items-center">
              <Database className="h-4 w-4 mr-3 text-gray-600" />
              <span className="text-sm">Datasets</span>
            </button>
            <button className="w-full text-left px-4 py-2 rounded-lg hover:bg-gray-100 flex items-center">
              <Users className="h-4 w-4 mr-3 text-gray-600" />
              <span className="text-sm">Collaborations</span>
            </button>
          </div>

          <div className="mt-8 p-4 bg-gradient-to-r from-purple-500 to-pink-500 rounded-lg text-white">
            <Zap className="h-6 w-6 mb-2" />
            <h4 className="font-semibold">AI Models Status</h4>
            <p className="text-xs mt-1 opacity-90">All systems operational</p>
            <div className="mt-3 space-y-1">
              <div className="flex justify-between text-xs">
                <span>GPU Utilization</span>
                <span>67%</span>
              </div>
              <div className="flex justify-between text-xs">
                <span>Model Accuracy</span>
                <span>98.2%</span>
              </div>
            </div>
          </div>
        </div>

        {/* Main Content */}
        <div className="flex-1 overflow-y-auto h-[calc(100vh-80px)]">
          {activeTab === 'dashboard' && <Dashboard />}
          {activeTab === 'diagnostics' && <Diagnostics />}
          {activeTab === 'research' && <ResearchHub />}
          {activeTab === 'education' && <EducationPortal />}
          {activeTab === 'ai-assistant' && <AIAssistant />}
        </div>
      </div>
    </div>
  );
};

export default MedAGIPlatform;