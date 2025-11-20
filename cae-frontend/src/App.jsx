import React, { useState, useEffect, useRef } from 'react';
import { 
  Upload, 
  Play, 
  Settings, 
  Activity, 
  Trash2, 
  BarChart3, 
  Cpu,
  Layers,
  Move,
  Lock,
  Plus,
  Image as ImageIcon,
  Info,
  Thermometer,
  TrendingUp,
  Save,
  Database,
  CheckCircle,
  RefreshCw
} from 'lucide-react';

// --- Configuration Constants ---

const PCA_CLASSES = [
  'kL1-50-4', 'kL2-50-4', 'kL3-50-4', 
  'kR1-50-4', 'kR2-50-4', 'kR3-50-4',
  'mL1c-50-15', 'mL1r-50-15', 'mL1t-50-15', 'mL2c-50-15', 'mL2r-50-15',
  'mL2t-50-15', 'mL3c-50-15', 'mL3r-50-15', 'mL3t-50-15', 'mR1c-50-15',
  'mR1r-50-15', 'mR1t-50-15', 'mR2c-50-15', 'mR2r-50-15', 'mR2t-50-15',
  'mR3c-50-15', 'mR3r-50-15', 'mR3t-50-15', 'sL1-50-33', 'sL2-50-33',
  'sL3-50-33', 'sR1-50-33', 'sR2-50-33', 'sR3-50-33'
];

const PCA_RAW_COLORS = [
  'pink', 'pink', 'pink', 
  'red', 'red', 'red',
  'grey', 'grey', 'grey', 'grey', 'grey',
  'grey', 'grey', 'grey', 'grey', 'grey',
  'grey', 'grey', 'grey', 'grey', 'grey',
  'grey', 'grey', 'grey', 'aquamarine', 'aquamarine',
  'aquamarine', 'blue', 'blue', 'blue'
];

const COLOR_HEX_MAP = {
  'pink': '#ef4444',       // Mapped to Red group
  'red': '#ef4444',        // Red-500
  'grey': '#94a3b8',       // Slate-400
  'aquamarine': '#3b82f6', // Mapped to Blue group
  'blue': '#3b82f6'        // Blue-500
};

// Consolidated Legend Items (3 Groups)
const LEGEND_ITEMS = [
  { label: 'Kneading Elements', color: '#ef4444' }, // Red
  { label: 'Mixing Elements', color: '#94a3b8' },   // Grey
  { label: 'Screw Elements', color: '#3b82f6' },    // Blue
];

// --- Data Helpers ---
const GENERATE_FULL_PALETTE = () => {
  return PCA_CLASSES.map((name, index) => {
    const rawColor = PCA_RAW_COLORS[index];
    const hexColor = COLOR_HEX_MAP[rawColor] || '#94a3b8'; 
    return {
      id: `std-${index}`,
      name: name,
      type: 'standard',
      color: hexColor
    };
  });
};

const DEFAULT_PALETTE = GENERATE_FULL_PALETTE();
const FIXED_ELEMENT = { id: 'fixed-base', name: 'Fixed', type: 'locked', color: '#e2e8f0', locked: true };

// --- Helper Components ---

const RTDChart = ({ data }) => {
  if (!data || data.length === 0) return null;
  const height = 100;
  const width = 300;
  const maxVal = Math.max(...data);
  const points = data.map((val, i) => {
    const x = (i / (data.length - 1)) * width;
    const y = height - (val / maxVal) * height;
    return `${x},${y}`;
  }).join(' ');

  return (
    <div className="w-full h-32 bg-slate-50 rounded border border-slate-200 flex items-end relative overflow-hidden">
      <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-full p-2">
        <polyline fill="none" stroke="#3b82f6" strokeWidth="3" points={points} strokeLinecap="round" strokeLinejoin="round" />
        <polygon fill="rgba(59, 130, 246, 0.1)" points={`0,${height} ${points} ${width},${height}`} />
      </svg>
      <div className="absolute bottom-1 right-2 text-[10px] text-slate-400 font-mono">Time (s) →</div>
      <div className="absolute top-1 left-2 text-[10px] text-slate-400 font-mono">Conc.</div>
    </div>
  );
};

const PCAGraph = ({ data }) => (
  <div className="w-full h-64 bg-slate-900 rounded-lg border border-slate-800 relative overflow-hidden group">
    <div className="absolute inset-0 opacity-20" 
          style={{ backgroundImage: 'linear-gradient(#444 1px, transparent 1px), linear-gradient(90deg, #444 1px, transparent 1px)', backgroundSize: '20px 20px' }}>
    </div>
    {data.map((pt, i) => (
      <div 
        key={i}
        title={pt.name}
        className="absolute w-2 h-2 rounded-full shadow-lg transform -translate-x-1/2 -translate-y-1/2 transition-all hover:scale-150 hover:z-10"
        style={{ 
          left: `${pt.x}%`, 
          top: `${pt.y}%`,
          backgroundColor: pt.color,
          opacity: 0.7 + (pt.z / 500)
        }}
      />
    ))}
    <div className="absolute bottom-2 right-2 text-[10px] text-slate-500 font-mono">PC1 vs PC2</div>
  </div>
);

// --- Main Application ---

const TwinScrewApp = () => {
  const [mode, setMode] = useState('test'); // 'test', 'train'
  const [extruderConfig, setExtruderConfig] = useState(Array(12).fill(FIXED_ELEMENT));
  const [testRegionStart, setTestRegionStart] = useState(4); 
  
  const [paletteElements, setPaletteElements] = useState(DEFAULT_PALETTE);
  const [customElement, setCustomElement] = useState(null);
  const [draggedItem, setDraggedItem] = useState(null);
  
  // --- Training State ---
  const [trainingFile, setTrainingFile] = useState(null);
  const [isTraining, setIsTraining] = useState(false);
  const [newModelName, setNewModelName] = useState("");
  const [baseModelForTraining, setBaseModelForTraining] = useState("Default Model (v1.0)");
  const [trainingResults, setTrainingResults] = useState(null); // New state for training outputs

  // --- Model Management State ---
  const [availableModels, setAvailableModels] = useState(["Default Model (v1.0)"]);
  const [selectedTestModel, setSelectedTestModel] = useState("Default Model (v1.0)");

  // --- Testing Results State ---
  const [results, setResults] = useState(null);
  const fileInputRef = useRef(null);

  const refreshModels = (newModel) => {
    if (newModel && !availableModels.includes(newModel)) {
      setAvailableModels(prev => [...prev, newModel]);
    }
  };

  // --- Handlers ---

  const handleCustomImageUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      const imageUrl = URL.createObjectURL(file);
      const newCustom = {
        id: `custom-${Date.now()}`,
        name: file.name.substring(0, 10),
        type: 'custom',
        image: imageUrl,
        color: '#8b5cf6' 
      };
      setCustomElement(newCustom);
    }
  };

  const handleDragStart = (e, element) => {
    setDraggedItem(element);
    e.dataTransfer.effectAllowed = "copy";
  };

  const handleDrop = (index) => {
    if (index >= testRegionStart && index < testRegionStart + 3) {
      if (draggedItem) {
        const newConfig = [...extruderConfig];
        newConfig[index] = { ...draggedItem, instanceId: Date.now(), locked: false };
        setExtruderConfig(newConfig);
        setDraggedItem(null);
      }
    } else {
      alert("Restricted: You can only modify elements inside the Red Test Region.");
    }
  };

  const handleRemoveElement = (index) => {
    if (index >= testRegionStart && index < testRegionStart + 3) {
      const newConfig = [...extruderConfig];
      newConfig[index] = FIXED_ELEMENT; 
      setExtruderConfig(newConfig);
    }
  };

  const handleUploadTraining = (e) => {
    setTrainingFile(e.target.files[0]);
    setTrainingResults(null); // Reset previous results on new file
  };

  const startTraining = () => {
    if (!trainingFile) return alert("Please upload a CSV first");
    if (!newModelName.trim()) return alert("Please give your new model a name.");

    setIsTraining(true);
    setTrainingResults(null);
    
    // Simulate Backend Training Delay
    setTimeout(() => {
      setIsTraining(false);
      refreshModels(newModelName);
      
      // Generate Mock Training Results (PCA + Metrics)
      const mockPca = Array.from({ length: 80 }, (_, i) => {
        const randIdx = Math.floor(Math.random() * PCA_CLASSES.length);
        const rawColor = PCA_RAW_COLORS[randIdx];
        return {
          x: Math.random() * 100,
          y: Math.random() * 100,
          z: Math.random() * 100,
          name: PCA_CLASSES[randIdx],
          color: COLOR_HEX_MAP[rawColor] || '#ffffff'
        };
      });

      setTrainingResults({
        modelName: newModelName,
        mse: (0.01 + Math.random() * 0.02).toFixed(4),
        r2: (0.90 + Math.random() * 0.09).toFixed(4),
        pcaData: mockPca
      });
      
      setTrainingFile(null);
      setNewModelName("");
      alert(`Training Complete!`);
    }, 2500);
  };


  const runSimulation = async () => {
    const activeRegion = extruderConfig.slice(testRegionStart, testRegionStart + 3);
    
    try {
      const response = await fetch('http://localhost:8000/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          elements: activeRegion,
          model_name: selectedTestModel 
        }),
      });
      
      const data = await response.json();
      
      // Update state with real data from Python
      setResults({
        mse: data.metrics.mse,
        r2: data.metrics.r2,
        temperature: data.temperature,
        rtdCurve: data.rtdCurve
      });
    } catch (error) {
      console.error("Error connecting to backend:", error);
      alert("Failed to connect to simulation server.");
    }
  };
  // --- UI Components ---

  const Header = () => (
    <header className="bg-slate-900 text-white p-4 shadow-lg flex justify-between items-center sticky top-0 z-50">
      <div className="flex items-center gap-3">
        <Cpu className="w-8 h-8 text-blue-400" />
        <h1 className="text-xl font-bold tracking-wide">Twin Screw CAE Interface</h1>
      </div>
      <div className="flex gap-2 bg-slate-800 p-1 rounded-lg">
        <button 
          onClick={() => setMode('test')}
          className={`px-4 py-2 rounded-md transition-colors text-sm font-medium ${mode === 'test' ? 'bg-blue-600 text-white' : 'text-slate-400 hover:text-white'}`}
        >
          Testing & Assembly
        </button>
        <button 
          onClick={() => setMode('train')}
          className={`px-4 py-2 rounded-md transition-colors text-sm font-medium ${mode === 'train' ? 'bg-blue-600 text-white' : 'text-slate-400 hover:text-white'}`}
        >
          Training Mode
        </button>
      </div>
    </header>
  );

  const TrainingPanel = () => (
      <div className="max-w-4xl mx-auto mt-10 space-y-8">
        
        {/* Input Section */}
        <div className="bg-white p-8 rounded-xl shadow-sm border border-slate-200">
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-2xl font-bold text-slate-800 flex items-center gap-2">
              <Settings className="w-6 h-6 text-blue-600" /> Training Center
            </h2>
            <span className="bg-green-100 text-green-700 text-xs font-bold px-2 py-1 rounded border border-green-200">
              System Ready
            </span>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            <div className="space-y-4">
               <div>
                  <label className="block text-sm font-bold text-slate-700 mb-2">1. Select Base Model</label>
                  <select 
                    className="w-full p-2 border border-slate-300 rounded-md text-slate-600 bg-slate-50"
                    value={baseModelForTraining}
                    onChange={(e) => setBaseModelForTraining(e.target.value)}
                  >
                    {availableModels.map((m, i) => <option key={i} value={m}>{m}</option>)}
                  </select>
                </div>
                <div>
                  <label className="block text-sm font-bold text-slate-700 mb-2">2. Save New Model As</label>
                  <input 
                    type="text"
                    placeholder="e.g., 'High-Viscosity-v2'"
                    className="w-full p-2 border border-slate-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:outline-none"
                    value={newModelName}
                    onChange={(e) => setNewModelName(e.target.value)}
                  />
                </div>
            </div>

            <div className="border-2 border-dashed border-slate-300 rounded-xl p-4 text-center hover:bg-slate-50 transition-colors group cursor-pointer flex flex-col items-center justify-center h-full">
              <input 
                type="file" 
                accept=".csv" 
                onChange={handleUploadTraining}
                className="hidden" 
                id="csv-upload" 
              />
              <label htmlFor="csv-upload" className="cursor-pointer flex flex-col items-center gap-2 w-full h-full justify-center">
                <div className="p-3 bg-blue-50 rounded-full group-hover:bg-blue-100 transition-colors">
                   <Upload className="w-6 h-6 text-blue-500" />
                </div>
                <div>
                  <span className="text-sm font-medium text-slate-700 block">
                    {trainingFile ? trainingFile.name : "3. Upload CSV Dataset"}
                  </span>
                </div>
              </label>
            </div>
          </div>
    
          <div className="flex justify-end border-t border-slate-100 pt-6 mt-6">
            <button 
              onClick={startTraining}
              disabled={isTraining || !trainingFile || !newModelName}
              className="bg-slate-900 hover:bg-slate-800 text-white px-8 py-3 rounded-lg font-semibold flex items-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed transition-all shadow-lg"
            >
              {isTraining ? (
                <>
                  <span className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></span>
                  Training...
                </>
              ) : (
                <>
                  <Save className="w-4 h-4" /> Start Training
                </>
              )}
            </button>
          </div>
        </div>

        {/* Results Section (PCA + Metrics) - Only shows AFTER training */}
        {trainingResults && (
          <div className="bg-white p-6 rounded-xl shadow-md border border-green-200 animate-in fade-in slide-in-from-bottom-4 duration-500">
            <div className="flex items-center gap-2 mb-6 border-b border-slate-100 pb-4">
              <CheckCircle className="w-6 h-6 text-green-500" />
              <h3 className="text-xl font-bold text-slate-800">Training Results: {trainingResults.modelName}</h3>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              {/* Column 1: Metrics */}
              <div className="space-y-4">
                 <div className="bg-slate-50 p-4 rounded-lg border border-slate-100">
                    <div className="text-xs text-slate-500 uppercase font-bold">Average MSE Loss</div>
                    <div className="text-3xl font-mono font-bold text-slate-800">{trainingResults.mse}</div>
                 </div>
                 <div className="bg-slate-50 p-4 rounded-lg border border-slate-100">
                    <div className="text-xs text-slate-500 uppercase font-bold">Average R² Score</div>
                    <div className="text-3xl font-mono font-bold text-green-600">{trainingResults.r2}</div>
                 </div>
              </div>

              {/* Column 2 & 3: PCA Graph */}
              <div className="md:col-span-2">
                 <div className="flex justify-between items-center mb-2">
                    <span className="text-sm font-bold text-slate-700 flex items-center gap-2">
                       <Move className="w-4 h-4" /> Dataset Distribution (PCA)
                    </span>
                    {/* Legend Mini */}
                    <div className="flex gap-3">
                      {LEGEND_ITEMS.map((item, i) => (
                        <div key={i} className="flex items-center gap-1.5">
                          <div className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: item.color }} />
                          <span className="text-[10px] text-slate-500 font-medium">{item.label}</span>
                        </div>
                      ))}
                    </div>
                 </div>
                 <PCAGraph data={trainingResults.pcaData} />
              </div>
            </div>
          </div>
        )}
      </div>
    );

  const TestingPanel = () => (
    <div className="flex flex-col h-full gap-6">
      
      {/* SECTION 1: ASSEMBLY (Same as before) */}
      <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-200">
        <div className="flex justify-between items-start mb-8">
          <div>
            <h2 className="text-lg font-bold text-slate-800 flex items-center gap-2">
              <Layers className="w-5 h-5 text-blue-600" /> Assembly Configuration
            </h2>
            <p className="text-xs text-slate-500 mt-1">Configure active region and select simulation model.</p>
          </div>
          
          <div className="flex flex-col items-end gap-3">
            <div className="flex items-center gap-2">
               <Database className="w-4 h-4 text-slate-400" />
               <select 
                 className="p-1.5 text-sm border border-slate-300 rounded bg-slate-50 text-slate-700 focus:outline-none focus:border-blue-500"
                 value={selectedTestModel}
                 onChange={(e) => setSelectedTestModel(e.target.value)}
               >
                 {availableModels.map((m, i) => <option key={i} value={m}>{m}</option>)}
               </select>
            </div>
            <div className="flex items-center gap-4 bg-slate-50 px-3 py-1.5 rounded-lg border border-slate-200">
              <span className="text-[10px] font-bold text-slate-600 uppercase">Active Region:</span>
              <input 
                type="range" 
                min="0" 
                max="9" 
                value={testRegionStart} 
                onChange={(e) => setTestRegionStart(Number(e.target.value))}
                className="w-24 h-2 bg-slate-300 rounded-lg appearance-none cursor-pointer accent-red-500"
              />
            </div>
          </div>
        </div>

        <div className="flex justify-end mb-4">
           <button 
            onClick={runSimulation}
            className="bg-indigo-600 hover:bg-indigo-700 text-white px-5 py-2 rounded-lg text-sm font-bold flex items-center gap-2 shadow-md hover:shadow-lg transition-all"
          >
            <Activity className="w-4 h-4" /> Run Simulation ({selectedTestModel})
          </button>
        </div>

        <div className="flex justify-center pb-8 overflow-x-auto">
          <div className="flex gap-0.5 bg-slate-200 p-2 rounded-lg border border-slate-300 inner-shadow relative">
            {extruderConfig.map((item, idx) => {
              const isEditable = idx >= testRegionStart && idx < testRegionStart + 3;
              return (
                <div 
                  key={idx}
                  onDragOver={(e) => e.preventDefault()}
                  onDrop={() => handleDrop(idx)}
                  className={`
                    w-14 h-28 flex flex-col items-center justify-center transition-all relative group border-r border-slate-300 last:border-r-0
                    ${isEditable 
                      ? 'bg-white z-10 shadow-[0_0_0_2px_rgba(239,68,68,1)] scale-105' 
                      : 'bg-slate-100 opacity-70 grayscale-[0.5]'
                    }
                  `}
                >
                  {idx === testRegionStart && (
                    <div className="absolute -top-8 left-0 w-[168px] flex justify-center pointer-events-none">
                      <span className="bg-red-500 text-white text-[9px] font-bold px-3 py-0.5 rounded-full shadow-sm tracking-wider uppercase">
                        Test Region
                      </span>
                    </div>
                  )}
                  <div className="relative w-full h-full p-1 flex items-center justify-center">
                    {item.image ? (
                       <img src={item.image} alt="elem" className="max-w-full max-h-full object-contain drop-shadow-sm" />
                    ) : (
                       <div className="w-8 h-16 rounded shadow-sm" style={{ backgroundColor: item.color }} />
                    )}
                    <span className="absolute bottom-1 text-[8px] font-bold text-slate-600 bg-white/90 px-1 rounded max-w-full truncate">
                      {item.name}
                    </span>
                  </div>
                  
                  {isEditable && !item.locked ? (
                    <button 
                      onClick={() => handleRemoveElement(idx)}
                      className="absolute -top-2 -right-2 bg-red-500 text-white p-0.5 rounded-full opacity-0 group-hover:opacity-100 transition-all shadow-md z-20"
                    >
                      <Trash2 className="w-3 h-3" />
                    </button>
                  ) : null}
                  <span className="absolute -bottom-5 text-[9px] text-slate-400 font-mono">{idx + 1}</span>
                </div>
              );
            })}
          </div>
        </div>

        <div className="mt-6 pt-4 border-t border-slate-100">
          <div className="flex justify-between items-center mb-3">
             <span className="text-xs font-bold text-slate-500 uppercase tracking-wider">Element Palette ({paletteElements.length} Standard + Custom)</span>
             <div className="flex items-center">
                <input 
                  type="file" 
                  accept="image/png, image/jpeg" 
                  ref={fileInputRef} 
                  className="hidden"
                  onChange={handleCustomImageUpload}
                />
                <button 
                  onClick={() => fileInputRef.current?.click()}
                  className="text-[10px] bg-blue-50 text-blue-600 px-2 py-1 rounded hover:bg-blue-100 transition-colors flex items-center gap-1"
                >
                  <Plus className="w-3 h-3" /> Add Custom
                </button>
              </div>
          </div>
          <div className="flex gap-2 overflow-x-auto pb-2">
             {customElement && (
                <div
                  draggable
                  onDragStart={(e) => handleDragStart(e, customElement)}
                  className="flex-shrink-0 w-12 h-16 bg-purple-50 border border-purple-200 rounded flex flex-col items-center justify-center cursor-move hover:shadow-md"
                >
                  <img src={customElement.image} className="w-6 h-6 object-contain" />
                  <span className="text-[8px] mt-1 text-purple-700">Custom</span>
                </div>
             )}
             {paletteElements.map((el) => (
                <div
                  key={el.id}
                  draggable
                  onDragStart={(e) => handleDragStart(e, el)}
                  className="flex-shrink-0 w-12 h-16 bg-slate-50 border border-slate-200 rounded flex flex-col items-center justify-center cursor-move hover:shadow-md"
                  title={el.name}
                >
                  <div className="w-3 h-6 rounded-sm" style={{ backgroundColor: el.color }}></div>
                  <span className="text-[8px] mt-1 text-slate-600 w-full text-center truncate px-0.5">{el.name}</span>
                </div>
             ))}
          </div>
        </div>
      </div>

      {/* SECTION 2: TEST RESULTS (Modified: No PCA, only Temp & RTD) */}
      {results && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
          
          {/* Output 1: Melt Temperature */}
          <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-200 flex flex-col justify-between">
            <h3 className="font-bold text-slate-800 mb-4 flex items-center gap-2">
              <Thermometer className="w-5 h-5 text-orange-500" /> Melt Temperature
            </h3>
            <div className="flex flex-col items-center justify-center h-full">
               <div className="text-6xl font-bold text-slate-800 tracking-tighter">
                 {results.temperature}<span className="text-2xl text-slate-400">°C</span>
               </div>
               <div className="text-sm text-orange-500 font-medium mt-4 bg-orange-50 px-4 py-1 rounded-full border border-orange-100">
                 Model: {selectedTestModel}
               </div>
            </div>
          </div>

          {/* Output 2: RTD Curve */}
          <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-200">
            <h3 className="font-bold text-slate-800 mb-4 flex items-center gap-2">
              <TrendingUp className="w-5 h-5 text-blue-500" /> RTD Prediction
            </h3>
            <div className="flex items-center justify-center h-48">
               <RTDChart data={results.rtdCurve} />
            </div>
            <p className="text-xs text-slate-400 mt-3 text-center">
              Residence Time Distribution across active region.
            </p>
          </div>
        </div>
      )}
    </div>
  );

  return (
    <div className="min-h-screen bg-slate-50 font-sans text-slate-900">
      <Header />
      <main className="max-w-7xl mx-auto p-6">
        {mode === 'train' ? <TrainingPanel /> : <TestingPanel />}
      </main>
    </div>
  );
};

export default TwinScrewApp;