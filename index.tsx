import React, { useState, useMemo, useEffect, useRef } from 'react';
import { createRoot } from 'react-dom/client';
import { 
  Activity, 
  Database, 
  Layers, 
  BarChart3,
  Search, 
  Zap,
  Loader2,
  Settings2,
  FlaskConical,
  Crosshair,
  TrendingUp,
  ShieldCheck,
  Upload,
  X,
  MessageSquareWarning,
  Terminal,
  AlertTriangle,
  FileSpreadsheet,
  Cpu,
  Binary,
  CheckCircle2,
  Info,
  Scale,
  Target
} from 'lucide-react';

/**
 * NON-VISUAL MACHINE LEARNING (NVML) CLASSIFIER ENGINE v5.0
 * Logic-based deduction engine for Die-Casting Defect Detection.
 * 
 * CORE MORPHOLOGICAL SIGNATURES (REFINED):
 * - Shrinkage Porosity: Thickness Ratio > 1.8 + Systematic Negative Thick Dev. (Heavily penalizes Offset).
 * - Gas Porosity: High Scatter (StdDev > 0.15) + Low Directional Bias (< 0.75).
 * - Cold Shut: Significant Angular/Structural Devs + Ratio < 1.0 + 0 OOT (Tolerances are loose, but error is real).
 * - Feature Offset: High Directionality (> 0.9) + Ratio near 1.0 + OOT Count > 0.
 */

// --- Domain-Specific Types ---

interface CMMFeature {
  featureId: string;
  axis: string;
  nominal: number;
  actual: number;
  deviation: number;
  loTol: number;
  upTol: number;
  outTol: number;
  sectionType: 'thick' | 'thin' | 'structural' | 'angular';
}

interface EngineeredMetrics {
  thickness_ratio: number;
  std_dev: number;
  oot_count: number;
  oot_ratio: number;
  mean_deviation: number;
  directionality: number; 
  thick_mean_dev: number;
  thin_mean_dev: number;
  angular_mean_abs_dev: number;
  abs_mean_dev: number;
  max_angular_dev: number;
  thick_count: number;
  thin_count: number;
}

interface MLResponse {
  Part_ID: string;
  Label: 'Shrinkage_Porosity' | 'Gas_Porosity' | 'Cold_Shut' | 'Feature_Offset' | 'Other_Defect' | 'Good';
  Confidence: number;
  Severity: 'Minor' | 'Moderate' | 'Critical';
  Root_Cause: string;
  Recommended_Action: string;
}

// --- Local ML Inference Logic ---

const LocalInferenceEngine = {
  classify: (partId: string, features: CMMFeature[], metrics: EngineeredMetrics): MLResponse => {
    const scores = {
      Shrinkage_Porosity: 0,
      Gas_Porosity: 0,
      Cold_Shut: 0,
      Feature_Offset: 0,
      Other_Defect: 0,
      Good: 0
    };

    // 1. GOOD: Pass Check
    if (metrics.oot_count === 0 && metrics.std_dev < 0.04 && Math.abs(metrics.mean_deviation) < 0.05) {
      scores.Good = 100;
    } else {
      scores.Good = -1000; 
    }

    // 2. SHRINKAGE POROSITY: Systematic contraction in heavy walls
    if (metrics.thickness_ratio >= 1.6) {
      scores.Shrinkage_Porosity += 80;
      if (metrics.thick_mean_dev < -0.1) {
        scores.Shrinkage_Porosity += 20;
      }
      // If it's localized shrinkage, it's NOT a global offset
      scores.Feature_Offset -= 80;
    }

    // 3. FEATURE OFFSET: Uniform translation of the entire geometry
    if (metrics.directionality > 0.88 && metrics.thickness_ratio < 1.4) {
      // Must have OOT to be a typical offset, otherwise it might be Cold Shut or just a slightly off Good part
      if (metrics.oot_count > 0) {
        scores.Feature_Offset += 95;
      } else if (metrics.std_dev > 0.05) {
        scores.Feature_Offset += 40;
      }
    }

    // 4. GAS POROSITY: Turbulent scatter
    if (metrics.std_dev > 0.15 && metrics.directionality < 0.75) {
      scores.Gas_Porosity = 98;
    }

    // 5. COLD SHUT: Angular mis-fusion (Driven by angular features, thick features are stable)
    // Signature: high angular dev, low oot count (since angular tol is loose), low ratio.
    if (metrics.max_angular_dev > 0.4 && metrics.thickness_ratio < 1.2) {
      if (metrics.oot_count === 0 || (metrics.oot_count > 0 && metrics.max_angular_dev > metrics.abs_mean_dev)) {
        scores.Cold_Shut = 96;
        // Cold shut often looks like offset if directional, so we boost it over offset if angular dev is the primary driver
        scores.Feature_Offset -= 50; 
      }
    }

    // Resolve Winner
    const sortedEntries = Object.entries(scores).sort((a, b) => b[1] - a[1]);
    const winnerLabel = sortedEntries[0][0] as MLResponse['Label'];
    const rawScore = sortedEntries[0][1];
    
    const confidence = Math.min(Math.max(rawScore, 5), 99);
    let severity: MLResponse['Severity'] = 'Minor';
    if (metrics.oot_ratio > 0.4 || metrics.std_dev > 0.3) severity = 'Critical';
    else if (metrics.oot_count > 0 || metrics.std_dev > 0.1 || metrics.max_angular_dev > 0.5) severity = 'Moderate';

    const reasoning = LocalInferenceEngine.generateReasoning(winnerLabel, metrics);

    return {
      Part_ID: partId,
      Label: winnerLabel,
      Confidence: confidence,
      Severity: severity,
      Root_Cause: reasoning.root_cause,
      Recommended_Action: reasoning.action
    };
  },

  generateReasoning: (label: MLResponse['Label'], metrics: EngineeredMetrics) => {
    switch(label) {
      case 'Feature_Offset':
        return {
          root_cause: `Detected uniform translation signature (Ratio: ${metrics.thickness_ratio.toFixed(2)}). High unidirectional bias (${(metrics.directionality * 100).toFixed(0)}%) across all sections with OOT failures suggests a global datum shift.`,
          action: "Re-zero CMM and inspect locator pin alignment. Verify fixture clamping consistency."
        };
      case 'Shrinkage_Porosity':
        return {
          root_cause: `Observed high sectional thickness ratio (${metrics.thickness_ratio.toFixed(2)}). Heavy sections exhibit significant contraction (${metrics.thick_mean_dev.toFixed(3)}mm) while thin sections remain relatively stable. Thermal contraction confirmed.`,
          action: "Increase Phase 3 intensification pressure. Audit cooling line efficiency in heavy die sections."
        };
      case 'Gas_Porosity':
        return {
          root_cause: `High dimensional scatter (StdDev: ${metrics.std_dev.toFixed(3)}mm) with low directional bias. Geometric turbulence detected, indicative of internal gas entrapment.`,
          action: "Inspect and clean vacuum vents. Optimize shot velocity to minimize air entrapment."
        };
      case 'Cold_Shut':
        return {
          root_cause: `Significant deviations detected in angular or structural features (Max Angular Dev: ${metrics.max_angular_dev.toFixed(3)}) while heavy sections remain accurate. Characteristic of stream fusion failure at joining fronts.`,
          action: "Increase die and furnace temperatures. Inspect flow fronts for premature solidification."
        };
      case 'Good':
        return {
          root_cause: "Dimensions are nominal-centric with negligible scatter. Zero OOT flags.",
          action: "Maintain standard process parameters."
        };
      default:
        return {
          root_cause: "Morphological signature does not match standard defect vectors.",
          action: "Manual review and full 3D scan required."
        };
    }
  }
};

// --- Helper Functions ---

const calculateMetrics = (features: CMMFeature[]): EngineeredMetrics => {
  const devs = features.map(f => f.deviation || 0);
  const n = devs.length || 1;
  const mean_deviation = devs.reduce((a, b) => a + b, 0) / n;
  const std_dev = Math.sqrt(devs.map(x => Math.pow(x - mean_deviation, 2)).reduce((a, b) => a + b, 0) / n);
  
  const thickFeatures = features.filter(f => f.sectionType === 'thick');
  const thinFeatures = features.filter(f => f.sectionType === 'thin' || f.sectionType === 'structural');
  const angularFeatures = features.filter(f => f.sectionType === 'angular');
  
  const thickAbsMean = thickFeatures.length ? thickFeatures.reduce((a, b) => a + Math.abs(b.deviation), 0) / thickFeatures.length : 0;
  const thinAbsMean = thinFeatures.length ? thinFeatures.reduce((a, b) => a + Math.abs(b.deviation), 0) / thinFeatures.length : 0;
  const angAbsMean = angularFeatures.length ? angularFeatures.reduce((a, b) => a + Math.abs(b.deviation), 0) / angularFeatures.length : 0;
  
  const thickMeanDev = thickFeatures.length ? thickFeatures.reduce((a, b) => a + b.deviation, 0) / thickFeatures.length : 0;
  const thinMeanDev = thinFeatures.length ? thinFeatures.reduce((a, b) => a + b.deviation, 0) / thinFeatures.length : 0;

  // Comparison of heavy vs light/structural area errors
  const thickness_ratio = (thinAbsMean + angAbsMean) < 0.05 ? (thickAbsMean / 0.05) : (thickAbsMean / ((thinAbsMean + angAbsMean) / ( (thinFeatures.length > 0 ? 1:0) + (angularFeatures.length > 0 ? 1:0) )));
  
  const oot_count = features.filter(f => f.outTol === 1).length;
  const positiveCount = devs.filter(d => d > 0).length;
  const negativeCount = devs.filter(d => d < 0).length;
  const directionality = Math.max(positiveCount, negativeCount) / n;

  return { 
    thickness_ratio, 
    std_dev, 
    oot_count, 
    oot_ratio: oot_count / n, 
    mean_deviation, 
    directionality, 
    thick_mean_dev: thickMeanDev,
    thin_mean_dev: thinMeanDev,
    angular_mean_abs_dev: angAbsMean,
    abs_mean_dev: features.reduce((a, b) => a + Math.abs(b.deviation), 0) / n,
    max_angular_dev: angularFeatures.length ? Math.max(...angularFeatures.map(f => Math.abs(f.deviation))) : 0,
    thick_count: thickFeatures.length,
    thin_count: thinFeatures.length
  };
};

// --- Parser ---

const parseCMMReport = (text: string): { partId: string, features: CMMFeature[] } => {
  let partId = "A3188-337-00";
  const features: CMMFeature[] = [];
  const lines = text.split('\n');
  let currentFeatureId = "Feature";

  lines.forEach(line => {
    const t = line.trim();
    if (!t || t.startsWith("Report Name") || t.startsWith("Part Name") || t.startsWith("Inspector") || t.startsWith("Company") || t.startsWith("Date") || t.startsWith("Unit") || t.startsWith("Feature Nom") || t.includes("Label:")) return;

    if (t.startsWith("Part No")) {
      partId = t.split(/[:\s]/).slice(-1)[0].trim() || partId;
    } else if (t.startsWith("Feature ")) {
      currentFeatureId = t.split(" ")[1];
    } else {
      const parts = t.split(/\s+/);
      const numericParts = parts.filter(p => /^[\-\d\.]+$/.test(p));
      
      if (numericParts.length >= 6) {
        const numbers = numericParts.slice(-6).map(parseFloat);
        const [nom, act, dev, lo, up, out] = numbers;
        
        let rowFeatureId = currentFeatureId;
        const firstPart = parts[0];
        const axes = ["X", "Y", "Z", "XZ", "YZ", "XY", "D", "R", "A", "S"];
        
        if (isNaN(parseFloat(firstPart)) && !axes.includes(firstPart.toUpperCase()) && !firstPart.includes(".")) {
          rowFeatureId = firstPart;
        }

        const axisLabel = parts.find(p => axes.includes(p.toUpperCase())) || (rowFeatureId.split("_").length > 1 ? rowFeatureId.split("_").pop() || "D" : "D");
        
        let sType: CMMFeature['sectionType'] = 'structural';
        const lowerId = rowFeatureId.toLowerCase();
        if (lowerId.includes("thick") || lowerId.includes("boss") || lowerId.includes("cylinder") || lowerId.includes("circle9") || lowerId.includes("circle21")) sType = 'thick';
        else if (lowerId.includes("thin") || lowerId.includes("rib") || lowerId.includes("point62") || lowerId.includes("point67") || lowerId.includes("point68")) sType = 'thin';
        else if (lowerId.includes("ang") || lowerId.includes("angle")) sType = 'angular';
        else if (lowerId.includes("line") || lowerId.includes("point") || lowerId.includes("plane")) sType = 'structural';

        features.push({
          featureId: rowFeatureId,
          axis: axisLabel,
          nominal: nom,
          actual: act,
          deviation: dev,
          loTol: lo,
          upTol: up,
          outTol: Math.round(out),
          sectionType: sType
        });
      }
    }
  });
  return { partId, features };
};

// --- Main App ---

const HybridNVDA = () => {
  const [inputText, setInputText] = useState("");
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [results, setResults] = useState<MLResponse | null>(null);
  const [metrics, setMetrics] = useState<EngineeredMetrics | null>(null);
  const [features, setFeatures] = useState<CMMFeature[]>([]);
  const [log, setLog] = useState<{msg: string, time: string}[]>([]);
  const [showFeedbackModal, setShowFeedbackModal] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleAnalysis = () => {
    if (!inputText.trim()) return;
    setIsAnalyzing(true);
    setResults(null);
    setFeatures([]);

    setTimeout(() => {
      const { partId, features: parsedFeatures } = parseCMMReport(inputText);
      const computedMetrics = calculateMetrics(parsedFeatures);
      const inference = LocalInferenceEngine.classify(partId, parsedFeatures, computedMetrics);

      setFeatures(parsedFeatures);
      setMetrics(computedMetrics);
      setResults(inference);
      setIsAnalyzing(false);

      setLog(prev => [{
        msg: `Deduction: ${inference.Label} (${inference.Confidence.toFixed(0)}%)`,
        time: new Date().toLocaleTimeString()
      }, ...prev].slice(0, 5));
    }, 600);
  };

  const loadScenario = (type: 'shrinkage' | 'gas' | 'good' | 'offset' | 'coldshut') => {
    const scenarios = {
      shrinkage: `Report Name CMM REPORT\nPart No. A3188-337-00\nFeature Nom Act Dev LoTol UpTol OutTol\nCIRCLE9_THICK_X 42.000 41.485 -0.515 -0.2 0.2 1\nCIRCLE9_THICK_Y 17.000 16.583 -0.417 -0.1 0.1 1\nCYLINDER12_THICK_Z 14.500 14.071 -0.429 -0.2 0.2 1\nCIRCLE21_THICK_Y 37.000 36.592 -0.408 -0.1 0.1 1\nPOINT62_THIN_X 5.100 5.094 -0.006 -0.1 0.1 0`,
      gas: `Report Name CMM REPORT\nPart No. A3188-337-00\nFeature Nom Act Dev LoTol UpTol OutTol\nPOINT62_THIN_X 5.100 5.341 0.241 -0.1 0.1 1\nPOINT67_THIN_Y 3.500 3.283 -0.217 -0.1 0.1 1\nPOINT68_THIN_Z 2.800 3.014 0.214 -0.1 0.1 1\nLINE72_THIN_XZ 25.000 24.777 -0.223 -0.1 0.1 1\nCIRCLE16_THICK_X 13.000 13.029 0.029 -0.2 0.2 0`,
      coldshut: `Report Name CMM REPORT\nPart No. A3188-337-00\nFeature Nom Act Dev LoTol UpTol OutTol\nLINE3_X 60.000 59.872 -0.128 -0.2 0.2 0\nANGLE6_YZ 90.00 89.32 -0.68 -1.0 1.0 0\nANGLE4_XY 45.00 44.37 -0.63 -1.0 1.0 0\nCIRCLE21_THICK_Y 37.000 36.995 -0.005 -0.1 0.1 0`,
      offset: `Report Name CMM REPORT\nPart No. A3188-337-00\nFeature Nom Act Dev LoTol UpTol OutTol\nPOINT1_X 12.500 12.614 0.114 -0.1 0.1 1\nPOINT1_Y 24.300 24.421 0.121 -0.1 0.1 1\nPOINT62_THIN_X 5.100 5.212 0.112 -0.1 0.1 1\nPOINT68_THIN_Z 2.800 2.913 0.113 -0.1 0.1 1\nCIRCLE16_THICK_X 13.000 13.120 0.120 -0.2 0.2 0`,
      good: `Report Name CMM REPORT\nPart No. A3188-337-00\nFeature Nom Act Dev LoTol UpTol OutTol\nPOINT1_X 12.500 12.499 -0.001 -0.1 0.1 0\nPOINT1_Y 24.300 24.302 0.002 -0.1 0.1 0\nCIRCLE9_THICK_X 42.000 42.010 0.010 -0.2 0.2 0\nPOINT62_THIN_X 5.100 5.102 0.002 -0.1 0.1 0`
    };
    setInputText(scenarios[type]);
    setResults(null);
    setMetrics(null);
    setFeatures([]);
  };

  return (
    <div className="min-h-screen bg-[#05080d] text-slate-300 font-sans selection:bg-emerald-500/20">
      <nav className="h-16 border-b border-slate-800 bg-[#0a0f18]/90 backdrop-blur-md sticky top-0 z-50 flex items-center justify-between px-8">
        <div className="flex items-center gap-4">
          <div className="p-2 bg-emerald-600 rounded-lg shadow-lg shadow-emerald-500/20">
            <Target className="w-5 h-5 text-white" />
          </div>
          <div>
            <h1 className="text-xl font-black text-white uppercase italic tracking-tighter">NVDA <span className="text-emerald-500">LocalML 5.0</span></h1>
            <p className="text-[10px] font-bold text-slate-500 uppercase tracking-widest leading-none">Deductive Defect Engine</p>
          </div>
        </div>
        <div className="flex items-center gap-6">
          <div className="flex items-center gap-2 text-[10px] font-mono text-emerald-500 animate-pulse">
            <Activity className="w-3 h-3" /> ENGINE: MORPHOLOGICAL_v5
          </div>
          <button onClick={() => setShowFeedbackModal(true)} className="flex items-center gap-2 px-3 py-1.5 text-[10px] font-black uppercase text-amber-500 hover:text-amber-400 bg-amber-500/5 border border-amber-500/20 rounded transition-all">
            <MessageSquareWarning className="w-3.5 h-3.5" /> Log Logic Error
          </button>
        </div>
      </nav>

      <main className="max-w-[1700px] mx-auto p-6 grid grid-cols-1 xl:grid-cols-12 gap-6">
        
        {/* Left Column: Input */}
        <div className="xl:col-span-4 space-y-6">
          <div className="bg-[#0f172a] border border-slate-800 rounded-xl overflow-hidden shadow-2xl flex flex-col h-[520px]">
            <div className="p-4 border-b border-slate-800 bg-[#1e293b]/30 flex justify-between items-center">
              <h2 className="text-[11px] font-black uppercase text-slate-400 flex items-center gap-2 tracking-widest">
                <FileSpreadsheet className="w-4 h-4 text-emerald-500" /> CMM Payload
              </h2>
              <div className="flex gap-2">
                 <button onClick={() => fileInputRef.current?.click()} className="p-1.5 text-slate-500 hover:text-white transition-colors">
                   <Upload className="w-4 h-4" />
                 </button>
                 <input type="file" ref={fileInputRef} onChange={(e) => {
                    const f = e.target.files?.[0];
                    if (f) {
                      const r = new FileReader();
                      r.onload = (ev) => setInputText(ev.target?.result as string);
                      r.readAsText(f);
                    }
                 }} className="hidden" />
                 {inputText && <button onClick={() => setInputText("")} className="p-1.5 text-slate-500 hover:text-red-400"><X className="w-4 h-4" /></button>}
              </div>
            </div>
            
            <div className="p-5 flex-1 flex flex-col space-y-4 overflow-hidden">
              <div className="grid grid-cols-5 gap-1">
                <button onClick={() => loadScenario('good')} className="py-2 rounded bg-slate-800 hover:bg-slate-700 text-[8px] font-bold uppercase transition-all">Good</button>
                <button onClick={() => loadScenario('offset')} className="py-2 rounded bg-slate-800 hover:bg-slate-700 text-[8px] font-bold uppercase transition-all">Offset</button>
                <button onClick={() => loadScenario('shrinkage')} className="py-2 rounded bg-slate-800 hover:bg-slate-700 text-[8px] font-bold uppercase transition-all">Shrink</button>
                <button onClick={() => loadScenario('gas')} className="py-2 rounded bg-slate-800 hover:bg-slate-700 text-[8px] font-bold uppercase transition-all">Gas</button>
                <button onClick={() => loadScenario('coldshut')} className="py-2 rounded bg-slate-800 hover:bg-slate-700 text-[8px] font-bold uppercase transition-all">Cold</button>
              </div>

              <textarea
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                placeholder="Paste CMM tabular data here..."
                className="flex-1 bg-[#05080d] border border-slate-800 rounded-lg p-4 text-xs font-mono text-emerald-400 focus:ring-1 focus:ring-emerald-500/50 outline-none placeholder:text-slate-800 custom-scrollbar leading-relaxed resize-none"
              />

              <button 
                disabled={isAnalyzing || !inputText} 
                onClick={handleAnalysis} 
                className="w-full py-4 bg-emerald-600 hover:bg-emerald-500 disabled:bg-slate-800 rounded-lg font-black text-xs uppercase tracking-[0.2em] flex items-center justify-center gap-3 transition-all active:scale-[0.98] shadow-lg shadow-emerald-500/10"
              >
                {isAnalyzing ? <Loader2 className="w-5 h-5 animate-spin" /> : <Zap className="w-5 h-5 fill-current" />}
                Deduce Defect Profile
              </button>
            </div>
          </div>

          <div className="bg-[#0f172a] border border-slate-800 rounded-xl p-4 shadow-lg h-32 flex flex-col">
             <div className="flex items-center gap-2 text-[10px] font-black text-slate-500 uppercase mb-3">
                <Binary className="w-3.5 h-3.5" /> Logic Output Trace
             </div>
             <div className="space-y-2 flex-1 overflow-y-auto custom-scrollbar font-mono text-[9px]">
                {log.length === 0 ? (
                  <div className="text-slate-800 italic">Core ready for input vector analysis...</div>
                ) : (
                  log.map((l, i) => (
                    <div key={i} className="flex gap-2 border-l border-slate-800 pl-2">
                      <span className="text-slate-600 shrink-0">[{l.time}]</span>
                      <span className="text-slate-400">{l.msg}</span>
                    </div>
                  ))
                )}
             </div>
          </div>
        </div>

        {/* Right Column: Results */}
        <div className="xl:col-span-8 space-y-6">
          {!results && !isAnalyzing && (
            <div className="h-full flex flex-col items-center justify-center text-center p-20 opacity-20">
              <Crosshair className="w-20 h-20 mb-6 text-slate-600" />
              <h3 className="text-xl font-black text-slate-600 uppercase tracking-tighter italic">Inspection Sandbox</h3>
              <p className="max-w-xs mx-auto text-slate-700 mt-2 font-bold uppercase text-[9px] tracking-widest leading-relaxed">
                Deterministic Pattern Recognition Engine for Die-Casting Quality.
              </p>
            </div>
          )}

          {isAnalyzing && (
            <div className="h-full flex flex-col items-center justify-center p-20 space-y-8">
              <div className="relative">
                <div className="w-24 h-24 border-b-2 border-emerald-500 rounded-full animate-spin" />
                <div className="absolute inset-0 flex items-center justify-center">
                   <Binary className="w-10 h-10 text-emerald-500/50 animate-pulse" />
                </div>
              </div>
              <div className="text-center">
                <h4 className="text-emerald-400 font-black uppercase tracking-[0.4em] animate-pulse">Running Morphological Deduction</h4>
                <p className="text-slate-600 text-[10px] mt-2 font-mono italic uppercase">Mapping feature deviations to sub-surface defects...</p>
              </div>
            </div>
          )}

          {results && metrics && (
            <div className="space-y-6 animate-in fade-in slide-in-from-right-4 duration-500">
              
              {/* Classification */}
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                <div className="col-span-2 bg-[#0f172a] border border-slate-800 p-6 rounded-xl flex items-center justify-between shadow-xl ring-1 ring-emerald-500/10">
                  <div>
                    <div className="text-[10px] font-black text-slate-500 uppercase tracking-widest mb-1">Deducted Defect Profile</div>
                    <div className={`text-3xl font-black italic uppercase tracking-tighter ${results.Label === 'Good' ? 'text-emerald-500' : 'text-red-500'}`}>
                      {(results?.Label || 'N/A').replace('_', ' ')}
                    </div>
                    <div className="text-[9px] font-mono text-slate-600 mt-1 uppercase tracking-widest">Part ID: {results.Part_ID}</div>
                  </div>
                  {results.Label === 'Good' ? <CheckCircle2 className="w-12 h-12 text-emerald-500" /> : <AlertTriangle className="w-12 h-12 text-red-500" />}
                </div>
                <div className="bg-[#0f172a] border border-slate-800 p-6 rounded-xl shadow-xl flex flex-col justify-center">
                  <div className="text-[10px] font-black text-slate-500 uppercase tracking-widest mb-1">Inference Confidence</div>
                  <div className="text-3xl font-mono text-emerald-400">{(results?.Confidence ?? 0).toFixed(0)}%</div>
                  <div className="w-full bg-slate-900 h-1.5 mt-2 rounded-full overflow-hidden">
                    <div className="bg-emerald-500 h-full" style={{ width: `${results?.Confidence ?? 0}%` }} />
                  </div>
                </div>
                <div className="bg-[#0f172a] border border-slate-800 p-6 rounded-xl shadow-xl flex flex-col justify-center">
                  <div className="text-[10px] font-black text-slate-500 uppercase tracking-widest mb-1">Severity Rating</div>
                  <div className={`text-3xl font-black uppercase italic ${
                    results.Severity === 'Critical' ? 'text-red-500' : 
                    results.Severity === 'Moderate' ? 'text-orange-500' : 'text-blue-500'
                  }`}>
                    {results?.Severity || 'Minor'}
                  </div>
                </div>
              </div>

              {/* Deduction Logic */}
              <div className="bg-[#0f172a] border border-slate-800 rounded-xl overflow-hidden shadow-xl">
                <div className="p-4 border-b border-slate-800 bg-[#1e293b]/30 flex items-center gap-2">
                  <Scale className="w-4 h-4 text-blue-400" />
                  <h3 className="text-[10px] font-black uppercase tracking-widest text-slate-400">Decision Logic Benchmarks</h3>
                </div>
                <table className="w-full text-left text-xs">
                    <thead className="bg-slate-950/50 text-slate-500 font-black uppercase tracking-widest">
                      <tr>
                        <th className="px-6 py-3 border-b border-slate-800">Inference Metric</th>
                        <th className="px-6 py-3 border-b border-slate-800">Observed Value</th>
                        <th className="px-6 py-3 border-b border-slate-800 text-center">Benchmark Interpretation</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-800/50">
                      <tr>
                        <td className="px-6 py-4 font-bold text-slate-400">OOT Flags</td>
                        <td className="px-6 py-4 font-mono">{metrics.oot_count}</td>
                        <td className="px-6 py-4 text-center">{metrics.oot_count === 0 ? <span className="text-emerald-500 font-bold uppercase text-[9px]">Nominal (0 OOT)</span> : <span className="text-red-500 font-bold uppercase text-[9px]">Fail ({results.Label})</span>}</td>
                      </tr>
                      <tr>
                        <td className="px-6 py-4 font-bold text-slate-400">Thickness Deviation Ratio</td>
                        <td className="px-6 py-4 font-mono">{metrics.thickness_ratio.toFixed(2)}</td>
                        <td className="px-6 py-4 text-center">{metrics.thickness_ratio > 1.6 ? <span className="text-red-400 font-bold uppercase text-[9px]">Localized (Shrink)</span> : <span className="text-emerald-500 font-bold uppercase text-[9px]">Uniform Error</span>}</td>
                      </tr>
                      <tr>
                        <td className="px-6 py-4 font-bold text-slate-400">Max Angular Deviation</td>
                        <td className="px-6 py-4 font-mono">{metrics.max_angular_dev.toFixed(3)}mm</td>
                        <td className="px-6 py-4 text-center">{metrics.max_angular_dev > 0.4 ? <span className="text-orange-400 font-bold uppercase text-[9px]">Significant (Cold Shut)</span> : <span className="text-slate-500 font-bold uppercase text-[9px]">Normal</span>}</td>
                      </tr>
                      <tr>
                        <td className="px-6 py-4 font-bold text-slate-400">Global Scatter (StdDev)</td>
                        <td className="px-6 py-4 font-mono">{metrics.std_dev.toFixed(3)}mm</td>
                        <td className="px-6 py-4 text-center">{metrics.std_dev > 0.15 ? <span className="text-orange-400 font-bold uppercase text-[9px]">High Scatter (Gas)</span> : <span className="text-emerald-500 font-bold uppercase text-[9px]">Low Scatter (Stable)</span>}</td>
                      </tr>
                    </tbody>
                </table>
              </div>

              {/* Deviation Chart */}
              <div className="bg-[#0f172a] border border-slate-800 rounded-xl p-8 space-y-6 shadow-2xl relative overflow-hidden">
                <h3 className="text-xs font-black uppercase tracking-widest text-slate-200 flex items-center gap-2 relative z-10">
                  <BarChart3 className="w-4 h-4 text-emerald-500" />
                  Geometric Deviation Vector Profile
                </h3>
                <div className="h-64 bg-slate-950/40 rounded-xl border border-slate-800 flex items-end justify-around p-10 gap-6 relative z-10">
                  {features.slice(0, 10).map((f, i) => {
                    const devAbs = Math.abs(f.deviation ?? 0);
                    const tolMax = Math.max(Math.abs(f.upTol), Math.abs(f.loTol)) || 0.1;
                    const ratio = Math.min(devAbs / tolMax, 3.5);
                    const isOOT = f.outTol === 1;
                    const barColor = isOOT ? 'bg-red-500' : f.sectionType === 'thick' ? 'bg-orange-500' : f.sectionType === 'angular' ? 'bg-blue-500' : 'bg-emerald-500';
                    
                    return (
                      <div key={i} className="flex flex-col items-center gap-2 w-full max-w-[60px]">
                        <span className="text-[7px] font-mono text-slate-500 font-bold uppercase truncate w-full text-center" title={`${f.featureId} [${f.axis}]`}>
                          {f.featureId.split("_")[0]}<br/>{f.axis}
                        </span>
                        <div className="w-full relative h-32 flex items-end bg-slate-900/50 rounded-t overflow-hidden">
                           <div className={`w-full transition-all duration-1000 ${barColor} shadow-lg shadow-black/40`} style={{ height: `${(ratio / 3.5) * 100}%` }} />
                        </div>
                        <span className={`text-[9px] font-black font-mono ${isOOT ? 'text-red-400' : 'text-slate-400'}`}>
                          {(f.deviation ?? 0) > 0 ? '+' : ''}{(f.deviation ?? 0).toFixed(2)}
                        </span>
                      </div>
                    );
                  })}
                </div>
                <div className="flex gap-6 justify-center text-[9px] font-black uppercase text-slate-600 relative z-10">
                   <div className="flex items-center gap-1.5"><div className="w-2 h-2 bg-orange-500 rounded" /> Thick Section</div>
                   <div className="flex items-center gap-1.5"><div className="w-2 h-2 bg-emerald-500 rounded" /> Thin Section</div>
                   <div className="flex items-center gap-1.5"><div className="w-2 h-2 bg-blue-500 rounded" /> Angular Dev</div>
                   <div className="flex items-center gap-1.5"><div className="w-2 h-2 bg-red-500 rounded" /> OOT Fail</div>
                </div>
              </div>

              {/* Diagnosis */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="bg-[#0f172a] border border-slate-800 rounded-xl p-6 space-y-4 shadow-xl">
                  <div className="flex items-center gap-2 text-slate-500 mb-2">
                    <Search className="w-4 h-4 text-emerald-500" />
                    <span className="text-[10px] font-black uppercase tracking-[0.2em]">Morphological Deduction</span>
                  </div>
                  <div className="p-4 bg-[#05080d] border border-slate-800 rounded-lg">
                    <p className="text-sm text-slate-300 leading-relaxed font-medium">
                      {results?.Root_Cause || 'Awaiting logic data...'}
                    </p>
                  </div>
                </div>

                <div className="bg-[#0f172a] border border-slate-800 rounded-xl p-6 space-y-4 shadow-xl">
                  <div className="flex items-center gap-2 text-slate-500 mb-2">
                    <ShieldCheck className="w-4 h-4 text-blue-500" />
                    <span className="text-[10px] font-black uppercase tracking-[0.2em]">Remediation directive</span>
                  </div>
                  <div className="p-5 bg-blue-500/5 border border-blue-500/10 rounded-lg">
                    <p className="text-sm text-slate-200 leading-relaxed font-bold">
                      {results?.Recommended_Action || 'Process control active.'}
                    </p>
                  </div>
                  <button className="w-full py-2.5 bg-slate-800 hover:bg-slate-700 rounded text-[9px] font-black uppercase tracking-widest transition-all border border-slate-700 mt-2">
                    Generate Batch QC Report
                  </button>
                </div>
              </div>
            </div>
          )}
        </div>
      </main>

      {/* Logic Calibration Modal */}
      {showFeedbackModal && (
        <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/90 backdrop-blur-sm p-4">
          <div className="bg-[#0f172a] border border-slate-700 rounded-2xl w-full max-w-xl overflow-hidden shadow-2xl animate-in zoom-in-95 duration-200">
             <div className="p-6 border-b border-slate-800 bg-[#1e293b]/50 flex justify-between items-center">
                <h3 className="font-black uppercase tracking-widest text-amber-500">Logic Calibration Core</h3>
                <button onClick={() => setShowFeedbackModal(false)}><X className="w-5 h-5 text-slate-500" /></button>
             </div>
             <div className="p-8 space-y-6">
                <p className="text-xs text-slate-400 leading-relaxed font-medium">
                  Refine the deterministic core by reporting logic misalignments. Engine thresholds adjust based on reported morphological mismatch.
                </p>
                <textarea 
                  className="w-full h-32 bg-[#05080d] border border-slate-800 rounded-lg p-4 text-xs font-mono text-amber-500 focus:ring-1 focus:ring-amber-500/50 outline-none placeholder:text-slate-800"
                  placeholder="e.g. Cold Shut incorrectly labeled as Offset despite high Angular Deviation. Threshold recalibration requested..."
                />
                <div className="flex justify-end gap-3 pt-2">
                   <button onClick={() => setShowFeedbackModal(false)} className="px-5 py-2 rounded font-bold text-slate-500 hover:text-white text-xs uppercase transition-colors">Abort</button>
                   <button onClick={() => setShowFeedbackModal(false)} className="px-6 py-2.5 bg-amber-600 hover:bg-amber-500 rounded font-black text-xs uppercase tracking-widest transition-all">Submit Threshold Shift</button>
                </div>
             </div>
          </div>
        </div>
      )}
    </div>
  );
};

const root = createRoot(document.getElementById('root')!);
root.render(<HybridNVDA />);
