import React, { useEffect, useRef } from 'react';
import mermaid from 'mermaid';
import { BrainCircuit, Activity, Shrink, Zap, ShieldAlert, Cpu, Github, Package, CheckCircle2, Factory } from 'lucide-react';

// Initialize mermaid for Light Mode Premium Academic layout
mermaid.initialize({
  startOnLoad: true,
  theme: 'base',
  themeVariables: {
    fontFamily: 'Inter, sans-serif',
    primaryColor: '#ffffff',
    primaryTextColor: '#1e293b',
    primaryBorderColor: '#cbd5e1',
    lineColor: '#6366f1',
    secondaryColor: '#f8fafc',
    tertiaryColor: '#f1f5f9',
    edgeLabelBackground: '#ffffff'
  }
});

const MermaidDiagram = ({ chart }) => {
  const ref = useRef(null);

  useEffect(() => {
    if (ref.current) {
      mermaid.render(`mermaid-${Math.random().toString(36).substr(2, 9)}`, chart).then((result) => {
        ref.current.innerHTML = result.svg;
      });
    }
  }, [chart]);

  return <div ref={ref} className="flex justify-center my-8 p-8 glass rounded-3xl overflow-x-auto shadow-md shadow-indigo-100/50" />;
};

const FeatureCard = ({ icon: Icon, title, description }) => (
  <div className="glass-card">
    <div className="h-14 w-14 rounded-2xl bg-indigo-50 flex items-center justify-center mb-6 border border-indigo-100/50 text-indigo-600 shadow-sm shadow-indigo-100">
      <Icon size={26} strokeWidth={2} />
    </div>
    <h3 className="text-xl font-semibold mb-3 text-slate-800">{title}</h3>
    <p className="text-slate-500 leading-relaxed">{description}</p>
  </div>
);

function App() {
  const architectureChart = `
    flowchart TD
      classDef tsNode fill:#ffffff,stroke:#8b5cf6,stroke-width:2px,color:#334155,rx:12px,ry:12px
      classDef pyNode fill:#f8fafc,stroke:#3b82f6,stroke-width:2px,color:#334155,rx:12px,ry:12px
      classDef endpoint fill:#eff6ff,stroke:#60a5fa,stroke-width:2px,color:#1e3a8a,rx:8px,ry:8px

      subgraph "TypeScript Agentic Orchestrator"
      direction TB
        B[Agentic Query Engine<br/>Bun / TypeScript]:::tsNode
        E{Artifact Rejection Tool<br/>Zod Schemas}:::tsNode
      end

      subgraph "Python OmniPulse Engine (PyPI)"
      direction TB
        C[Wavelet Scattering Transform<br/>Kymatio Backend]:::pyNode
        D[PCA Manifold Compression<br/>scikit-learn]:::pyNode
      end

      A([Noisy Signal<br>EEG / FRB]):::endpoint -->|Continuous Input| B
      B -.->|MCP stdio Transport| C
      C -->|S0, S1, S2 Tensors| D
      D -->|K-dimensional Vectors| E
      E -->|Mean + 3σ Analysis| F([Clean Foundation Tokens]):::endpoint
      E -->|Anomalous Variance| G([Denoise / Halt]):::endpoint
  `;

  return (
    <div className="min-h-screen selection:bg-indigo-100 selection:text-indigo-900">
      {/* Navigation */}
      <nav className="fixed w-full z-50 glass border-b border-slate-200/50">
        <div className="max-w-7xl mx-auto px-6 h-20 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="h-10 w-10 bg-gradient-to-tr from-blue-600 to-indigo-600 rounded-xl flex items-center justify-center shadow-md shadow-indigo-200">
              <Activity className="text-white" size={20} />
            </div>
            <span className="font-bold text-xl tracking-tight text-slate-800">OmniPulse</span>
          </div>
          <div className="hidden md:flex items-center gap-8 text-sm font-medium text-slate-500">
            <a href="#features" className="hover:text-indigo-600 transition-colors">Architecture</a>
            <a href="#pipeline" className="hover:text-indigo-600 transition-colors">Pipeline</a>
            <a href="#enterprise" className="hover:text-indigo-600 transition-colors">Enterprise</a>
            <a href="https://github.com/samvardhan03/OmniPulse" target="_blank" rel="noreferrer" className="flex items-center gap-2 text-slate-600 hover:text-slate-900 transition-colors">
              <Github size={20} /> GitHub
            </a>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="pt-40 pb-24 px-6 relative overflow-hidden flex flex-col items-center">
        {/* Abstract Background Elements for Light Mode */}
        <div className="absolute top-1/4 left-1/4 w-[600px] h-[600px] bg-blue-100/50 rounded-full blur-[120px] -z-10 animate-pulse" />
        <div className="absolute top-1/3 right-1/4 w-[500px] h-[500px] bg-indigo-100/50 rounded-full blur-[100px] -z-10" />
        
        <div className="max-w-4xl mx-auto text-center">
          <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full border border-indigo-200 bg-indigo-50 text-indigo-700 text-sm font-medium mb-8 shadow-sm">
            <span className="relative flex h-2 w-2">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-indigo-400 opacity-75"></span>
              <span className="relative inline-flex rounded-full h-2 w-2 bg-indigo-500"></span>
            </span>
            v0.1.0 Released on PyPI
          </div>
          <h1 className="text-5xl md:text-7xl font-extrabold tracking-tight mb-8 text-slate-900">
            OmniPulse: Universal Transient <br/>
            <span className="text-gradient">Detection Pipeline.</span>
          </h1>
          <p className="text-lg md:text-xl text-slate-500 mb-12 max-w-2xl mx-auto leading-relaxed">
            Domain-agnostic Agentic MLOps for high-noise time-series data. 
            Isolate non-stationary anomalies across EEG and astrophysical telemetry with autonomous mathematical rigor.
          </p>
          <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
            <a href="https://github.com/samvardhan03/OmniPulse" target="_blank" rel="noreferrer" className="h-14 px-8 rounded-full bg-slate-900 text-white font-semibold flex items-center gap-2 shadow-lg shadow-slate-300 hover:shadow-slate-400 hover:-translate-y-0.5 transition-all w-full sm:w-auto justify-center">
              View on GitHub <Github size={18} />
            </a>
            <a href="https://pypi.org/project/omnipulse/" target="_blank" rel="noreferrer" className="h-14 px-8 rounded-full bg-white border border-slate-200 text-indigo-600 font-semibold flex items-center gap-2 hover:bg-slate-50 hover:border-indigo-200 shadow-sm transition-all w-full sm:w-auto justify-center">
              View on PyPI <Package size={18} />
            </a>
          </div>
        </div>
      </section>

      {/* Visual Diagram Section */}
      <section id="pipeline" className="py-24 px-6 relative z-10 bg-slate-100/50">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-3xl md:text-4xl font-bold mb-4 text-slate-900">Pipeline Architecture</h2>
            <p className="text-slate-500 max-w-2xl mx-auto text-lg">
              A decoupled, fully autonomous orchestrator bridging a TypeScript Protocol Client with an advanced signal processing engine.
            </p>
          </div>
          <MermaidDiagram chart={architectureChart} />
        </div>
      </section>

      {/* Features Grid */}
      <section id="features" className="py-24 px-6 relative z-10">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-3xl md:text-4xl font-bold mb-4 text-slate-900">Core Capabilities</h2>
            <p className="text-slate-500 max-w-2xl mx-auto text-lg">
              Next-generation signal decomposition mapped natively to agentic workflows.
            </p>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
            <FeatureCard 
              icon={Activity}
              title="Kymatio WST"
              description="Cascaded Wavelet Scattering Transforms extracting non-stationary transients from extreme high-noise environments."
            />
            <FeatureCard 
              icon={Shrink}
              title="Manifold Compression"
              description="Algorithmic dimensionality reduction to a compact K-dimensional plane retaining 95% of statistical variance."
            />
            <FeatureCard 
              icon={BrainCircuit}
              title="Agentic Orchestrator"
              description="Dynamic TS-powered LLM loops managing mathematical transformations and JSON schema payload parsing."
            />
            <FeatureCard 
              icon={ShieldAlert}
              title="Anomaly Thresholds"
              description="Dual-mode rejection utilizing rigorous 'mean + 3σ' statistical gating to autonomously flag corruption streams."
            />
            <FeatureCard 
              icon={Zap}
              title="MCP Tool Linking"
              description="Type-safe bridging isolating Node.js process state from the underlying Python mathematical engine."
            />
            <FeatureCard 
              icon={Cpu}
              title="Foundation Extensibility"
              description="Ready for PyTorch contrastive alignments. Easily project temporal data arrays into generalized transformer spaces."
            />
          </div>
        </div>
      </section>

      {/* Enterprise Consulting */}
      <section id="enterprise" className="py-24 px-6 bg-slate-900 text-white relative overflow-hidden">
        <div className="absolute inset-0 bg-[url('https://www.transparenttextures.com/patterns/cubes.png')] opacity-10"></div>
        <div className="max-w-7xl mx-auto relative z-10">
          <div className="text-center mb-16">
            <h2 className="text-3xl md:text-4xl font-bold mb-4">Enterprise & Lab Integrations</h2>
            <p className="text-slate-400 max-w-2xl mx-auto text-lg">
              Whether you are an astrophysics research lab or an industrial IoT cluster, OmniPulse scales to your demands.
            </p>
          </div>
          
          <div className="grid md:grid-cols-2 gap-8 max-w-5xl mx-auto">
            {/* Free Tier */}
            <div className="bg-slate-800 border border-slate-700 rounded-3xl p-10 flex flex-col">
              <h3 className="text-2xl font-bold mb-2">Academic / Open Source</h3>
              <p className="text-slate-400 mb-8">For researchers and independent developers.</p>
              <div className="text-4xl font-extrabold mb-8">$0<span className="text-xl text-slate-500 font-medium">/forever</span></div>
              <ul className="space-y-4 mb-10 flex-grow">
                <li className="flex items-center gap-3 text-slate-300"><CheckCircle2 className="text-indigo-400" size={20}/> Core Python Engine (PyPI)</li>
                <li className="flex items-center gap-3 text-slate-300"><CheckCircle2 className="text-indigo-400" size={20}/> TypeScript Orchestrator Agent</li>
                <li className="flex items-center gap-3 text-slate-300"><CheckCircle2 className="text-indigo-400" size={20}/> Standard MCP Documentation</li>
                <li className="flex items-center gap-3 text-slate-300"><CheckCircle2 className="text-indigo-400" size={20}/> GitHub Community Support</li>
              </ul>
            </div>

            {/* Pro Tier */}
            <div className="bg-gradient-to-b from-indigo-900 to-slate-800 border border-indigo-500/30 rounded-3xl p-10 flex flex-col relative shadow-2xl shadow-indigo-900/50 hover:-translate-y-1 transition-transform">
              <div className="absolute top-0 right-0 bg-indigo-500 text-white text-xs font-bold px-4 py-1 rounded-bl-xl rounded-tr-3xl uppercase tracking-wider">
                Recommended
              </div>
              <h3 className="text-2xl font-bold mb-2 text-white">Pro / Custom Lab Implementation</h3>
              <p className="text-indigo-200 mb-8">For enterprise and large-scale research initiatives.</p>
              <div className="text-4xl font-extrabold mb-8">Custom<span className="text-xl text-indigo-300 font-medium"> build</span></div>
              <ul className="space-y-4 mb-10 flex-grow">
                <li className="flex items-center gap-3 text-white font-medium"><Factory className="text-indigo-300" size={20}/> Proprietary Data Ingestion Pipelines</li>
                <li className="flex items-center gap-3 text-white font-medium"><BrainCircuit className="text-indigo-300" size={20}/> Custom Quantum Kernel Integrations</li>
                <li className="flex items-center gap-3 text-white font-medium"><Cpu className="text-indigo-300" size={20}/> Exascale Multi-Node Orchestration</li>
                <li className="flex items-center gap-3 text-white font-medium"><ShieldAlert className="text-indigo-300" size={20}/> SLA & Dedicated Support</li>
              </ul>
              <a href="mailto:shekhawatsamvardhan@gmail.com" className="w-full h-14 bg-white text-indigo-900 rounded-full font-bold flex items-center justify-center hover:bg-slate-100 transition-colors shadow-lg">
                Contact for Implementation
              </a>
            </div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="bg-slate-50 border-t border-slate-200 py-12 px-6">
        <div className="max-w-7xl mx-auto flex flex-col md:flex-row text-center justify-between items-center gap-4 text-slate-500 text-sm">
          <div className="flex items-center gap-2 justify-center">
             <Activity size={16} className="text-indigo-600" />
             <span className="font-semibold text-slate-700">OmniPulse</span>
          </div>
          <p>Released under the <a href="https://github.com/samvardhan03/OmniPulse/blob/main/LICENSE" className="text-indigo-600 hover:underline">Apache 2.0 License</a> | Designed and Developed by <strong>Samvardhan Singh</strong>.</p>
        </div>
      </footer>
    </div>
  );
}

export default App;
