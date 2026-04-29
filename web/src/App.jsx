import React from 'react';
import 'katex/dist/katex.min.css';
import { InlineMath, BlockMath } from 'react-katex';
import { 
  Activity, Cpu, Github, Package, CheckCircle2, Mail,
  Radio, Brain, Dna, BarChart3, Shield, Zap, Lock, Database,
  Layers, ArrowRight, Terminal, Star, Sparkles, Server
} from 'lucide-react';
import { ScatteringVisualizer } from './ScatteringVisualizer';

/* ─────────────────────────────────── Cards ─────────────────────────────────── */

const DomainCard = ({ icon: Icon, title, subtitle, items, gradient, borderColor }) => (
  <div className={`bg-white border-2 ${borderColor} rounded-3xl p-8 transition-all duration-300 hover:-translate-y-1 hover:shadow-xl group`}>
    <div className={`h-14 w-14 rounded-2xl bg-gradient-to-br ${gradient} flex items-center justify-center mb-6 shadow-lg group-hover:scale-110 transition-transform`}>
      <Icon size={26} className="text-white" strokeWidth={2} />
    </div>
    <h3 className="text-xl font-bold mb-1 text-slate-900">{title}</h3>
    <p className="text-sm font-semibold text-slate-400 uppercase tracking-wider mb-4">{subtitle}</p>
    <ul className="space-y-2">
      {items.map((item, i) => (
        <li key={i} className="flex items-start gap-2 text-slate-600 text-sm font-medium leading-relaxed">
          <Sparkles size={14} className="mt-1 text-teal-500 shrink-0" />
          {item}
        </li>
      ))}
    </ul>
  </div>
);

/* ─────────────────────────────────── App ───────────────────────────────────── */

function App() {
  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 selection:bg-teal-100 selection:text-teal-900 font-sans">
      {/* ── Navigation ────────────────────────────────────────────────────── */}
      <nav className="fixed w-full z-50 bg-white/80 backdrop-blur-xl border-b border-slate-200 shadow-sm">
        <div className="max-w-7xl mx-auto px-6 h-20 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="h-10 w-10 bg-gradient-to-tr from-teal-600 to-indigo-600 rounded-xl flex items-center justify-center shadow-md shadow-teal-200">
              <Activity className="text-white" size={20} />
            </div>
            <div className="flex items-baseline gap-2">
              <span className="font-bold text-2xl tracking-tight text-slate-900">Vikshep</span>
              <span className="text-xs font-bold text-slate-400 hidden sm:inline">by OmniPulse</span>
            </div>
          </div>
          <div className="hidden md:flex items-center gap-8 text-sm font-bold text-slate-600">
            <a href="#visualizer" className="hover:text-teal-600 transition-colors">Visualizer</a>
            <a href="#domains" className="hover:text-teal-600 transition-colors">Domains</a>
            <a href="#mechanics" className="hover:text-teal-600 transition-colors">Mechanics</a>
            <a href="#pricing" className="hover:text-teal-600 transition-colors">Pricing</a>
          </div>
        </div>
      </nav>

      {/* ── Hero Section ──────────────────────────────────────────────────── */}
      <section className="pt-40 pb-24 px-6 relative overflow-hidden flex flex-col items-center bg-gradient-to-b from-white to-slate-50 border-b border-slate-200">
        <div className="max-w-4xl mx-auto text-center">
          <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full border border-teal-200 bg-teal-50 text-teal-700 text-sm font-bold mb-8 shadow-sm">
            <span className="relative flex h-2 w-2">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-teal-400 opacity-75"></span>
              <span className="relative inline-flex rounded-full h-2 w-2 bg-teal-500"></span>
            </span>
            v0.2.0 — Now <code className="bg-teal-100 px-1.5 py-0.5 rounded font-mono text-xs">pip install vikshep</code>
          </div>
          <h1 className="text-5xl md:text-7xl font-extrabold tracking-tight mb-6 text-slate-900">
            <span className="bg-clip-text text-transparent bg-gradient-to-r from-teal-600 to-indigo-600">Vikshep</span>
          </h1>
          <p className="text-lg md:text-xl text-slate-500 mb-2 font-bold tracking-wide">
            Vectorized Invariant Kernels for Scattering &amp; High-performance Extraction Pipelines
          </p>
          <p className="text-lg md:text-xl text-slate-700 mb-12 max-w-2xl mx-auto leading-relaxed font-medium">
            A Zero-Copy C++/CUDA Wavelet Scattering Transform engine with a mathematically rigorous Radix-2 Cooley-Tukey CPU fallback.
          </p>
          <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
            <div className="h-14 px-6 rounded-full bg-slate-900 text-white font-bold flex items-center gap-3 shadow-lg shadow-slate-200 select-all">
              <Terminal size={18} className="text-teal-400" />
              <code className="text-sm font-mono">pip install vikshep</code>
            </div>
            <a href="https://github.com/samvardhan03/OmniPulse" target="_blank" rel="noreferrer" className="h-14 px-8 rounded-full bg-white border-2 border-slate-200 text-slate-700 font-bold flex items-center gap-2 hover:bg-slate-50 hover:border-teal-200 shadow-sm transition-all w-full sm:w-auto justify-center">
              <Github size={18} /> Source on GitHub
            </a>
          </div>
        </div>
      </section>

      {/* ── Interactive Math Visualizer ────────────────────────────────────── */}
      <section id="visualizer" className="py-24 px-6 bg-gradient-to-r from-teal-50/50 to-indigo-50/50 border-b border-slate-200">
        <div className="max-w-5xl mx-auto">
          <div className="text-center mb-12">
            <h2 className="text-3xl md:text-4xl font-extrabold mb-4 text-slate-900">The Scattering Cascade</h2>
            <p className="text-slate-600 max-w-2xl mx-auto text-lg font-medium">
              Step through the mathematical machinery. Each stage is a real operation executed by the C++/CUDA engine.
            </p>
          </div>
          <ScatteringVisualizer />
        </div>
      </section>

      {/* ── Cross-Disciplinary Domains Grid ────────────────────────────────── */}
      <section id="domains" className="py-24 px-6 bg-white">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-3xl md:text-4xl font-extrabold mb-4 text-slate-900">Cross-Disciplinary Domains</h2>
            <p className="text-slate-600 max-w-2xl mx-auto text-lg font-medium">
              One domain-agnostic engine. Exact use cases across astrophysics, genomics, and quantitative finance.
            </p>
          </div>
          <div className="grid md:grid-cols-3 gap-8">
            <DomainCard
              icon={Radio}
              title="Astrophysics"
              subtitle="GW Chirps & FRBs"
              gradient="from-teal-600 to-cyan-600"
              borderColor="border-teal-100"
              items={[
                'Gravitational wave chirp extraction from LIGO strain data buried in laser glitches',
                'Fast Radio Burst (FRB) detection via dispersion measure isolation in non-Gaussian RFI',
                'Sub-threshold transient recovery in multi-messenger astronomy pipelines',
              ]}
            />
            <DomainCard
              icon={Dna}
              title="Genomics"
              subtitle="ChIP-seq Peaks"
              gradient="from-indigo-600 to-violet-600"
              borderColor="border-indigo-100"
              items={[
                'ChIP-seq peak calling via multi-scale scattering of read-depth coverage signals',
                'Transcription factor binding site detection in noisy sequencing pipelines',
                'Epigenomic signature extraction across histone modification tracks',
              ]}
            />
            <DomainCard
              icon={BarChart3}
              title="Quantitative Finance"
              subtitle="HFT Regime Classification"
              gradient="from-amber-500 to-orange-600"
              borderColor="border-amber-100"
              items={[
                'High-frequency trading regime classification via microstructure scattering features',
                'Volatility surface anomaly detection across options chains',
                'Order book imbalance transient detection in sub-millisecond tick data',
              ]}
            />
          </div>
        </div>
      </section>

      {/* ── Mechanics: Lipschitz Guarantee ──────────────────────────────────── */}
      <section id="mechanics" className="py-24 px-6 bg-gradient-to-r from-teal-50 to-indigo-50 border-y border-slate-200 relative overflow-hidden">
        <div className="max-w-7xl mx-auto grid lg:grid-cols-2 gap-12 items-center">
          <div>
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full border border-teal-200 bg-white text-teal-700 text-xs font-bold mb-6 tracking-wider uppercase">
              Mathematical Rigor
            </div>
            <h2 className="text-3xl md:text-4xl font-extrabold mb-6 text-slate-900">Lipschitz-Stable Representations</h2>
            <p className="text-lg text-slate-700 font-medium leading-relaxed mb-6">
              The scattering cascade is provably contractive. The Morlet filter bank is constructed with
              peak <InlineMath math="\|\psi\|_1 = 0.98" />, guaranteeing exponential decay of the Lipschitz constant with cascade depth.
            </p>
            <div className="bg-white rounded-2xl p-6 border border-slate-200 shadow-sm">
              <BlockMath math="\|Sx - Sy\|_2 \;\leq\; (0.98)^m \cdot \|x - y\|_2" />
              <p className="text-sm text-slate-500 font-medium mt-2 text-center">
                Verified over 200 adversarial trials per depth in the automated test suite.
              </p>
            </div>
          </div>

          <div className="bg-slate-800 rounded-3xl p-8 shadow-2xl shadow-indigo-900/20 font-mono text-sm relative overflow-hidden">
            <div className="flex gap-2 mb-6 opacity-30">
              <div className="w-3 h-3 rounded-full bg-red-400"></div>
              <div className="w-3 h-3 rounded-full bg-yellow-400"></div>
              <div className="w-3 h-3 rounded-full bg-green-400"></div>
            </div>
            <div className="space-y-6">
              <div>
                <span className="text-slate-400 text-xs uppercase tracking-widest font-bold">Zeroth Order (Averaging)</span>
                <div className="text-teal-400 mt-2 bg-slate-900/50 p-4 rounded-xl border border-teal-900/30">
                  <BlockMath math="S_0x = x * \phi" />
                </div>
              </div>
              <div>
                <span className="text-slate-400 text-xs uppercase tracking-widest font-bold">First Order (Scalogram)</span>
                <div className="text-teal-400 mt-2 bg-slate-900/50 p-4 rounded-xl border border-teal-900/30 overflow-x-auto">
                  <BlockMath math="S_1x(t,\lambda) = |x * \psi_\lambda| * \phi(t)" />
                </div>
              </div>
              <div>
                <span className="text-slate-400 text-xs uppercase tracking-widest font-bold">Second Order (Non-linear Modulation)</span>
                <div className="text-teal-400 mt-2 bg-slate-900/50 p-4 rounded-xl border border-teal-900/30 overflow-x-auto">
                  <BlockMath math="S_2x(t,\lambda_1,\lambda_2) = ||x * \psi_{\lambda_1}| * \psi_{\lambda_2}| * \phi(t)" />
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ── Dual-Tier Pricing ──────────────────────────────────────────────── */}
      <section id="pricing" className="py-24 px-6 bg-white border-t border-slate-200 relative overflow-hidden">
        <div className="max-w-7xl mx-auto relative z-10">
          <div className="text-center mb-16">
            <h2 className="text-3xl md:text-4xl font-extrabold mb-4 text-slate-900">Two Distinct Tiers</h2>
            <p className="text-slate-600 max-w-2xl mx-auto text-lg font-medium">
              <strong className="text-teal-600">Vikshep</strong> is the open-source C++/CUDA math engine.{' '}
              <strong className="text-indigo-600">OmniPulse</strong> is the enterprise differential licensing SaaS.
            </p>
          </div>

          <div className="grid md:grid-cols-2 gap-8 max-w-5xl mx-auto items-stretch">
            {/* Tier 1: Vikshep */}
            <div className="bg-gradient-to-b from-teal-50/50 to-white border-2 border-teal-200 rounded-3xl p-10 flex flex-col shadow-sm hover:shadow-xl hover:border-teal-400 transition-all">
              <div className="flex items-center gap-3 mb-2">
                <div className="h-10 w-10 bg-gradient-to-br from-teal-500 to-cyan-600 rounded-xl flex items-center justify-center">
                  <Package size={20} className="text-white" />
                </div>
                <h3 className="text-2xl font-bold text-slate-900">Vikshep</h3>
              </div>
              <p className="text-teal-600 font-bold text-sm uppercase tracking-wider mb-2">Open Source Research Tier</p>
              <p className="text-slate-600 mb-6 font-medium">Apache 2.0 licensed. For researchers, labs, and independent developers.</p>
              <div className="text-5xl font-extrabold mb-6 text-slate-900">$0<span className="text-xl text-slate-500 font-bold">/forever</span></div>

              <div className="bg-slate-800 rounded-xl px-4 py-3 mb-6 flex items-center gap-2 font-mono text-sm text-teal-400">
                <Terminal size={16} />
                <code>pip install vikshep</code>
              </div>

              <ul className="space-y-4 mb-8 flex-grow">
                <li className="flex items-center gap-3 text-slate-700 font-medium"><CheckCircle2 className="text-teal-600 shrink-0" size={22}/> C++/CUDA WST Engine (pybind11)</li>
                <li className="flex items-center gap-3 text-slate-700 font-medium"><CheckCircle2 className="text-teal-600 shrink-0" size={22}/> Radix-2 Cooley-Tukey CPU Fallback</li>
                <li className="flex items-center gap-3 text-slate-700 font-medium"><CheckCircle2 className="text-teal-600 shrink-0" size={22}/> Morlet Filter Bank (ℓ₁ ≤ 0.98)</li>
                <li className="flex items-center gap-3 text-slate-700 font-medium"><CheckCircle2 className="text-teal-600 shrink-0" size={22}/> MCP Server + Kymatio Backend</li>
                <li className="flex items-center gap-3 text-slate-700 font-medium"><CheckCircle2 className="text-teal-600 shrink-0" size={22}/> Full Test Suite (Lipschitz bounds)</li>
              </ul>

              <a href="https://pypi.org/project/vikshep/" target="_blank" rel="noreferrer" className="w-full h-14 bg-teal-600 text-white rounded-full font-bold flex items-center justify-center gap-2 hover:bg-teal-700 transition-colors shadow-md mt-auto">
                <Package size={18}/> View on PyPI
              </a>
            </div>

            {/* Tier 2: OmniPulse Enterprise */}
            <div className="bg-gradient-to-b from-indigo-50/50 to-white border-2 border-indigo-600 rounded-3xl p-10 flex flex-col relative shadow-xl shadow-indigo-100/50 hover:shadow-2xl transition-all">
              <div className="absolute top-0 right-0 bg-indigo-600 text-white text-xs font-bold px-4 py-1.5 rounded-bl-xl rounded-tr-3xl uppercase tracking-wider">
                Enterprise
              </div>
              <div className="flex items-center gap-3 mb-2">
                <div className="h-10 w-10 bg-gradient-to-br from-indigo-600 to-violet-700 rounded-xl flex items-center justify-center">
                  <Layers size={20} className="text-white" />
                </div>
                <h3 className="text-2xl font-bold text-slate-900">OmniPulse</h3>
              </div>
              <p className="text-indigo-600 font-bold text-sm uppercase tracking-wider mb-2">Enterprise SaaS Tier</p>
              <p className="text-slate-600 mb-6 font-medium">Differential licensing for exascale deployments and clinical pipelines.</p>
              <div className="text-5xl font-extrabold mb-8 text-indigo-600">Custom</div>

              <ul className="space-y-4 mb-8 flex-grow">
                <li className="flex items-center gap-3 text-slate-900 font-bold">
                  <Server className="text-indigo-600 shrink-0" size={22}/> Rust Orchestrator Engine
                </li>
                <li className="flex items-center gap-3 text-slate-900 font-bold">
                  <Database className="text-indigo-600 shrink-0" size={22}/> Arrow Plasma Zero-Copy Memory
                </li>
                <li className="flex items-center gap-3 text-slate-900 font-bold">
                  <Zap className="text-indigo-600 shrink-0" size={22}/> HNSW Vector Database Integration
                </li>
                <li className="flex items-center gap-3 text-slate-900 font-bold">
                  <Lock className="text-indigo-600 shrink-0" size={22}/> Cryptographic Ed25519 IPFS Licensing
                </li>
                <li className="flex items-center gap-3 text-slate-900 font-bold">
                  <Shield className="text-indigo-600 shrink-0" size={22}/> SLA & Dedicated Engineering Support
                </li>
              </ul>

              <a href="mailto:shekhawatsamvardhan@gmail.com" className="w-full h-14 bg-indigo-600 text-white rounded-full font-bold flex items-center justify-center gap-2 hover:bg-indigo-700 transition-colors shadow-md mt-auto">
                <Mail size={18}/> Contact for Implementation
              </a>
            </div>
          </div>
        </div>
      </section>

      {/* ── Footer ──────────────────────────────────────────────────────────── */}
      <footer className="bg-slate-900 border-t border-slate-800 py-16 px-6">
        <div className="max-w-7xl mx-auto">
          <div className="grid md:grid-cols-3 gap-12 mb-12">
            <div>
              <div className="flex items-center gap-3 mb-4">
                <div className="h-10 w-10 bg-gradient-to-tr from-teal-500 to-indigo-500 rounded-xl flex items-center justify-center">
                  <Activity className="text-white" size={20} />
                </div>
                <div>
                  <span className="font-bold text-xl text-white">Vikshep</span>
                  <p className="text-xs text-slate-400 font-bold">by OmniPulse</p>
                </div>
              </div>
              <p className="text-slate-400 text-sm leading-relaxed font-medium">
                High-Performance Wavelet Scattering Primitives.
                Zero-Copy C++/CUDA engine with mathematically rigorous CPU fallback.
              </p>
            </div>
            <div>
              <h4 className="text-white font-bold mb-4">Resources</h4>
              <ul className="space-y-2 text-sm">
                <li><a href="https://github.com/samvardhan03/OmniPulse" className="text-slate-400 hover:text-teal-400 transition-colors font-medium flex items-center gap-2"><Github size={14}/> GitHub Repository</a></li>
                <li><a href="https://pypi.org/project/vikshep/" className="text-slate-400 hover:text-teal-400 transition-colors font-medium flex items-center gap-2"><Package size={14}/> PyPI Package</a></li>
              </ul>
            </div>
            <div>
              <h4 className="text-white font-bold mb-4">Engine Stats</h4>
              <div className="grid grid-cols-2 gap-3">
                <div className="bg-slate-800 rounded-xl p-3 text-center border border-slate-700">
                  <div className="text-teal-400 font-extrabold text-xl">47</div>
                  <div className="text-slate-400 text-xs font-bold">Tests Passing</div>
                </div>
                <div className="bg-slate-800 rounded-xl p-3 text-center border border-slate-700">
                  <div className="text-teal-400 font-extrabold text-xl">0.98</div>
                  <div className="text-slate-400 text-xs font-bold">ℓ₁ Norm ψ</div>
                </div>
                <div className="bg-slate-800 rounded-xl p-3 text-center border border-slate-700">
                  <div className="text-indigo-400 font-extrabold text-xl">C++17</div>
                  <div className="text-slate-400 text-xs font-bold">Engine</div>
                </div>
                <div className="bg-slate-800 rounded-xl p-3 text-center border border-slate-700">
                  <div className="text-indigo-400 font-extrabold text-xl">0</div>
                  <div className="text-slate-400 text-xs font-bold">Mocks</div>
                </div>
              </div>
            </div>
          </div>
          <div className="border-t border-slate-800 pt-8 flex flex-col md:flex-row text-center justify-between items-center gap-4 text-slate-500 text-sm font-medium">
            <p>&copy; 2026 Samvardhan Singh. Released under the Apache 2.0 License.</p>
            <p className="text-slate-600 text-xs font-mono">vikshep v0.2.0 | _vikshep_core.cpython-311-darwin.so</p>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;
