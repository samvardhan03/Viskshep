import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import 'katex/dist/katex.min.css';
import { BlockMath, InlineMath } from 'react-katex';
import { ChevronRight, ChevronLeft, Waves, Sigma, Gauge, Filter } from 'lucide-react';

const steps = [
  {
    id: 'input',
    title: 'Raw Signal',
    subtitle: 'Time-domain input',
    icon: Waves,
    color: 'from-slate-600 to-slate-800',
    accentColor: 'text-slate-400',
    borderColor: 'border-slate-600',
    math: 'x(t) \\in \\mathbb{R}^T',
    description: 'A non-stationary 1-D time series — could be an EEG trace, gravitational wave strain, or HFT price series. Length T, sampled at rate fs.',
    visual: () => (
      <svg viewBox="0 0 400 120" className="w-full h-28">
        <defs>
          <linearGradient id="sig" x1="0" y1="0" x2="1" y2="0">
            <stop offset="0%" stopColor="#64748b" />
            <stop offset="100%" stopColor="#94a3b8" />
          </linearGradient>
        </defs>
        {/* Noisy signal with transient burst */}
        <path d={generateSignalPath()} fill="none" stroke="url(#sig)" strokeWidth="2" />
        {/* Burst region highlight */}
        <rect x="160" y="10" width="80" height="100" fill="#14b8a6" opacity="0.08" rx="4" />
        <text x="200" y="125" textAnchor="middle" fill="#5eead4" fontSize="9" fontWeight="600" fontFamily="monospace">transient burst</text>
      </svg>
    ),
  },
  {
    id: 'convolution',
    title: 'Morlet Convolution',
    subtitle: 'Wavelet filter bank',
    icon: Filter,
    color: 'from-teal-600 to-cyan-700',
    accentColor: 'text-teal-400',
    borderColor: 'border-teal-600',
    math: 'U_1 x(t, \\lambda_1) = x * \\psi_{\\lambda_1}(t)',
    description: 'Convolve x(t) with analytic Morlet wavelets ψ_λ at J·Q different center frequencies. Each wavelet has ℓ₁ norm ≤ 0.98, guaranteeing Lipschitz stability.',
    visual: () => (
      <svg viewBox="0 0 400 120" className="w-full h-28">
        <defs>
          <linearGradient id="morlet" x1="0" y1="0" x2="1" y2="0">
            <stop offset="0%" stopColor="#14b8a6" />
            <stop offset="100%" stopColor="#06b6d4" />
          </linearGradient>
        </defs>
        {[0, 1, 2, 3].map((i) => (
          <g key={i} transform={`translate(0, ${i * 28 + 5})`}>
            <path d={generateMorletPath(i)} fill="none" stroke="url(#morlet)" strokeWidth="1.5" opacity={1 - i * 0.15} />
            <text x="385" y="12" fill="#5eead4" fontSize="8" fontFamily="monospace" textAnchor="end">λ={i + 1}</text>
          </g>
        ))}
      </svg>
    ),
  },
  {
    id: 'modulus',
    title: 'Complex Modulus',
    subtitle: 'Nonlinear envelope',
    icon: Sigma,
    color: 'from-indigo-600 to-violet-700',
    accentColor: 'text-indigo-400',
    borderColor: 'border-indigo-600',
    math: 'S_1 x(t, \\lambda_1) = |x * \\psi_{\\lambda_1}|',
    description: 'Take the pointwise complex modulus |·|. This nonlinearity extracts the instantaneous amplitude envelope, creating translation-invariant energy representations at each scale.',
    visual: () => (
      <svg viewBox="0 0 400 120" className="w-full h-28">
        <defs>
          <linearGradient id="env" x1="0" y1="0" x2="1" y2="0">
            <stop offset="0%" stopColor="#818cf8" />
            <stop offset="100%" stopColor="#a78bfa" />
          </linearGradient>
        </defs>
        {[0, 1, 2].map((i) => (
          <g key={i} transform={`translate(0, ${i * 36 + 8})`}>
            <path d={generateEnvelopePath(i)} fill="none" stroke="url(#env)" strokeWidth="2" />
          </g>
        ))}
      </svg>
    ),
  },
  {
    id: 'averaging',
    title: 'Low-Pass Averaging',
    subtitle: 'Temporal invariance',
    icon: Gauge,
    color: 'from-amber-600 to-orange-600',
    accentColor: 'text-amber-400',
    borderColor: 'border-amber-600',
    math: 'S_1 x(t, \\lambda_1) = |x * \\psi_{\\lambda_1}| * \\phi(t)',
    description: 'Convolve with a low-pass filter φ(t) of scale 2^J to achieve translation invariance. The output S₁x captures energy envelopes at J·Q scattering paths.',
    visual: () => (
      <svg viewBox="0 0 400 120" className="w-full h-28">
        <defs>
          <linearGradient id="avg" x1="0" y1="0" x2="1" y2="0">
            <stop offset="0%" stopColor="#f59e0b" />
            <stop offset="100%" stopColor="#f97316" />
          </linearGradient>
        </defs>
        {[0, 1, 2].map((i) => (
          <g key={i} transform={`translate(0, ${i * 36 + 8})`}>
            <path d={generateSmoothPath(i)} fill="none" stroke="url(#avg)" strokeWidth="2.5" />
          </g>
        ))}
      </svg>
    ),
  },
];

function generateSignalPath() {
  let d = 'M 10 60';
  for (let i = 1; i < 390; i++) {
    const t = i / 390;
    const baseline = Math.sin(t * Math.PI * 4) * 15;
    const burst = (t > 0.4 && t < 0.6) ? Math.sin(t * Math.PI * 40) * 35 : 0;
    const noise = (Math.sin(i * 7.3) + Math.sin(i * 13.7)) * 5;
    d += ` L ${i + 10} ${60 - baseline - burst - noise}`;
  }
  return d;
}

function generateMorletPath(idx) {
  let d = 'M 10 14';
  const freq = 6 + idx * 4;
  const sigma = 30 + idx * 10;
  for (let i = 1; i < 380; i++) {
    const t = (i - 190);
    const gaussian = Math.exp(-(t * t) / (2 * sigma * sigma));
    const wave = Math.cos(t * freq * 0.03) * gaussian * 12;
    d += ` L ${i + 10} ${14 - wave}`;
  }
  return d;
}

function generateEnvelopePath(idx) {
  let d = 'M 10 18';
  for (let i = 1; i < 380; i++) {
    const t = i / 380;
    const env = Math.abs(Math.sin(t * Math.PI * (3 + idx * 2)) * (1 - Math.abs(t - 0.5) * 1.2)) * 15;
    const bump = (t > 0.35 && t < 0.65) ? Math.exp(-Math.pow((t - 0.5) * 8, 2)) * 10 : 0;
    d += ` L ${i + 10} ${18 - env - bump}`;
  }
  return d;
}

function generateSmoothPath(idx) {
  let d = 'M 10 18';
  for (let i = 1; i < 380; i++) {
    const t = i / 380;
    const smooth = Math.sin(t * Math.PI * (1.5 + idx * 0.8)) * 8 + Math.cos(t * Math.PI * 0.7) * 4;
    const peak = Math.exp(-Math.pow((t - 0.5) * 5, 2)) * 8;
    d += ` L ${i + 10} ${18 - smooth - peak}`;
  }
  return d;
}

export function ScatteringVisualizer() {
  const [activeStep, setActiveStep] = useState(0);
  const step = steps[activeStep];
  const StepIcon = step.icon;

  return (
    <div className="relative">
      {/* Step indicators */}
      <div className="flex items-center justify-center gap-2 mb-8">
        {steps.map((s, i) => {
          const Icon = s.icon;
          return (
            <button
              key={s.id}
              onClick={() => setActiveStep(i)}
              className={`flex items-center gap-2 px-4 py-2 rounded-full text-sm font-bold transition-all duration-300 ${
                i === activeStep
                  ? 'bg-slate-800 text-white shadow-lg scale-105'
                  : 'bg-white text-slate-500 border border-slate-200 hover:border-slate-400'
              }`}
            >
              <Icon size={16} />
              <span className="hidden sm:inline">{s.title}</span>
              <span className="sm:hidden">{i + 1}</span>
            </button>
          );
        })}
      </div>

      {/* Main visualizer card */}
      <AnimatePresence mode="wait">
        <motion.div
          key={step.id}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -20 }}
          transition={{ duration: 0.4, ease: 'easeOut' }}
          className="bg-slate-900 rounded-3xl p-8 md:p-10 shadow-2xl border border-slate-700/50"
        >
          <div className="grid md:grid-cols-2 gap-8 items-center">
            {/* Left — math & description */}
            <div>
              <div className="flex items-center gap-3 mb-4">
                <div className={`h-10 w-10 rounded-xl bg-gradient-to-br ${step.color} flex items-center justify-center shadow-lg`}>
                  <StepIcon size={20} className="text-white" />
                </div>
                <div>
                  <h3 className="text-white font-bold text-lg">{step.title}</h3>
                  <p className="text-slate-400 text-xs font-semibold uppercase tracking-wider">{step.subtitle}</p>
                </div>
              </div>

              <div className={`${step.accentColor} bg-slate-800/70 p-4 rounded-xl border ${step.borderColor}/20 mb-4`}>
                <BlockMath math={step.math} />
              </div>

              <p className="text-slate-300 leading-relaxed text-sm font-medium">
                {step.description}
              </p>
            </div>

            {/* Right — SVG visualization */}
            <div className="bg-slate-800/50 rounded-2xl p-4 border border-slate-700/30">
              <motion.div
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: 0.2 }}
              >
                {step.visual()}
              </motion.div>
            </div>
          </div>

          {/* Navigation */}
          <div className="flex justify-between items-center mt-6 pt-4 border-t border-slate-700/30">
            <button
              onClick={() => setActiveStep(Math.max(0, activeStep - 1))}
              disabled={activeStep === 0}
              className="flex items-center gap-1 text-slate-400 hover:text-white transition-colors disabled:opacity-30 disabled:cursor-not-allowed text-sm font-bold"
            >
              <ChevronLeft size={16} /> Previous
            </button>
            <div className="flex gap-1.5">
              {steps.map((_, i) => (
                <div key={i} className={`h-1.5 rounded-full transition-all duration-300 ${i === activeStep ? 'w-6 bg-teal-400' : 'w-1.5 bg-slate-600'}`} />
              ))}
            </div>
            <button
              onClick={() => setActiveStep(Math.min(steps.length - 1, activeStep + 1))}
              disabled={activeStep === steps.length - 1}
              className="flex items-center gap-1 text-slate-400 hover:text-white transition-colors disabled:opacity-30 disabled:cursor-not-allowed text-sm font-bold"
            >
              Next <ChevronRight size={16} />
            </button>
          </div>
        </motion.div>
      </AnimatePresence>
    </div>
  );
}
