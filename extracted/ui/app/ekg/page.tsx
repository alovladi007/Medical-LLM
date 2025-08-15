"use client";
import * as React from "react";

function Badge({triton}:{triton:any}){
  if(!triton) return <span className="ml-2 text-xs text-gray-500">checking…</span>;
  if(triton.available && triton.healthy) return <span className="ml-2 text-xs px-2 py-1 rounded bg-green-100 text-green-800">Triton: GPU ready</span>;
  if(triton.available && !triton.healthy) return <span className="ml-2 text-xs px-2 py-1 rounded bg-yellow-100 text-yellow-800">Triton: not ready</span>;
  return <span className="ml-2 text-xs px-2 py-1 rounded bg-gray-100 text-gray-700">Stub/CPU</span>;
}

export default function EKGPage(){
  const base = process.env.NEXT_PUBLIC_EKG_API || "http://127.0.0.1:8016";
  const [samples, setSamples] = React.useState<string>("");
  const [probs, setProbs] = React.useState<any|null>(null);
  const [unc, setUnc] = React.useState<number|null>(null);
  const [backend, setBackend] = React.useState<string>("");
  const [triton, setTriton] = React.useState<{available:boolean, healthy:boolean}|null>(null);

  React.useEffect(()=>{
    (async()=>{
      try{
        const r = await fetch(base + "/v1/ekg/health");
        const j = await r.json();
        setTriton(j);
      }catch{}
    })();
  },[base]);

  function genDemo(){
    const arr = Array.from({length:1000}, ()=> (Math.random()*2-1).toFixed(4));
    setSamples(arr.join(","));
  }

  async function infer(){
    try{
      const arr = samples.split(/[,\s]+/).filter(Boolean).map(Number);
      if(arr.length !== 1000) { alert("Provide exactly 1000 samples (comma or space separated)."); return; }
      const r = await fetch(base + "/v1/ekg/infer", {
        method: "POST",
        headers: {"Content-Type":"application/json"},
        body: JSON.stringify({ samples: arr })
      });
      const j = await r.json();
      setProbs(j.probs||null);
      setUnc(j.uncertainty ?? null);
      setBackend(j.backend || "");
    }catch(e:any){
      alert("Inference error: " + e?.message);
    }
  }

  const uncBadge = !unc && unc !== 0 ? null : (unc >= 0.50 ? <span className="px-2 py-1 rounded bg-red-100 text-red-800">High uncertainty</span> : unc >= 0.30 ? <span className="px-2 py-1 rounded bg-yellow-100 text-yellow-800">Moderate uncertainty</span> : <span className="px-2 py-1 rounded bg-green-100 text-green-800">Low uncertainty</span>);

  return (
    <main className="p-6 max-w-4xl mx-auto space-y-4">
      <h1 className="text-2xl font-bold">EKG Inference <Badge triton={triton}/></h1>
      <p className="text-sm text-gray-600">Paste exactly 1000 samples (CSV or space-separated), or generate a random demo.</p>
      <div className="flex gap-2">
        <button className="border rounded px-3 py-1" onClick={genDemo}>Generate demo waveform</button>
        <button className="border rounded px-3 py-1" onClick={infer}>Run inference</button>
        {uncBadge}
        {backend ? <span className="text-xs text-gray-500 ml-2">backend: {backend}</span> : null}
      </div>
      <textarea className="w-full h-40 border rounded p-2 font-mono text-xs" value={samples} onChange={e=>setSamples(e.target.value)} placeholder="v1, v2, ..., v1000" />
      <div className="border rounded p-3">
        <div className="font-medium mb-2">Probabilities</div>
        {!probs ? <div className="text-sm text-gray-500">No results yet.</div> :
          <table className="w-full text-sm">
            <thead><tr><th className="text-left">Label</th><th className="text-right">Prob</th></tr></thead>
            <tbody>
              {Object.entries(probs).map(([k,v]: any)=>(
                <tr key={k}><td>{k}</td><td className="text-right">{Number(v).toFixed(3)}</td></tr>
              ))}
            </tbody>
          </table>
        }
      </div>
      <div className="text-xs text-gray-500">Note: This is decision support only and not a substitute for clinical judgment.</div>
    </main>
  )
}
