// ui/components/StatusWidget.tsx
"use client";
import * as React from "react";

export default function StatusWidget(){
  const [ok, setOk] = React.useState(true);
  React.useEffect(()=>{
    const run = async()=>{
      try{
        const imaging = process.env.NEXT_PUBLIC_IMAGING_API || "http://127.0.0.1:8006";
        const r = await fetch(imaging + "/v1/triton/health");
        const j = await r.json();
        setOk(j.healthy || false);
      }catch{ setOk(false); }
    };
    run(); const id = setInterval(run, 30000); return ()=>clearInterval(id);
  },[]);
  return ok ? <span className="text-xs px-2 py-1 rounded bg-green-100 text-green-800">OK</span>
            : <span className="text-xs px-2 py-1 rounded bg-red-100 text-red-800">Issues</span>;
}
