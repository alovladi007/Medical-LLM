import http from 'k6/http';
import { sleep, check } from 'k6';
export let options = { vus: 10, duration: '30s' };
export default function(){ let wf=Array.from({length:1000},()=>Math.random()*2-1); let r=http.post(`${__ENV.EKG}/v1/ekg/infer`, JSON.stringify({samples:wf}), {headers:{'Content-Type':'application/json'}}); check(r,{ok:(res)=>res.status===200}); sleep(1);}