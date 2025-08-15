import http from 'k6/http';
import { sleep, check } from 'k6';
export let options = { vus: 10, duration: '30s' };
export default function(){ let r=http.get(`${__ENV.IMAGING}/v1/dicom/studies?limit=1`); check(r,{ok:(res)=>res.status===200 || res.status===404}); sleep(1);}