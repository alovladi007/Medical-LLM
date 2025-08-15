"use client";
export default function Home(){
  return (
    <main className="p-6 max-w-3xl mx-auto">
      <h1 className="text-2xl font-bold">Med‑AGI Home</h1>
      <div className="mt-2 space-x-4">
        <a className="underline" href="/dicom">Open DICOM</a>
        <a className="underline" href="/ekg">Open EKG</a>
        <a className="underline" href="/citations">Open Citations</a>
        <a className="underline" href="/adjudicate">Open Adjudication</a>
      </div>
    </main>
  );
}
