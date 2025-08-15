# Patch: Add /ekg link + model card link placeholder

## 1) Add a nav link to /ekg
In your `ui/app/page.tsx`, add a link somewhere visible (e.g., under existing nav):
```tsx
<div className="mt-2"><a className="underline" href="/ekg">Open EKG</a></div>
```

## 2) Show a model card link next to inference results
Import the component:
```tsx
import ModelCardLink from "./components/ModelCardLink";
```

When you render a result that includes a `model_id` (e.g., from CXR or EKG API),
place this next to your result header or scores:
```tsx
{result?.model_id ? <ModelCardLink modelId={result.model_id} /> : null}
```

If your result object uses a different field name, adapt accordingly.
The link will point to `/models?model=<id>&src=<MODELCARDS_API>`.

## 3) Environment variable for Model Cards API
Ensure your UI has:
```
NEXT_PUBLIC_MODELCARDS_API=http://127.0.0.1:8008
```
(or your deployed endpoint).

## 4) Optional home template
If you need a quick example home that has the /ekg link:
```tsx
// ui/app/page.tsx (minimal example)
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
```
