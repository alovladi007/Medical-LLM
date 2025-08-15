"use client";
import * as React from "react";

type Props = {
  modelId?: string | null;
  linkText?: string;
};

export default function ModelCardLink({ modelId, linkText }: Props) {
  if (!modelId) return null;
  const base = process.env.NEXT_PUBLIC_MODELCARDS_API || "http://127.0.0.1:8008";
  const href = `/models?model=${encodeURIComponent(modelId)}&src=${encodeURIComponent(base)}`;
  return (
    <a href={href} className="underline text-sm ml-2" title="Open model card">
      {linkText || "Open model card"}
    </a>
  );
}
