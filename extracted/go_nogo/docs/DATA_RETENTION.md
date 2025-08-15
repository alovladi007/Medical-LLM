# Data Retention & Backup Notes

## Logs & Audit
- Ship to SIEM with 30–90 day retention (per policy).
- Local audit log rotation: daily, keep 14 days; export weekly to archive.

## PHI & Images
- Masked PNG previews are ephemeral (do not persist) unless explicitly exported.
- When override is used to export, store masked image in a restricted bucket with 30-day TTL.
- Never persist original unmasked pixels outside the DICOM store.

## Metrics & Eval
- Store eval metrics JSON indefinitely for traceability; back up nightly with repo snapshots.
