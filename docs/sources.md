# Corpus sources

The machine-readable list is [`data/sources.jsonl`](../data/sources.jsonl): one record per PDF with `doc_id`
(SHA-256 of the file), URL, language, version label, download date and redistribution note. This page does not
repeat checksums.

| Group | Files | Terms |
| --- | --- | --- |
| US OSHA publications (English) | `osha*.pdf` | US federal government works, public domain |
| UK HSE leaflets (English) | `hse-indg*.pdf` | Open Government Licence v3.0, attribution required |
| Vietnamese regulations (QCVN, Nghị định, Thông tư, Công báo) | `vn-*.pdf` | Legal normative documents; downloaded from mirror sites, not the issuing body |

Selection rules: only PDFs with a real text layer (no OCR); a page with fewer than 40 characters of text is
skipped and counted. `source_version` is a label taken from the document, not a statement that the document
is still in force. Several Vietnamese regulations here have since been amended or replaced.

PDFs are kept out of Git. The ingest report (`manifest.json` of each index generation) records pages total/used
per file.
