# Rules corpus — drop PDFs here

The authoritative rule sources for the rules retrieval arm + rules gold. Any
subset is fine to start (even just the MCC Laws covers most rules questions).

- `mcc_laws.pdf`            MCC Laws of Cricket (2017 Code / latest edition)
- `icc_test.pdf`           ICC Standard Test Match Playing Conditions
- `icc_odi.pdf`            ICC Standard ODI Playing Conditions
- `icc_t20i.pdf`           ICC Standard T20I Playing Conditions
- `ipl_playing_conditions.pdf`  IPL Playing Conditions

After dropping any of these, run:

    PYTHONPATH=backend .venv/bin/python -m cricket_guru.ingest.load_rules
    PYTHONPATH=backend .venv/bin/python -m cricket_guru.index.build_index --source rules --chunking fixed

Git-ignored like the rest of data/. License: MCC/ICC/BCCI own these — used here
for retrieval, cited by clause; not redistributed.
