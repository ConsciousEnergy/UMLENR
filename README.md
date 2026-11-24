<!--
  PROGRAM HUB README (not a renamed sim-app README).
  UMLENR is the encompassing program / LENR surface.
  LENR Sim App is one child product among several.
-->

<p align="center">
  <img src="https://github.com/ConsciousEnergy/UMLENR/assets/23019934/63ef47ed-e045-4ea0-b909-aaff1a1acfd6" alt="UMLENR program mark" width="120" />
</p>

<h1 align="center">UMLENR</h1>

<p align="center">
  <strong>Utilizing Machine Learning &amp; Modeling for LENR / LANR</strong><br />
  Public <strong>program hub</strong> for Conscious Energy — an encompassing open-science effort spanning agentic research, simulation, and laboratory DAQ.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/status-In%20Development-e87722?style=flat-square" alt="Status: In Development" />
  <img src="https://img.shields.io/badge/License-GPLv3-002454?style=flat-square" alt="License GPL v3" />
  <a href="https://github.com/users/ConsciousEnergy/projects/6"><img src="https://img.shields.io/badge/roadmap-Project%206-005490?style=flat-square" alt="Roadmap Project 6" /></a>
  <a href="https://conscious.energy"><img src="https://img.shields.io/badge/site-program%20home-185f97?style=flat-square" alt="Program site" /></a>
</p>

> **Epistemic honesty.** This program builds research tooling and models. It does **not** prove that LENR occurs in nature. Simulation numbers are model output; agent summaries are not lab measurements. Scientific open obligations (`EQ-OPEN-*`) stay **OPEN** in the simulation ledgers.

**At a glance:** **In Development** program hub · GPL-3.0 · four pillars below · child products keep their own repos

---

## What UMLENR is

**UMLENR is larger than any single application.** It is the public umbrella for the Conscious Energy LENR program: shared narrative, schemas, registries, release manifests, and pointers into child products.

| This hub owns | Child products own |
|---------------|-------------------|
| Program story and status | Their source trees and release cadences |
| Shared schema ids and trust-plane policy | Runtime deploy, science ledgers, DAQ hardware |
| Roadmap links (Project 6) | Day-to-day engineering in their repos |

Private or unfinished product trees are **not** merged here until each passes a public-disclosure gate. Status on this page: **In Development**.

---

## Program pillars

| Pillar | Role in the program | Status |
|--------|---------------------|--------|
| **LENR Simulation Application** | Comparative CMNS/LENR modeling workbench (claim-tiered engines, exports, theory companion) | Public research preview **v0.4.0-beta** — one product under this program, not the whole program |
| **LENR-ARA** | Agentic lab / research assistant lineage (next-gen product) | **Coming soon** (private until ready) |
| **LENR-ARA DAQ** | Laboratory data acquisition software and sanitized sample contracts | **Coming soon** — separate trust plane (CI ≠ deploy ≠ runtime ≠ DAQ) |
| **Operator tooling** | Day-2 ops for program hosts (SSH / containers / Swarm) | Near-term optional D-Force (`d4s`); **later** [Purple](https://github.com/erickochen/purple) TUI — **not** a cutover gate |

```text
                    UMLENR (this hub)
                   /       |        \
          LENR Sim App   LENR-ARA    DAQ software
         (public beta)  (Coming soon) (Coming soon)
                   \       |        /
              later: Purple operator TUI
```

---

## LENR Simulation Application (one pillar)

The simulation application is the most mature **child** today. It is **not** synonymous with UMLENR.

- **Version (confirmed markers):** `0.4.0-beta`
- **Public repo:** [ConsciousEnergy/lenr_sim_app](https://github.com/ConsciousEnergy/lenr_sim_app)
- **Live research preview:** [lenr-sim-beta.conscious.energy](https://lenr-sim-beta.conscious.energy)
- **Deep product README / demos / local quick start:** see that repository’s root README — do not duplicate a full sim onboarding wall here
- **License:** GNU GPL-3.0

Hub docs **link** sim contracts (e.g. `lenr-run-export-v1`) and honesty ledgers; they do not replace them.

---

## LENR-ARA and DAQ (Coming soon)

**LENR-ARA** continues the program’s agentic research thread (historical public lineage includes earlier AutoGPT / Crew-style experiments under archived UMLENR trees). The next-generation product stays private until release-ready — listed here as **Coming soon**, without private clone URLs.

**DAQ software** will capture lab instrumentation data on a **separate trust plane**. Public surfaces will only ever see sanitized schemas (planned hub id: `lenr-ara-daq-sample-v1`). DAQ credentials must never land on CI runners or the public sim database.

---

## Later goals (locked — not cutover gates)

These directions are intentional. They do **not** block sim Swarm/share cutover or this hub’s docs baseline.

1. **Purple TUI** — [erickochen/purple](https://github.com/erickochen/purple) as the later program-wide operator workstation (SSH fleet, containers over SSH, MCP). Optional; not required for cutover.
2. **Connect LENR-ARA ↔ Simulation Application** — product integration (e.g. Research Mode / shared export contracts) after the sim train is stable.
3. **Ingest LENR-ARA DAQ data** — sanitized samples via hub-registered schema only.

---

## History

Earlier UMLENR trees (Py Sims, `LENR_ARA_GPT`, `LENR_Math_Sim`, training datasets) are **historical research inventory**. Prefer soft-archive branches (e.g. `archive/legacy-main`) over silent deletion. Conference materials are kept as **PDF** when available ([`Unified_LENR_Theory.docx (1).pdf`](./Unified_LENR_Theory.docx%20(1).pdf)); placeholder PowerPoint stubs are not restored when a PDF exists.

---

## Roadmap

Program tracking: GitHub **[Project 6](https://github.com/users/ConsciousEnergy/projects/6)** under ConsciousEnergy.

Suggested themes: hub staging · sim release train · LENR-ARA · DAQ trust plane · Purple evaluation · public-disclosure checklist.

---

## Limitations

- Hub is **In Development** — registries and schemas grow over time.
- LENR-ARA and DAQ remain **Coming soon**.
- This README is a **program map**, not the simulation product manual.
- Scientific claim maturity lives in the sim ledgers — this hub does not close `EQ-OPEN-*`.

---

## Cite

Cite **child products** with their own versions. For the hub:

```bibtex
@software{umlenr_hub,
  author       = {Conscious Energy},
  title        = {UMLENR — program hub for LENR / LANR open science},
  year         = {2026},
  howpublished = {GitHub repository — \texttt{ConsciousEnergy/UMLENR}},
  note         = {In Development; umbrella for simulation, agentic research, and DAQ.}
}
```

Simulation software citations belong with [ConsciousEnergy/lenr_sim_app](https://github.com/ConsciousEnergy/lenr_sim_app) (see that README).

---

## Contribute / License

- **Program / hub:** docs and schema PRs against this repository once the docs-first hub lands.
- **Simulation application:** contributions on that product’s `dev` branch — see its `CONTRIBUTING.md`.
- **LENR-ARA / DAQ:** private until public; do not expect contribution paths here yet.
- **Feedback:** program maintainer contact on [conscious.energy](https://conscious.energy)

**License:** [GNU GPL-3.0](LICENSE) across the program suite (hub, simulation application, and future public LENR-ARA / DAQ surfaces). Third-party dependencies keep their own licenses.

**Site:** [https://conscious.energy](https://conscious.energy)
