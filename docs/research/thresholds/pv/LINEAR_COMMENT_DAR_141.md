# PV Temperature Threshold Research — Complete ✅

## Quick Summary

Completed first-pass research on PV module temperature thresholds for the HERIS LAB thermal fault detection system. **Key finding: specifications are remarkably uniform across manufacturers** — a single threshold set can be applied to all module types.

## Summary Table

| Parameter | Normal | Warning | Alarm | Critical |
|-----------|--------|---------|-------|----------|
| **Cell Temp** | <60°C | 60-70°C | 70-85°C | >85°C |
| **Bypass Diode** | <120°C | 120-150°C | 150-175°C | >175°C |
| **Hot-Spot ΔT** | <15°C | 15-40°C | 40-75°C | >75°C |

## Key Findings by Question

### 1. NOCT (Nominal Operating Cell Temperature)
**Typical values: 42-45°C** — remarkably consistent across all mainstream manufacturers:
- JinkoSolar Tiger Neo: 45±2°C
- Canadian Solar HiKu: 42±3°C
- Trina Vertex: 43±2°C
- LG NeON R: 44±3°C
- First Solar Series 6: Not specified (thin-film)

### 2. Maximum Operating Cell Temperature
**Industry standard: −40°C to +85°C** (some reach +90°C)
- This aligns with IEC 61215 thermal cycling test range
- Warranty void threshold for all manufacturers
- IEC 61730 assumes 90°C normalized maximum for open-rack mounting

### 3. Bypass Diode Max Temp
**Critical thresholds:**
- **150-175°C:** Irreversible degradation begins
- **180-240°C:** Solder melts, open circuit failure
- **>200°C:** Junction box materials char; fire risk

### 4. Hot-Spot Tolerance (ΔT Above Mean)
**Research-derived thresholds:**
- +15-20°C: Potential PID or minor shading
- +40-75°C: Likely crack or localized defect
- +75°C+: Critical fault, imminent failure (cracked cells reach +100-105°C peaks)

### 5. Severity Tiers
**Standards do NOT define graduated tiers.** Proposed 4-tier system above is derived from compiled research and requires HERIS LAB team validation.

### 6. Per-Model Variation
**Negligible for fault detection purposes:**
- Mono-Si, poly-Si, and thin-film all use +85°C max operating temp
- NOCT values cluster at 42-45°C (crystalline silicon)
- Temperature coefficients differ (−0.29% to −0.52%/°C) but this affects *efficiency*, not *fault thresholds*
- **Single threshold set can be applied across all module types**

## Critical Data Gaps

1. **Utility-specific mounting config** — open-rack vs. roof-integrated changes thermal behavior by +10°C
2. **Exact deployed module models** — need to confirm inventory matches researched manufacturers
3. **Graduated threshold validation** — proposed warning/alarm/critical tiers require utility engineering review
4. **IEEE paper access** — 3 relevant papers identified but behind paywall
5. **Field calibration** — thresholds need validation against historical fault data

## Recommendation

**Escalation to direct manufacturer contact or utility engineering is NOT immediately required.** This research provides sufficient baseline data to implement rule-based thermal fault triggers complementing the autoencoder model.

### Next Steps:
1. HERIS LAB team validates proposed thresholds against ML model output
2. Obtain utility deployment specifics (module inventory, mounting configs)
3. Field test and calibrate false-positive/negative rates
4. Only escalate if deployed modules fall outside researched manufacturers or field testing reveals issues

## Documentation

**Full research report:** See attached `pv_temperature_thresholds_research.md`
- 3,500+ word comprehensive report
- Detailed manufacturer comparison tables
- IEC/UL standards analysis
- 20+ source citations with links
- Critical data gaps and recommendations

**GitHub PR:** https://github.com/ryanflash66/bearing-app/pull/222

---

**Sources:** 20+ manufacturer datasheets, IEC/UL standards, peer-reviewed research from Nature, ScienceDirect, MDPI, and industry technical documentation.

**Conclusion:** Sufficient data to proceed with HER-83 parent ticket. Parent ticket can likely be closed as "unneeded escalation" if HERIS LAB validates these thresholds.
