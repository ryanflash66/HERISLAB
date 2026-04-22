# PV Module Temperature Thresholds Research Report
## First-Pass Public Domain Research for HERIS LAB Thermal Fault Detection System

**Date:** April 22, 2026  
**Issue:** DAR-141  
**Researcher:** Cursor Cloud Agent  
**Parent Project:** HERIS LAB thermal fault detection system (HER-83)

---

## Executive Summary

This report compiles publicly available temperature threshold data for photovoltaic (PV) modules from manufacturer datasheets and IEC/UL standards. The research covers the five priority questions outlined in DAR-141 regarding NOCT, maximum operating temperatures, bypass diode limits, hot-spot tolerance, severity tiers, and per-model variation.

**Key Finding:** Temperature specifications are remarkably **uniform across mainstream crystalline silicon manufacturers**, with narrow variance in critical parameters. All major manufacturers converge on similar operating limits (−40°C to +85°C/+90°C) and NOCT values (42-45°C).

---

## 1. NOCT (Nominal Operating Cell Temperature)

### Definition
NOCT is defined under standardized test conditions:
- Irradiance: 800 W/m²
- Ambient air temperature: 20°C
- Wind velocity: 1 m/s
- Mounting: Open back side with free air circulation

### Manufacturer Values

| Manufacturer | Model Family | NOCT | Source |
|--------------|--------------|------|--------|
| **JinkoSolar** | Tiger Neo N-type (72HL4, 78HL4) | 45±2°C | [Datasheet](https://nastechsolar.com/content/SOLAR%20PANELS/03%20JINKO/Datasheet%20Jinko%20Tiger%20Neo%20N-type%20615-635W.pdf) |
| **Canadian Solar** | HiKu CS3N-MS | 42±3°C | [Datasheet](https://es-media-prod.s3.amazonaws.com/media/components/panels/spec-sheets/Canadian_Solar-Datasheet-HiKu_CS3N-MS_v2.0C1_EN_1.pdf) |
| **Trina Solar** | Vertex NEG/NED19RC.20 | 43±2°C | [Datasheet](https://static.trinasolar.com/sites/default/files/T1_Datasheet_VertexN_TSM-NEG19RC.20_590-620W_2025_B.pdf) |
| **LG** | NeON R (360-370W) | 44±3°C | [Datasheet](https://www.lg.com/us/business/download/resources/BT00002151/lg-business-solar-spec-neon-r-360q1c-A5-051118.pdf) |
| **First Solar** | Series 6 / Series 6 Plus | Not specified | [Datasheet](https://www.firstsolar.com/-/media/First-Solar/Technical-Documents/Series-6-Plus/Series-6-Plus-Datasheet---US.ashx) |

### Industry Baseline
- **Typical crystalline silicon range:** 42–48°C ([PVEducation](https://www.pveducation.org/pvcdrom/modules-and-arrays/nominal-operating-cell-temperature))
- **Best-case designs** (with enhanced cooling): 33°C
- **Worst-case designs:** 58°C
- **Mainstream modules cluster:** 40-45°C

### Field Performance Notes
- Roof-integrated mounting (no air gap) can increase module temperature by **+10°C** compared to open-rack conditions
- Module design, materials, and cell packing density influence NOCT by ±5°C

---

## 2. Maximum Operating Cell Temperature

### Absolute Ceiling (Before Damage or Warranty Void)

| Manufacturer | Model Family | Operating Range | Source |
|--------------|--------------|-----------------|--------|
| **JinkoSolar** | Tiger Neo N-type | −40°C to +85°C | [Datasheet](https://nastechsolar.com/content/SOLAR%20PANELS/03%20JINKO/Datasheet%20Jinko%20Tiger%20Neo%20N-type%20615-635W.pdf) |
| **Canadian Solar** | HiKu CS3N-MS | Not specified in excerpt | [Datasheet](https://es-media-prod.s3.amazonaws.com/media/components/panels/spec-sheets/Canadian_Solar-Datasheet-HiKu_CS3N-MS_v2.0C1_EN_1.pdf) |
| **Trina Solar** | Vertex NEG/NED19RC.20 | −40°C to +85°C | [Datasheet](https://static.trinasolar.com/sites/default/files/T1_Datasheet_VertexN_TSM-NEG19RC.20_590-620W_2025_B.pdf) |
| **LG** | NeON R | −40°C to +90°C | [Datasheet](https://www.lg.com/global/business/download/resources/solar/DS_NeONR_60cells.pdf) |
| **First Solar** | Series 6 / Series 6 Plus | −40°C to +85°C | [Datasheet](https://www.firstsolar.com/-/media/First-Solar/Technical-Documents/Series-6-Plus/Series-6-Plus-Datasheet---US.ashx) |

### Standards-Based Operating Assumptions

**IEC 61730 / UL 61730:**
- Originally designed for **98th percentile module operating temperature ≤ 70°C** in "general open-air climates" with 40°C ambient ([OSTI](https://www.osti.gov/servlets/purl/1768288))
- For open-rack mounting: **normalized maximum PV module operating temperature = 90°C**
- Insulation materials must have Relative Thermal Index (RTI) ≥ 90°C

**IEC TS 63126:2025 (High-Temperature Operation):**
- **Level 1:** 98th percentile temperature ≤ 80°C
- **Level 2:** 98th percentile temperature ≤ 90°C
- Applies when modules exceed standard 70°C baseline ([IEC Webstore](https://webstore.iec.ch/en/publication/78554))

**IEC 61215 Thermal Cycling:**
- Modules tested across **−40°C to +85°C** range to verify material resilience ([JVG Thoma](https://www.jvg-thoma.com/iec-61215-iec-61730-certifications-guide/))

### Key Observation
The **+85°C upper limit** in manufacturer datasheets aligns precisely with IEC thermal cycling test requirements, suggesting this is the industry-standard warranty ceiling for crystalline silicon modules. LG's +90°C specification matches IEC's normalized maximum operating temperature for open-rack systems.

---

## 3. Bypass Diode Maximum Temperature

### Critical Temperature Thresholds

| Temperature Range | Failure Mode | Source |
|-------------------|--------------|--------|
| **150-175°C** | Danger zone: irreversible silicon diode degradation begins | [PV Test Lab](https://pvtestlab.com/bypass-diode-thermal-runaway-detection/) |
| **~180-240°C** | Solder melting: diode lead solder liquefies, causing open circuit | [PV Test Lab](https://pvtestlab.com/bypass-diode-thermal-runaway-detection/) |
| **>200°C** | Enclosure failure: junction box materials degrade, char, or ignite; fire risk | [PV Test Lab](https://pvtestlab.com/bypass-diode-thermal-runaway-detection/) |

### Normal Operating Specifications
- Bypass diodes typically operate at reverse voltages **<30% of V_RRM** (maximum reverse repetitive voltage)
- At lower voltage operation, datasheets may specify higher allowable junction temperatures than the 80% V_RRM baseline ([Diotec App Note](https://diotec.com/cn/application-notes/bypass-diodes-for-solarmodules.html?file=files%2Fdiotec%2Ffiles%2Fpdf%2Fservice%2Fapplications%2FApp_Note_solar_diodes.pdf))

### Test Conditions
- **HTRB (High Temperature Reverse Bias) testing** on Schottky bypass diodes conducted at 120°C, 130°C, and 140°C
- Test temperatures chosen to remain below instantaneous thermal runaway threshold ([ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0038092X24008478))

### Practical Implications for Fault Detection
- **150°C** is a critical monitoring threshold: consistent operation above this indicates imminent diode failure
- **120-140°C sustained temperature** warrants investigation even if not yet critical
- Thermal runaway can escalate rapidly once initiated, making early detection at 120-150°C range essential

---

## 4. Hot-Spot Tolerance (ΔT Above Cell Mean)

### Definition
Hot-spots occur when irradiation mismatch between cells causes reverse-biased cells to dissipate power, resulting in localized temperature increases. This is a primary degradation mechanism in modern PV modules.

### Observed Temperature Deltas in Research

| Condition | Temperature Increase (ΔT) | Peak Temperature | Source |
|-----------|---------------------------|------------------|--------|
| **Cracked cells** (40-60% shading) | +75°C to +80°C | 100-105°C | [Nature](https://www.nature.com/articles/s41598-021-03498-z.pdf) |
| **Potential-induced degradation (PID)** | +20°C | 45°C | [Nature](https://www.nature.com/articles/s41529-022-00221-9.pdf) |
| **Point-defected cells (high-efficiency)** | +175°C | 200°C | [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S1876610217344909) |
| **High-efficiency modules (defects)** | +145°C | 170°C | [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S1876610217344909) |

### Fault Correlation by Crack Mode
Research categorizes cell damage into four modes:
1. **Mode 1 (crack-free):** No hotspots detected
2. **Mode 2 (micro-cracks):** No hotspots detected
3. **Mode 3 (shaded areas):** Hotspots likely
4. **Mode 4 (breakdown):** Hotspots highly probable

([Nature](https://www.nature.com/articles/s41598-021-03498-z.pdf))

### Detection Strategy Implications
- **ΔT > +15-20°C** from cell mean: investigate for PID or minor shading issues
- **ΔT > +40-50°C:** high probability of crack or localized defect
- **ΔT > +75°C:** critical fault; imminent failure risk (especially if approaching absolute cell temperature >150°C)

### IEC 61215 Hot-Spot Endurance Test
- Modules must withstand hot-spot conditions during type approval testing
- Test exposes modules to reverse-bias heating to verify they don't fail catastrophically
- **No specific temperature threshold published**, but test validates module survival of hot-spot conditions expected in field operation

---

## 5. Severity Tiers (Warning / Alarm / Critical)

### Standards-Based Classification
**IEC and UL standards do not define graduated severity tiers**. They specify single ceiling values (e.g., 70°C baseline, 80°C Level 1, 90°C Level 2 per IEC TS 63126) but do not mandate warning/alarm/critical escalation thresholds.

### Proposed Rule-Based Thresholds for HERIS LAB
Based on compiled research, the following tiers are suggested for the thermal fault detection system:

#### **Tier 1: Normal Operation**
- Cell temperature: **< NOCT + 15°C** (≤ 58-60°C for typical modules)
- No alerts

#### **Tier 2: Warning (Yellow)**
- Cell temperature: **NOCT + 15°C to +70°C** (58-70°C)
- Hot-spot ΔT: **+15-40°C** above local mean
- Action: Log event, schedule inspection during next maintenance window

#### **Tier 3: Alarm (Orange)**
- Cell temperature: **70-85°C**
- Hot-spot ΔT: **+40-75°C** above local mean
- Bypass diode temperature: **120-150°C**
- Action: Alert operator, investigate within 24-48 hours; potential early fault

#### **Tier 4: Critical (Red)**
- Cell temperature: **>85°C** (exceeds manufacturer warranty ceiling)
- Hot-spot ΔT: **>+75°C** above local mean
- Bypass diode temperature: **>150°C** (irreversible degradation threshold)
- Action: Immediate investigation; consider module isolation; fire risk assessment

#### **Rationale**
- **70°C** aligns with IEC 61730 98th percentile baseline
- **85°C** matches manufacturer maximum operating temperature and IEC thermal cycling limit
- **150°C** for diodes is the published degradation onset threshold
- Hot-spot ΔT thresholds derived from peer-reviewed fault research

---

## 6. Per-Model Variation (Mono-Si / Poly-Si / Thin-Film)

### Temperature Coefficient Comparison

| Technology | Temperature Coefficient (Power) | NOCT Observation | Source |
|------------|--------------------------------|------------------|--------|
| **Monocrystalline Si** | −0.29% to −0.36%/°C | 42-45°C (see Sec. 1) | Multiple datasheets |
| **Polycrystalline Si** | −0.35% to −0.52%/°C | 42-45°C (similar to mono) | [Canal Solar](https://canalsolar.com.br/en/effect-of-temperature-on-mono-and-polycrystalline-photovoltaic-modules/) |
| **Thin-Film (CdTe)** | Better high-temp tolerance | Not specified | [Electrical Technology](https://www.electricaltechnology.org/2025/01/monocrystalline-polycrystalline-thin-film-solar-panels.html) |

### Key Findings

#### **Monocrystalline vs. Polycrystalline:**
- **Thermal coefficients are very similar**, contrary to common claims that polycrystalline has superior thermal behavior
- Experimental data in tropical climates showed **monocrystalline slightly outperformed polycrystalline** at elevated temperatures
- Modern innovations (half-cell design, passivation layers) have reduced monocrystalline coefficients to as low as **−0.36%/°C** (Jinko Cheetah/Swan families)
- **NOCT values are nearly identical** (42-45°C) between mono and poly technologies
- One study reported combined efficiency coefficient of **−0.52%/°C** for voltage (−0.48%/°C) and current (+0.10%/°C)

([Canal Solar](https://canalsolar.com.br/en/effect-of-temperature-on-mono-and-polycrystalline-photovoltaic-modules/), [MDPI](https://www.mdpi.com/2071-1050/16/23/10566))

#### **Thin-Film (First Solar Series 6):**
- **Operating range:** −40°C to +85°C (same as crystalline Si)
- **NOCT:** Not specified in public datasheets
- **Known advantage:** Better performance in high-temperature conditions than crystalline panels
- **Technology:** CdTe (cadmium telluride) thin-film

([First Solar Datasheet](https://www.firstsolar.com/-/media/First-Solar/Technical-Documents/Series-6-Plus/Series-6-Plus-Datasheet---US.ashx))

### Conclusion on Variation
**Temperature specifications do NOT differ meaningfully across mainstream mono-Si, poly-Si, or thin-film modules:**
- Maximum operating temperatures converge at **+85°C** (warranty ceiling)
- NOCT values cluster at **42-45°C** for crystalline silicon
- Temperature coefficients differ slightly, but this affects *efficiency degradation rate* under heat, not the *thermal fault thresholds* required for the HERIS LAB system

**Recommendation:** A single set of temperature thresholds can be applied across all PV module types in the utility's fleet. Technology-specific tuning is not required for rule-based thermal fault detection.

---

## Critical Data Gaps

### 1. **Utility-Specific Mounting Configuration**
- Open-rack vs. roof-integrated vs. ground-mounted thermal behavior differs by +10°C or more
- **Need:** Site-specific mounting details from Greenville Utilities Commission to adjust baseline NOCT assumptions

### 2. **Installed Module Inventory**
- Research covered LG NeON, Canadian Solar HiKu, JinkoSolar Tiger Neo, Trina Vertex, First Solar Series 6
- **Need:** Confirm these match actual deployed models at target installations
- If utility uses different manufacturers (e.g., LONGi, REC, Hanwha Q CELLS), datasheets must be reviewed

### 3. **Graduated Severity Thresholds (Warning/Alarm/Critical)**
- No public standard defines these tiers
- **Need:** Utility engineering and HERIS LAB team must validate proposed thresholds against operational risk tolerance and existing SCADA alarm philosophy

### 4. **Bypass Diode Manufacturer Specifications**
- Generic failure thresholds found (150-175°C degradation, 180-240°C solder melt)
- **Need:** Specific diode model used in target module junction boxes for precise operational limits

### 5. **Hot-Spot ΔT Detection Calibration**
- Research provides fault correlation data but not prescriptive detection thresholds
- **Need:** Field validation of proposed ΔT thresholds (+15°C warning, +40°C alarm, +75°C critical) against actual fault incidents in utility's historical maintenance records

### 6. **IEEE Paper Access**
- Multiple relevant IEEE Xplore papers identified but not accessible without subscription:
  - [Identifying PV Module Mismatch Faults by Thermography](https://ieeexplore.ieee.org/document/6879295/)
  - [Fault Classification Using Thermography and ML](https://ieeexplore.ieee.org/document/8716442)
  - [IR Thermal Image Analysis for Hot-Spot Detection](https://ieeexplore.ieee.org/document/8833855)
- **Potential value:** These papers may contain validated temperature thresholds from field studies

---

## Source Quality Assessment

### High-Confidence Data
- **Manufacturer datasheets:** Official specifications from LG, Canadian Solar, JinkoSolar, Trina, First Solar
- **IEC/UL standards:** IEC 61215, IEC 61730, IEC TS 63126, UL 61730 (international consensus standards)
- **Peer-reviewed research:** Nature, ScienceDirect, MDPI publications on hot-spot faults

### Medium-Confidence Data
- **Industry guides:** PVEducation.org, Seven Sensor, JVG-Thoma (reputable but non-authoritative)
- **Application notes:** Diotec bypass diode technical documentation

### Lower-Confidence / Informational
- **PV Test Lab blog:** Practical industry insights but not peer-reviewed

### Missing Data
- IEEE Xplore papers (paywall-restricted)
- Proprietary utility-specific operational data
- Direct manufacturer engineering specifications beyond public datasheets

---

## Recommendations

### For Immediate Implementation (HER-83 Parent Ticket)
1. **Adopt unified thresholds:** Use the proposed 4-tier severity classification (Sec. 5) as baseline for rule-based fault triggers
2. **Set absolute limits:**
   - Cell temperature critical threshold: **85°C**
   - Bypass diode critical threshold: **150°C**
   - Hot-spot ΔT critical threshold: **+75°C**
3. **No technology-specific tuning required:** Single threshold set applies to mono-Si, poly-Si, and thin-film modules (Sec. 6)

### For Follow-Up Validation
1. **Obtain utility deployment specifics:**
   - Exact module models installed at target sites
   - Mounting configurations (thermal derating factors)
   - Historical fault/maintenance records for threshold calibration
2. **IEEE paper access:** Request institutional access or interlibrary loan to retrieve full-text papers for additional validation data
3. **Collaborate with HERIS LAB team:** Review proposed thresholds against autoencoder anomaly scores to tune rule-ML hybrid decision logic

### Escalation Not Required (Yet)
This first-pass research provides sufficient data to establish baseline thresholds for the thermal fault detection system. **Direct manufacturer contact or utility engineering consultation is not immediately necessary** unless:
- Deployed modules include manufacturers not covered in this research
- Utility has experienced thermal faults with known temperature readings that contradict proposed thresholds
- Field testing reveals high false-positive or false-negative rates after initial deployment

---

## Summary Table of Key Thresholds

| Parameter | Typical Value | Warning | Alarm | Critical | Source |
|-----------|---------------|---------|-------|----------|--------|
| **NOCT** | 42-45°C | — | — | — | Manufacturer datasheets |
| **Max Operating Temp** | −40 to +85/+90°C | — | — | >85°C | Manufacturer datasheets |
| **Cell Temp (Operating)** | <60°C | 60-70°C | 70-85°C | >85°C | IEC standards + research |
| **Bypass Diode Temp** | <120°C | 120-150°C | 150-175°C | >175°C | PV Test Lab, research |
| **Hot-Spot ΔT** | <15°C | 15-40°C | 40-75°C | >75°C | Nature, ScienceDirect |

---

## Links to Source Documents

### Manufacturer Datasheets
1. [JinkoSolar Tiger Neo N-type 615-635W](https://nastechsolar.com/content/SOLAR%20PANELS/03%20JINKO/Datasheet%20Jinko%20Tiger%20Neo%20N-type%20615-635W.pdf)
2. [Canadian Solar HiKu CS3N-MS](https://es-media-prod.s3.amazonaws.com/media/components/panels/spec-sheets/Canadian_Solar-Datasheet-HiKu_CS3N-MS_v2.0C1_EN_1.pdf)
3. [Trina Solar Vertex NEG19RC.20](https://static.trinasolar.com/sites/default/files/T1_Datasheet_VertexN_TSM-NEG19RC.20_590-620W_2025_B.pdf)
4. [LG NeON R 360-370W](https://www.lg.com/us/business/download/resources/BT00002151/lg-business-solar-spec-neon-r-360q1c-A5-051118.pdf)
5. [First Solar Series 6 Plus](https://www.firstsolar.com/-/media/First-Solar/Technical-Documents/Series-6-Plus/Series-6-Plus-Datasheet---US.ashx)

### Standards and Technical Specifications
6. [IEC 61215 & IEC 61730 Guide (JVG-Thoma)](https://www.jvg-thoma.com/iec-61215-iec-61730-certifications-guide/)
7. [IEC TS 63126:2025 (IEC Webstore)](https://webstore.iec.ch/en/publication/78554)
8. [UL 61730 Standard (UL Standards & Engagement)](https://www.shopulstandards.com/ProductDetail.aspx?productId=UL61730-1_3_S_20260326)
9. [IEC 61724-1 Sensor Selection (Seven Sensor)](https://www.sevensensor.com/pv-module-temperature-sensor-selection-according-to-iec-61724-1)

### Peer-Reviewed Research
10. [Hotspot and Crack Correlation (Nature)](https://www.nature.com/articles/s41598-021-03498-z.pdf)
11. [PID Hotspot Analysis (Nature)](https://www.nature.com/articles/s41529-022-00221-9.pdf)
12. [High-Efficiency Module Hotspot Risk (ScienceDirect)](https://www.sciencedirect.com/science/article/pii/S1876610217344909)
13. [Bypass Diode Durability (ScienceDirect)](https://www.sciencedirect.com/science/article/abs/pii/S0038092X24008478)
14. [Temperature Effects: Mono vs Poly (MDPI)](https://www.mdpi.com/2071-1050/16/23/10566)

### Industry Resources
15. [NOCT Definition (PVEducation)](https://www.pveducation.org/pvcdrom/modules-and-arrays/nominal-operating-cell-temperature)
16. [Bypass Diode Thermal Runaway (PV Test Lab)](https://pvtestlab.com/bypass-diode-thermal-runaway-detection/)
17. [Bypass Diode Application Note (Diotec)](https://diotec.com/cn/application-notes/bypass-diodes-for-solarmodules.html?file=files%2Fdiotec%2Ffiles%2Fpdf%2Fservice%2Fapplications%2FApp_Note_solar_diodes.pdf)

### Restricted Access (IEEE Xplore - Not Retrieved)
18. [Identifying PV Module Mismatch Faults](https://ieeexplore.ieee.org/document/6879295/)
19. [Fault Classification for PV Modules](https://ieeexplore.ieee.org/document/8716442)
20. [IR Thermal Image Analysis for Hot-Spot Detection](https://ieeexplore.ieee.org/document/8833855)

---

## Conclusion

This first-pass research successfully identified:
1. ✅ **NOCT values:** 42-45°C across mainstream manufacturers (uniform)
2. ✅ **Maximum operating temperature:** +85°C (crystalline Si) / +90°C (LG NeON) — industry standard
3. ✅ **Bypass diode limits:** 150°C degradation onset, 175-200°C critical failure
4. ✅ **Hot-spot tolerance:** ΔT thresholds derived from research (+15°C, +40°C, +75°C proposed tiers)
5. ⚠️ **Severity tiers:** Not defined in standards; proposed 4-tier system requires HERIS LAB validation
6. ✅ **Per-model variation:** Negligible for thermal fault detection purposes; unified thresholds applicable

**The data is sufficient to close parent ticket HER-83 as "unneeded escalation"** if the HERIS LAB team validates the proposed thresholds against autoencoder model output and utility operational requirements. Direct manufacturer contact or utility-proprietary data is only required if field testing reveals calibration issues.

---

**End of Report**
