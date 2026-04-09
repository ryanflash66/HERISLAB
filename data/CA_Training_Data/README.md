# CA_Training_Data -- Convolutional Autoencoder (Track A)

Curated dataset for training and evaluating the Convolutional Autoencoder anomaly detection model.

## Approach

The autoencoder is trained **only on normal (no-fault) images**. At inference, high reconstruction error indicates an anomaly/fault.

## Structure

```
train/normal/          -- Normal images for autoencoder training
  electric_motor/      -- 168 PNG  (Electric Motor Thermal Fault Diagnosis, no_fault class)
  induction_motor/     -- 20 BMP   (Thermal Images of Induction Motor, Noload class)
  pv_om_inspection/    -- 7,836 TIFF (PV System O&M Inspection, double-row + single-row)
  pv_thermal_inspection/ -- 1,075 TIFF (PV System Thermal Inspection)
  solar_modules/       -- 2,302 JPG (Infrared Solar Modules, No-Anomaly class)

test/normal/           -- Held-out normal images for threshold calibration
  electric_motor/      -- 28 PNG
  induction_motor/     -- 5 BMP

test/fault/            -- Fault images for evaluating anomaly detection
  electric_motor/      -- 173 PNG  (Electric Motor Thermal Fault Diagnosis, fault class)
  induction_motor/     -- 344 BMP  (Thermal Images of Induction Motor, 10 fault conditions)
```

## Total Counts

| Split | Normal | Fault | Total |
|-------|--------|-------|-------|
| Train | 11,401 | 0     | 11,401 |
| Test  | 33     | 517   | 550    |

## Source Datasets

| Directory | Source Dataset | Domain |
|-----------|--------------|--------|
| electric_motor | Electric Motor Thermal image Fault Diagnosis DATASET | Electrical (primary) |
| induction_motor | Thermal Images of Induction Motor Dataset | Electrical (primary) |
| pv_om_inspection | Photovoltaic System O&M inspection | Solar PV (adjacent) |
| pv_thermal_inspection | Photovoltaic system thermal inspection | Solar PV (adjacent) |
| solar_modules | Infrared Solar Modules (No-Anomaly only) | Solar PV (adjacent) |

## Notes

- PV O&M files are prefixed `dr_` (double-row) and `sr_` (single-row) to avoid filename collisions
- Solar module images were filtered from module_metadata.json (anomaly_class == "No-Anomaly")
- Test/normal hold-out is ~14-20% of electrical equipment normal images
- Image formats are mixed (PNG, BMP, TIFF, JPG) -- preprocessing/normalization is required before training
