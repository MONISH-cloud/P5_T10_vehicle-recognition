# Ablation Study and Failure-Case Analysis

Ablation measured on a fixed random subset of 2000 of the 8041 unseen test images. Failure analysis uses all of them.

## Ablation study

| Pipeline | Top-1 | Top-5 | Change from deep-only |
|---|---|---|---|
| Full image, no detector | 92.80% | 99.30% | -1.20 pts |
| Ground-truth box (upper bound, not deployable) | 94.10% | 99.65% | +0.10 pts |
| Detector crop (deep-only baseline) | 94.00% | 99.55% | +0.00 pts |
| Detector + adaptive enhancement | 94.15% | 99.60% | +0.15 pts |
| Detector + GrabCut refinement | 93.80% | 99.50% | -0.20 pts |
| Detector + flip TTA | 93.90% | 99.45% | -0.10 pts |
| Detector + classical fusion | 94.00% | 99.55% | +0.00 pts |
| Full hybrid | 93.75% | 99.55% | -0.25 pts |
| Hybrid without enhancement | 93.80% | 99.45% | -0.20 pts |
| Hybrid without refinement | 94.15% | 99.50% | +0.15 pts |
| Hybrid without TTA | 93.95% | 99.55% | -0.05 pts |
| Hybrid without classical fusion | 94.00% | 99.50% | +0.00 pts |

Full hybrid against deep-only: -0.25 points. A component with no positive effect is reported as such.

## Error profile

444 of 8041 images were misclassified.

| Tag | Errors |
|---|---|
| same_make_wrong_model | 284 |
| wrong_make | 125 |
| difficult_exposure | 77 |
| year_only_confusion | 35 |
| small_vehicle | 8 |
| wrong_or_missed_box | 2 |
| blurred | 1 |

Tags can overlap. Most frequent confusions:

- Audi TTS Coupe 2012 -> Audi TT Hatchback 2011: 18
- Audi S5 Coupe 2012 -> Audi A5 Coupe 2012: 11
- Dodge Caliber Wagon 2012 -> Dodge Caliber Wagon 2007: 11
- Chevrolet Silverado 1500 Hybrid Crew Cab 2012 -> Chevrolet Silverado 1500 Extended Cab 2012: 11
- Audi V8 Sedan 1994 -> Audi 100 Sedan 1994: 11
- BMW 6 Series Convertible 2007 -> BMW M6 Convertible 2010: 9
- Bentley Continental GT Coupe 2012 -> Bentley Continental GT Coupe 2007: 9
- Audi 100 Sedan 1994 -> Audi V8 Sedan 1994: 8
- Chevrolet Express Cargo Van 2007 -> Chevrolet Express Van 2007: 8
- Chevrolet Express Van 2007 -> GMC Savana Van 2012: 8
- GMC Savana Van 2012 -> Chevrolet Express Van 2007: 8
- Chevrolet Express Van 2007 -> Chevrolet Express Cargo Van 2007: 7
- Audi 100 Sedan 1994 -> Audi 100 Wagon 1994: 7
- Chevrolet Tahoe Hybrid SUV 2012 -> Chevrolet Avalanche Crew Cab 2012: 7
- Mercedes-Benz Sprinter Van 2012 -> Dodge Sprinter Cargo Van 2009: 7

## Failure cases

Example images are in `results/failures/<tag>/`.

### Failure case 1: unusual viewpoint

- **Image:** `results/failures/wrong_or_missed_box/00_00353.jpg` (test image `00353.jpg`)
- **True class:** Aston Martin V8 Vantage Convertible 2012
- **Predicted:** Bugatti Veyron 16.4 Convertible 2009, confidence 0.20
- **Cause:** the photo is taken from directly above, a viewpoint that is rare in the training data. The car fills a small part of a large frame with a motion-blurred background, and the detector box overlapped the annotation by only 0.18 IoU. The low confidence correctly signalled uncertainty, but the make-level answer was still wrong.
- **Possible fix:** add overhead-view augmentation, and classify several candidate boxes rather than only the largest.

### Failure case 2: probable label error between twin vans

- **Image:** `results/failures/wrong_make/03_07470.jpg` (test image `07470.jpg`)
- **True class (dataset label):** GMC Savana Van 2012
- **Predicted:** Chevrolet Express Van 2007, confidence 0.98
- **Cause:** the van in the photo clearly carries a Chevrolet badge on its grille, so the dataset label is probably wrong and the prediction may be right. The Chevrolet Express and GMC Savana are badge-engineered versions of one vehicle, confused 16 times in both directions on the test set.
- **Possible fix:** check and clean noisy labels; treat badge-engineered twins as one group, or add a model focused on the badge region.

### Failure case 3: wrong model year

- **Image:** `results/failures/year_only_confusion/02_02012.jpg` (test image `02012.jpg`)
- **True class:** Bentley Continental GT Coupe 2012
- **Predicted:** Bentley Continental GT Coupe 2007, confidence 1.00
- **Cause:** only the rear is visible, and the model years differ in small styling details such as the lights and bumper. The model was fully confident, so no confidence threshold could have caught this error.
- **Possible fix:** give the model an explicit year signal, for example a separate year output, and add more rear-view training examples.
