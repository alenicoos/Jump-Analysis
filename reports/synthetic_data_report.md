# Dati sintetici — Come e perché li abbiamo creati

Questo documento spiega in modo semplice tutto il dato sintetico usato nel progetto: da dove viene, come viene generato, e a cosa serve ciascun tipo.

---

## Il problema di fondo

Abbiamo due modelli temporali principali da trainare e pochissimi dati reali:

- **PitchTransformer** — predice il pitch dello stinco frame per frame
- **JumpAutoencoder** — rileva movimenti anomali durante il drop jump

I dati reali disponibili sono **20 trial personali** (un soggetto, con sensori IMU) e **183 atleti mocap**. Dai file mocap grezzi vengono estratte anche sequenze temporali, ma il numero di soggetti resta piccolo per trainare reti neurali robuste senza augmentazione.

La soluzione: generare dati sintetici che preservino le proprietà biomeccaniche reali ma aumentino la variabilità.

---

## Tipo 1 — Sequenze mocap temporali (183 atleti)

**Script:** `scripts/generate_mocap_sequences.py`  
**Output:** `data/generated/mocap_sequences.npz`

### Cosa fa

I file `.mat` del dataset mocap contengono le traiettorie 3D complete di ogni atleta durante il drop jump. La pipeline attuale usa queste registrazioni come **sequenze temporali complete**, perché PitchTransformer e JumpAutoencoder lavorano sul movimento frame per frame.

### Finestra estratta

Dalla discesa dalla pedana (~0.5s prima dell'atterraggio) fino al takeoff del secondo salto. Questa finestra corrisponde esattamente a quella registrata dalla webcam nella pipeline reale.

### Conversione a 2D

I marker 3D vengono proiettati sul piano frontale (Y = mediale-laterale, Z = verticale) per ottenere 17 keypoint 2D — lo stesso formato COCO prodotto da YOLO. I keypoint vengono normalizzati per l'altezza del corpo.

### Ricampionamento

Il mocap gira a ~100 Hz, la webcam a ~30 fps. Ogni sequenza viene ricampionata a 30 fps con interpolazione lineare, preservando la durata temporale reale del movimento.

### Target pitch

Il pitch della tibia (angolo nel piano sagittale rispetto alla verticale) viene calcolato direttamente dai marker 3D: `arctan2(knee_x - ankle_x, knee_z - ankle_z)`. Viene espresso come delta dal frame di initial contact (baseline = 0), esattamente come il segnale IMU.

---

## Tipo 2 — Trial sintetici dai 20 trial reali

**Script:** `scripts/generate_synthetic_trials.py`  
**Output:** `data/generated/synthetic_trials.npz`  
**Usato per:** training PitchTransformer

### Cosa fa

Prende i 20 trial reali personali (con segnale IMU ground truth) e genera **200 varianti augmentate** applicando trasformazioni casuali che preservano la biomeccanica.

### Augmentazioni applicate

| Augmentazione | Parametri | Razionale |
|---|---|---|
| **Time warp** | velocità ×0.80–1.20 | Atleti diversi eseguono il drop a velocità diverse |
| **Scale** | fattore ×0.85–1.15 | Distanza diversa dalla telecamera |
| **Shift orizzontale** | ±10% larghezza | Soggetto non perfettamente centrato |
| **Flip laterale** | 50% probabilità | Simmetria sinistra/destra + swap pitch |
| **Gaussian noise** | σ = 0.005–0.015 | Rumore di stima YOLO |

Il dataset finale usato per il training del Transformer contiene:
- 20 trial reali (usati solo per **validation**)
- 200 sintetici dai real + 183 sequenze mocap temporali (usati per **training**)

---

## Tipo 3 — Sequenze normali augmentate per l'Autoencoder

**Script:** `scripts/train_jump_autoencoder.py` (augmentazione on-the-fly)  
**Usato per:** training e validation JumpAutoencoder

### Il problema

L'autoencoder ha solo 183 sequenze normali. Con così pochi dati, un LSTM con ~200k parametri overfitterebbe — memorizzerebbe ogni atleta invece di imparare cosa significa un drop jump normale in generale.

### La soluzione

Durante il training, prima di ogni epoca, vengono generate **1000 varianti augmentate** dei 183 atleti normali. Di queste:
- 800 vanno in **training** (insieme ai 183 reali = 983 sequenze totali)
- 200 vanno in **validation** (mai viste durante il training)

Le stesse augmentazioni del tipo 2 vengono applicate sia ai keypoint che al pitch predetto, garantendo coerenza.

### Risultato

Gap train/val ridotto da 4× a 2× — il modello generalizza invece di memorizzare.

---

## Tipo 4 — Trial anomali sintetici

**Script:** `scripts/generate_anomalous_trials.py`  
**Output:** `data/generated/anomalous_trials.npz`  
**Usato per:** valutazione JumpAutoencoder

### Cosa fa

Per ogni atleta mocap genera **7 versioni anomale** del suo salto — movimenti che potrebbero capitare in un atleta con problemi di controllo motorio o deficit di forza. Totale: 183 × 7 = **1281 trial anomali**.

### Tipi di anomalia

| Tipo | Descrizione biomeccanica | Implementazione |
|---|---|---|
| **knee_valgus** | Ginocchia che collassano verso l'interno durante la flessione (rischio ACL) | I keypoint delle ginocchia si spostano medialmente del 4–10% dell'altezza corporea nella finestra ic→kfmax |
| **asymmetric_flexion** | Un lato si flette il doppio dell'altro (asimmetria di forza o dolore) | Il pitch di un lato viene amplificato ×1.4–1.8, l'altro ridotto ×0.3–0.6 |
| **shallow_landing** | Landing rigido, flessione troppo superficiale (alto impatto) | Pitch e discesa verticale delle anche/ginocchia ridotti al 20–45% del normale |
| **trunk_lateral_lean** | Il tronco si inclina lateralmente durante l'atterraggio | Spalle e testa traslate lateralmente del 4–9% dell'altezza corporea |
| **wide_stance** | Piedi troppo distanti durante il salto | Caviglie e ginocchia allargate del 6–14% per lato |
| **narrow_stance** | Piedi troppo vicini | Inverso di wide_stance |
| **asymmetric_pitch_only** | Solo il pitch è asimmetrico, i keypoint restano normali | Testa il contributo del segnale pitch all'autoencoder |

### Nota sulle severità

Le anomalie sono calibrate verso il **moderato-severo** intenzionalmente: servono per verificare che il sistema rilevi qualcosa di anomalo. Le anomalie reali (es. 3–4 cm di valgismo) sono più sottili e la soglia andrà ricalibrata quando saranno disponibili trial reali etichettati.

---

## Riepilogo

| Tipo | N campioni | Usato per | Script |
|---|---|---|---|
| Sequenze mocap temporali | 183 | Training PitchTransformer + Autoencoder | `generate_mocap_sequences.py` |
| Augmentazioni dai 20 real | 200 | Training PitchTransformer | `generate_synthetic_trials.py` |
| Augmentazioni normali on-the-fly | 1000 | Training/validation Autoencoder | `train_jump_autoencoder.py` |
| Trial anomali sintetici | 1281 | Valutazione Autoencoder | `generate_anomalous_trials.py` |
