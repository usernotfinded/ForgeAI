# ForgeAI Stretch v1 (`forge stretch`)

## 1) Cos'è e cosa fa davvero

`forge stretch` estende in modo persistente la context window di un modello già esistente e compatibile.

In v1:
- usa **un solo metodo tecnico**: `YaRN scaling`
- produce una variante persistente in formato **`adapter_plus_manifest`**
- non promette "memoria magica" o "modello più intelligente"

`forge stretch` non trasforma qualunque modello in 128k con un click: valuta compatibilità, propone target realistici, chiede consenso stratificato e valida i trade-off.

## 2) Cosa NON fa (limiti espliciti v1)

- non supporta metodi diversi da YaRN
- non supporta tutte le architetture
- non produce (ancora) `full_checkpoint`
- non garantisce qualità migliore in generale
- non sostituisce benchmark semantici applicativi reali

La validazione v1 è locale/proxy: utile per regressioni tecniche e integrità del flusso, non equivalente a una valutazione completa di prodotto.

## 3) Compatibilità v1

Compatibile quando:
- checkpoint ForgeAI con `metadata.json` + `model.pt`
- architettura Transformer RoPE-based supportata dal registry locale
- backend supportato per YaRN v1
- stato modello compatibile (es. no adapter separati non unificati)

Non compatibile:
- architetture senza ricetta locale stretch
- modello non RoPE-based
- target non realistico per hardware
- target `<=` contesto nativo

## 4) Regola target context (obbligatoria)

Il target deve essere **strettamente maggiore** del contesto nativo.

Esempio:
- nativo `32768`
- validi: `65536`, `131072`, `262144` (se realistici)
- non validi: `32768`, `16384`

## 5) Persistenza reale v1: `adapter_plus_manifest`

La variante finale include:
- modello base tracciato
- metadati aggiornati
- adapter reale prodotto dal processo stretch
- manifest deterministico con hash e passi di ricostruzione

Questo significa che la variante non è solo "config aggiornata": c'è un artefatto reale (`stretch_adapter.bin`) più manifest sufficiente a ricostruzione non ambigua.

## 6) Struttura artefatti output

Cartella tipica:

```text
models/stretched/modelname-128k-yarn/
  model.pt
  metadata.json
  stretch_metadata.json
  stretch_manifest.json
  stretch_adapter.bin
  validation_report.json
```

Significato file:
- `model.pt`: checkpoint base tracciato nella variante
- `metadata.json`: config modello + sezione stretch aggiornata
- `stretch_metadata.json`: metadati stretch operativi (profilo, target, tipo artefatto, adapter)
- `stretch_manifest.json`: manifest deterministico (hash input/output, passi ricostruzione)
- `stretch_adapter.bin`: artefatto reale generato da YaRN v1
- `validation_report.json`: esito validazione (strutturale, persistenza, short, long, ricostruzione)

Artefatti sessione:
- `.forge/stretch/stretch_state.json`
- `.forge/stretch/artifacts/stretch_config.json`
- `.forge/stretch/artifacts/stretch_report.md`

## 7) Ricostruzione variante da manifest

Entry point Python già disponibile:
- `app.stretch.reconstructor.reconstruct_variant_from_manifest(...)`
- `app.stretch.reconstructor.run_minimal_reconstruction_demo(...)`

### Input richiesto
- percorso a `stretch_manifest.json`
- file referenziati nel manifest (`source_model_path`, `stretch_adapter.bin`) presenti

### Cosa verifica
- `final_artifact_type == adapter_plus_manifest`
- hash modello sorgente
- hash adapter
- blocco `deterministic_reconstruction`
- coerenza contesti e metodo (`yarn`)

### Esempio uso

```python
from app.stretch.reconstructor import (
    reconstruct_variant_from_manifest,
    run_minimal_reconstruction_demo,
)

variant = reconstruct_variant_from_manifest(
    "./models/stretched/qwen-128k-yarn/stretch_manifest.json"
)
demo = run_minimal_reconstruction_demo(variant)
print(demo["stretch_retrieved_all"], demo["baseline_retrieved_all"])
```

Se la ricostruzione è corretta:
- `stretch_retrieved_all` tende a `True`
- `baseline_retrieved_all` può restare `False` su scenari long-context oltre il nativo

## 8) Esempi CLI concreti

### Esempio A — stretch prudente

```bash
forge stretch \
  --model ./models/qwen2.5-0.5b \
  --target-context 65536 \
  --aggressiveness prudent
```

Outcome atteso:
- target realistico e rischio più basso
- variante `*-64k-yarn/`
- output: `stretch_adapter.bin`, manifest, report, validation report

### Esempio B — stretch ambizioso con override

```bash
forge stretch \
  --model ./models/qwen2.5-0.5b \
  --target-context 262144 \
  --aggressiveness ambitious
```

Se il profilo/ratio è molto aggressivo, viene richiesto:

```text
Override livello 3 richiesto. Scrivi 'PROCEDI COMUNQUE' per continuare
```

Rischio:
- maggiore probabilità di drift qualitativo
- possibile fallimento validazione

### Esempio C — errore target uguale/minore del nativo

```bash
forge stretch --model ./models/qwen2.5-0.5b --target-context 32768
```

Messaggio atteso:
- rifiuto esplicito perché target non è strettamente maggiore del nativo
- proposta target validi realistici disponibili

### Esempio D — ricostruzione variante

```python
from app.stretch.reconstructor import (
    reconstruct_variant_from_manifest,
    run_minimal_reconstruction_demo,
)

variant = reconstruct_variant_from_manifest(
    "./models/stretched/qwen2.5-0.5b-128k-yarn/stretch_manifest.json"
)
demo = run_minimal_reconstruction_demo(variant)
print(demo)
```

## 9) Lettura del report finale

Il report finale distingue chiaramente:
- controlli strutturali
- controlli persistenza
- controllo ricostruzione variante
- validazione short-context
- validazione long-context

Mostra anche:
- modello sorgente
- contesto nativo e target
- profilo aggressività
- metodo (`yarn`)
- tipo persistenza (`adapter_plus_manifest`)
- path variante finale
- esito processo e artefatti prodotti

## 10) Anti-overselling (esplicito)

`forge stretch` v1 garantisce:
- una variante long-context persistente con tracciabilità
- controlli tecnici concreti su integrità, persistenza e comportamento proxy

`forge stretch` v1 **non** garantisce:
- miglioramento generale del modello su tutti i task
- equivalenza a benchmark semantici completi
- compatibilità universale con ogni checkpoint
