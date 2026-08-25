# Factibilidad: ¿qué dLLM podemos entrenar en KUHPC?

**Fecha:** 25 ago 2026 · **Basado en:** inventario real de la partición `sixhour` (scontrol, 25 ago 2026), throughput medido del entrenamiento Qwen3.5 LoRA, y leyes de escalado de dLLMs de alrobles/dLLM (literature_map, main_v3).

---

## 1. Hardware real disponible (partición sixhour)

Inventario medido con `scontrol show nodes` el 25-ago-2026:

| Tarjeta | Nodos | GPUs | BF16 TF | Nota |
|---|---|---|---|---|
| **MI210** | 27 | 81 | 181 | **columna vertebral del proyecto** |
| A100 | 8 | 20 | 312 | solo 2 nodos completos usables |
| Q6000 | 9 | 29 | 149 | parte mesh, no dedicada |
| V100 | 17 | 36 | 112 | arq. antigua |
| PRO6000 (Blackwell) | 3 | 5 | 238 | reservada para teacher/serve |
| A40 / L40 / Q8000 | — | 10 | ~150 | minoritaria |

**Restricción clave (verificada):** el scheduler **topea a 2 MI210 por nodo** (`gpu:mi210:2` agenda; `mi210:3` queda PENDING). Los 27 nodos MI210 sirven **54 GPUs** efectivas.

## 2. Restricción por walltime (el verdadero cuello de botella)

- Partición `sixhour`: **5:50 de walltime por job**.
- El entrenamiento **desde cero no cabe** en 6h. La única forma de usar toda la semana es:
  - **Checkpoint + resume cada ~6h** (patrón ya probado: el swarm `q35_wave.slurm` re-encadena vía SIGUSR1, y `swarm_watchdog.sh` lo mantiene vivo).
  - Solapando el arranque del siguiente job con la bajada del anterior para **~no perder tiempo**.
- Con encadenado continuo + watchdog, el efecto del walltime se reduce a ~5-10% overhead (reinicio de proceso, recarga de checkpoint), no a un factor multiplicativo.

## 3. Cómputo disponible a la semana (modelo FLOPs)

Costo de entrenar un modelo denso de `P` parámetros sobre `T` tokens: **FLOPs = 6·P·T**.

| Escenario | FLOPs/semana |
|---|---|
| 54 MI210 × MFU 40% | 2.4e21 |
| 54 MI210 × MFU 60% | 3.5e21 |
| 81 MI210 (nominal) × MFU 60% | 5.3e21 |

## 4. Modelo denso máximamente entrenable a la semana

Con 54 MI210 (MFU 40%) y la ratio data de cada familia:

| Ratio datos (tok/param) | Params máx | Tokens máx |
|---|---|---|
| AR-like (20 tok/param) | 4.4B | 89B |
| **diffusion conservador (~40 tok/param, motivo Quokka)** | **3.1B** | **126B** |
| diffusion agresivo (100 tok/param) | ~2.0B | ~200B |

> La difusión es **data-hungrier** que AR (Quokka: 2-5× más datos en una sola epoch). Por eso los dLLM de frontera entrenan con ratios ~40-100 tok/param.

## 5. ¿Qué modelos reales cambiarían? (referencias de escala)

| Modelo objetivo | Params | Tokens | Coste | ¿1 semana? |
|---|---|---|---|---|
| Calibración (Fase 0 protocol) | 10-50M | 1B | ~1e17 | ✅ horas |
| Baseline (Fase 1) | 110M | 1-3B | ~6.6e17 | ✅ <1 día |
| Confirmación (Fase 2) | 350M-1.3B | 10-40B | ~1.3e19 | ✅ 1-3 días |
| **dLLM medio propio (objetivo)** | **1-3B** | **60-150B** | **~1-5e21** | **✅ SI, en la ventana** |
| DiffuCoder-7B (130B tok) | 7B | 130B | 5.5e21 | ⚠️ ~2.3 semanas |
| LLaDA-8B (2.3T tok) | 8B | 2300B | 1.1e23 | ❌ ~47 semanas |

## 6. Conclusión / Veredicto

1. **LLaDA-8B desde cero: NO.** 46 semanas continuas superan cualquier plazo útil.
2. **DiffuCoder-7B (130B tok):** borderline — ~2.3 semanas, es decir >1 semana de margen pedido; solo viable con todas las MI210 y MFU alto.
3. **LO REALISTA (recomendado):** entrenar un **dLLM denso de ~1-3B parámetros sobre ~60-150B tokens** en una semana con las 54 MI210 + encadenamiento continuo. Esto **encaja exactamente** con el `experimental_protocol.md` ya integrado en el repo (Fases 0→1→2: calibración → 110M → 350M-1.3B).

**Por qué merece la pena:** la ratio de datos de difusión nos permite, con el mismo presupuesto FLOPs de una semana, un modelo ~3B/126B que es comparable en tamaño al régimen donde MDLM/LLaDA descubrieron ventajas de calidad y velocidad paralela — y el protocolo ya lo diseña para que sea **reproducible y con gates go/no-go**.

**Riesgo real que el análisis expone:** con el walltime de 6h el proyecto solo es factible si la cadena de checkpoint/resume funciona a la perfección (qué ya hemos probado con el swarm). Sin eso, una semana continua es imposible por diseño.