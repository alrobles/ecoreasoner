# Factibilidad: ¿qué dLLM podemos entrenar en KUHPC con 2 MI210?

**Fecha:** 25 ago 2026 · **Basado en:** inventario real de la partición `sixhour` (scontrol), throughput medido del entrenamiento Qwen3.5 LoRA, y leyes de escalado de dLLMs de alrobles/dLLM.

---

## 0. Escenario asumido (aclaración del 25-ago)

**El entrenamiento se hace en 2 MI210** (un nodo) — el resto de tarjetas del cluster comparten/están ocupadas por otros usuarios. El horizonte temporal **no es una limitación**: se entrena durante el tiempo que haga falta con checkpoint/resume. Lo que importa es el **cómputo real de 2 MI210**.

## 1. Hardware real disponible (partición sixhour)

Inventario medido con `scontrol show nodes` el 25-ago-2026:

| Tarjeta | Nodos | GPUs | BF16 TF | Nota |
|---|---|---|---|---|
| **MI210** | 27 | 81 | 181 | **usaremos 2 (1 nodo)** |
| A100 | 8 | 20 | 312 | compartida / no dedicada |
| Q6000 | 9 | 29 | 149 | parte mesh |
| V100 | 17 | 36 | 112 | arq. antigua |
| PRO6000 (Blackwell) | 3 | 5 | 238 | teacher/serve |
| A40 / L40 / Q8000 | — | 10 | ~150 | minoritaria |

**Restricción clave:** el scheduler topea a **2 MI210 por nodo** (`gpu:mi210:2` agenda). Nuestro training usaría exactamente esas 2.

## 2. Restricción por walltime

- Partición `sixhour`: **5:50 de walltime por job**.
- Entrenamiento desde cero no cabe en 6h → se necesita **checkpoint + resume cada ~6h**.
- Ya probado: `q35_wave.slurm` re-encadena vía SIGUSR1 + `swarm_watchdog.sh` mantiene vivas las cadenas. Con doble 2 MI210, el overhead de encadenado (~5-10%) se vuelve **dominante** frente a un cluster completo: por eso el análisis de "1 semana" previo infravaloraba el problema real.

## 3. Cómputo de 2 MI210 (modelo FLOPs)

Costo de entrenar modelo denso de `P` params sobre `T` tokens: **FLOPs = 6·P·T**.

| Escenario | FLOPs/semana |
|---|---|
| 2 MI210 × MFU 40% | 8.76e19 |
| 2 MI210 × MFU 60% | 1.31e20 |

## 4. Modelo denso máximamente entrenable en 2 MI210

| Ratio datos (tok/param) | Params máx | Tokens máx | Tiempo |
|---|---|---|---|
| AR-like (20 tok/param) | 0.85B | 17B | ~1 semana |
| **diffusion conservador (~40x)** | **0.60B** | **24B** | **~1 semana** |
| diffusion agresivo (100x) | 0.38B | 38B | ~1 semana |

## 5. ¿Qué modelos reales cambiarían? (2 MI210)

| Modelo | Params | Tokens | Coste | ¿Tiempo en 2 MI210? |
|---|---|---|---|---|
| Calibración (F0) | 10-50M | 1B | 2e17 | ✅ horas |
| Baseline 110M (F1) | 110M | 3B | 2e18 | ✅ <2h |
| Confirmación (F2) | 350M-1.3B | 40B | 2.4e20 | ⚠️ ~8 días |
| Qwen2.5-1.5B (~90B tok) | 1.5B | 90B | 8.1e20 | ❌ ~9 semanas |
| DiffuCoder-7B | 7B | 130B | 5.5e21 | ❌ ~62 semanas |
| LLaDA-8B | 8B | 2300B | 1.1e23 | ❌ ~24 años |

## 6. Conclusión / Veredicto (2 MI210)

1. **LLaDA-8B / DiffuCoder-7B: IMPOSIBLE** en 2 MI210 (años).
2. **Incluso 1B/180B (~12 sem) o 1.5B/90B (~9 sem) quedan fuera** de un objetivo razonable.
3. **LO REALISTA con 2 MI210:** entrenar un **dLLM denso de ~350M-1.3B parámetros sobre ~20-40B tokens** — el rango que cubre la **Fase 1 y Fase 2 del `experimental_protocol.md`** (110M baseline → 350M-1.3B confirmación).
   - F1 baseline 110M: <2h ✅
   - F2 confirmación 1B / ~40B tokens: ~1 semana ⚠️ (marginal)
   - El punto dulce es **350M-700M / 15-30B tokens**: ~2-4 días.

**Recomendación:** apuntar a la **Fase 2 del protocolo (modelo ~700M-1B, ~20-30B tokens)** como techo realista en 2 MI210 (~3-5 días), no a modelos de 7-8B. Esto demuestra RQ1/RQ3 (entrenabilidad + ventaja de inferencia) con rigor, sin comprometer el cómputo.

**Riesgo clave:** con solo 2 MI210 el overhead del encadenado por walltime (6h) es más significativo; hay que minimizar el tiempo de recarga de checkpoint entre jobs para que la Fase 2 no se estire.