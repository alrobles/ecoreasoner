# Frontera MoE-dLLM de dominio ecológico — diseño de experimentos

> ReumanLab · EcoReasoner · 2026-08-27
> Exploración del punto fronterizo: dLLM (masked diffusion) + MoE top-1 + dominio ecológico.
> Las tres direcciones son implementables sobre `train_mdlm_moe.py` (MoEMLP con gate lineal
> top-1, n_experts=8, SIN compartido, SIN router-por-paso).

---

## 1. Dirección A — Router en el espacio de difusión

**Idea:** el router top-k actual decide con solo el token (`gate(x)`). Pero en masked-diffusion,
el modelo ve el *timestep* de denoising (qué fracción está enmascarada / cuántos pasos quedan).
Activar expertos SEGÚN el paso de denoising (y no solo según el token) es lo novedoso — no
está publicado en dominio.

**Implementación (factible):**
- Añadir un embedding del **timestep de ruido** (o el número de pasos de denoising) que se
  suma al gate: `gate(x + t_emb)` donde `t_emb = nn.Embedding(seq_len, n_experts)` (o un
  vector por paso).
- Los primeros pasos (mucho [MASK]) podrían activar expertos "globales/de reconstrucción";
  los últimos (refinamiento) expertos "finos".
- Coste: trivial (una linear más), no cambia el cómputo de activación.

**Experimento:** variar `gate(x + t_emb)` vs `gate(x)`, medir loss de denoising y diversidad de
activación de expertos por paso.

## 2. Dirección B — MoE semántico de dominio (expertos con identidad ecológica)

**Idea:** en vez de 8 expertos genéricos, dar identidad semántica a cada experto según el
método ecológico: SDM, filogenética, bioclim, ecología de comunidades, estadística/métodos,
tool-call/agente, etc.

**Implementación (factible con el corpus):**
- **Asignación supervisada:** etiquetar tokens/muestras del corpus por dominio (los papers
  minados tienen tags: SDM, phylo, bioclim, etc.) → una **pérdida auxiliar de carga por
  experto** que premia que el token del paper SDM active el experto SDM.
- **Router de dominio:** `gate` condicionado por un "dominio" a nivel de secuencia (el paper
  trae la etiqueta) + el token. En inferencia, el dominio se infiere del contexto.
- **Balance de carga basado en semántica:** en vez de loss aux uniforme, loss que alinee
  dominios ↔ expertos (loading-balance dirigido).

**Experimento clave:** comparar 8 expertos "ciegos" vs 8 "con identidad" (misma arquitectura,
misma FLOPs, distinta pérdida aux) — ver si alinear dominio→experto mejora loss y especialización.

## 3. Dirección C — Expertos compartidos para tool-calling

**Idea:** un **experto compartido** (siempre activo) que capture la sintaxis agéntica
(tool-calls, JSON, razonamiento de agente) + expertos especializados rotados. Ya existe en
LLaDA-MoE/Qwen — pero aquí el aporte es **anclar el compartido al formato agéntico** (traces
de tool-call) para que nunca se pierda la habilidad de agente.

**Implementación (directa):**
- En `MoEMLP`, añadir un experto `shared` (siempre activo, aparte del top-k) — como DeepSeek.
- El experto compartido se entrena con las trazas agénticas (B2) + los expertos especializados
  con el dominio científico.
- `forward`: `x = shared(x) + sum(exp_k(x))` (additivo) con gate para los rotados.

**Experimento:** medir si el shared preserva el tool-calling cuando el top-k se sobre-especializa.

---

## 4. Estrategia de experimentos (compatibles con 8×Q6000)

| # | Experimento | Cambio | Métrica | Tiempo est. |
|---|---|---|---|---|
| E1 | Baseline actual (top-1, 8 exp, sin compartido) | — | loss denoising | ya corre |
| E2 | A: gate + t_emb (router por paso de denoising) | +embedding timestep en gate | loss, activación por paso | ~1 ola (5:50) |
| E3 | B: expertos con identidad de dominio (aux loss por etiqueta) | +etiquetado por dominio + aux loss | loss, load per domain | ~1-2 olas |
| E4 | C: experto compartido + top-k | +shared expert | loss, tool-call format | ~1 ola |
| E5 | Combinados A+B+C | todos | loss total + métricas agénticas | 2-3 olas |

**Métricas adicionales (no solo loss):** activación por experto (diversidad), carga por dominio,
precisión de tool-call format (≈30 prompts canónicos), tokens/serving.

## 5. Datos para el etiquetado por dominio (viene de la minería)

- Los papers PMC minados tienen dominio/tags (por concepto de malla: SDM, phylo, bioclim,
  ecología de comunidades, agente/tool).
- Las trazas agénticas (B2) y las tesis/malla dan la clase "agente".
- El etiquetado por dominio → la pérdida aux de B.

## 6. Riesgo / honestidad
- E2 (router por paso) puede no dar mejoría si el modelo ya aprovecha el contexto del mask —
  es un experimento, resultado negativo es válido (protocolo RQ).
- B requiere etiquetar el corpus (trabajo de datos, pero lo tenemos de la minería).
- C es lo más estándar (DeepSeek/LLaDA ya lo hacen); nuestro matiz es anclar el shared a
  trazas agénticas ecológicas.

---

*Este documento es el punto de partida. Primer paso concreto: implementar E2 (router con
timestep) sobre `train_mdlm_moe.py` y medirlo en 1 ola.*