# ROADMAP — EcoReasoner Fase 3 (refundación `-new`): excavar un dLLM científico

> ReumanLab · EcoReasoner · 2026-09-14 (última actualización)
> Este es el PLAN MAESTRO. El código vive en `scripts/` + `harness/`; los diseños
> en `docs/designs/`. Para el estado MÁS reciente de jobs/watchdogs: ver
> `docs/results/` + skill `ecoreasoner-swarm` (referencias f0-micro-sweep-* y f1-hetero-*).

---

## 0. OBJETIVO (declarado por Angel — la refundación)

Generar **agentes con capacidad de razonamiento científico**. La tesis excavada:
¿puede un **dLLM desde cero**, entrenado para denoising de *estructura de
argumento científico* (no prosa cruda), aprender a **discriminar la continuación
inferencialmente correcta** de un razonamiento frente a alternativas plausibles
pero incorrectas?

**Falsación pre-registrada (no exige victoria, exige respuesta con evidencia):**
> si tras N tokens el modelo no discrimina ≥55% pairwise (transitorio) NI
> completa esqueletos coherentemente → la tesis queda falsada a nuestra escala.

> Un negativo limpio con ablaciones (random/span/span-alto × prosa/esqueleto)
> es un resultado publicable.



**Fuera de la mesa:** LLaDA-MoE-7B como pieza (solo techo de medición), SFT como
vía productiva, horizonte corto. Ver `docs/designs/ecoreasoner-Fase3-DESIGN-new.md` §0.




---

## 1. HITOS(medidos por decisión, no por calendario)

### F0 — Harness F0 + eval nueva de discriminación — ✅ COMPLETO (07-09)
- `harness/`: run_micro.py (slurm 1-GPU declarativo), report.py (report.json + index.jsonl),
   compare.py, suite_smoke.py (discriminación pairwise + completación + fluidez),
   validate_configs.py (previene YAML roto), build_pairs.py (pares reales del corpus).
- `scripts/moe_v4_micro.slurm` — micro-run 1 GPU pro6000 (10K steps, dense 50-100M,
   olas SIGUSR1, eval+report al completar).
- **Micro-sweep 6/6 completado y evaluado** → ver §2.



### F0b — Extractor de esqueletos + corpus — ✅ COMPLETO(07-09)
- `scripts/build_skeleton.py`(IMRaD + abstracts estructurados + frases-faro,0 GPU)
  → `data/skeleton/train_skeleton.jsonl`(**315,299 docs** con ≥3 etapas,675MB)
  → `data/train_ids_skeleton.npy`(129.5M tok, OOB_OK}.
- `scripts/skel_full.slurm` + `skel_pairs.slurm` + `pre_tokenize_skeleton.slurm`.
- Cobertura REAL: 51.6% ge2,21.6% ge3 (el 70% era del mini-corpus sintético;
   el real es heterogéneo y el activo de 315K docs basta).

.



### F1 — Entrenamiento objetivo con el pool heterogéneo — ✅ COMPLETO (08-09): NO-GO
- **Trainer heterogéneo VALIDADO**: `scripts/train_mdlm_moe_hetero.py`
  (`--autosize` por VRAM medido con fwd+bwd real + `SCALE_I` gradiente exacto por
   tokens + grad_accum global; smoke L40×4 world=4 micro=25 COMPLETE)。
- **Lanzador multi-familia**: `scripts/f1_hetero.slurm` (torchrun rendezvous TCP,
  cada job = 1 familia; NCCL_IB_DISABLE para el muro IB/noib).
- **RESULTADO (1000 steps,17 ranks L40+Q6000,03:02)**: acc se mantuvo en
  el azar exacto (~0.5;max 0.508;criterio GO ≥0.55 NO alcanzado.**NO-GO**。
  Detalle + curva + contraste: `docs/results/F1-HETERO-RESULTADO.md`.
- **Lección (config, no escala)**: micro-span-esqueleto(batch8+accum2=12 288
  tok/update ×10K updates=122.9M tok) → acc 0.5352;F1(65 536 tok/update
  ×1K updates=65.5M tok) → acc  ️0.5. Mismo corpus/modelo/mask/lr. La señal
  estructural se diluye con batch gigante y pocos updates → **la vía es más updates**
  **pequeños (receta micro,50K-100K steps), NO más GPU.**
- Estimación: `docs/designs/ecoreasoner-F1-HETERO-ESTIMACION.md` (histórico).



###F2 — Agente(loop + tools encima del modelo)—[ ]
### Paper 1 (systems: entrenamiento heterogéneo + harness)—en paralelo
### Paper 2 (tesis desde-cero, positiva o negativa limpia)—[ ]

---

## 2. RESULTADO del micro-sweep F0 (fuente de verdad: runs/index.jsonl + docs/results/MICRO-SWEEP-F0-RESULTADOS.md)

| Run | mask × datos | pairwise_acc | mean_delta |
|---|---|---|---|
| f0-random-prosa | random 15% × v7 | 0.5117 | -0.032 |
| f0-span-prosa | span64×15% × v7 | 0.4492 | -0.064 |
| f0-spanhi-prosa | span64×60% × v7 |  ️0.4766 | -0.052 |
| f0-random-esqueleto | random × skeleton | 0.5078 | +0.011 |
| **f0-span-esqueleto** | span64×15% × skeleton | **0.5352** | +0.015 |
| f0-spanhi-esqueleto | span64×60% × skeleton | 0.5312 | +0.030 |

**Veredicto:** GO pre-registrado(width ≥0.55) NO alcanzado por ninguno(mejor 0.535,
mejor esqueleto 0.535/0.531 vs prosa 0.45-0.51 con delta NEGATIVO). **Tesis
reforzada en dirección**: estructura de argumento mueve la discriminación
(esqueleto > prosa; span > random en esqueleto). 10K steps ×12,288 tok/update
= **122.9M tok ≈ **~0.95 epochs del corpus esqueleto** (129.5M;;la cifra
"245K tok" de la nota original era un error) — el micro NO estaba "subentrenado al"""
0.006%";vio ~1 epoch del esqueleto y aun así 0.535 (vs F1 0.5 con 0.5 epoch
a batch gigante).. **Ganador: span-esqueleto.**

---

## 3. CÓMPUTO — pool NVIDIA real y tok/s medidos

| Familia | GPUs | tok/s (micro dense, solo entreno,08-09) |
|---|---|---|---|
| **A100** (cu121) | ~18 | **~22,026** |
| pro6000 Blackwell | 5 | **~12,553** |
| L40 (cu121) | 4 | **~11,443** |
| Q6000 (cu121) | ~29 | **~7,941** |

- **PITFALL medición**: walltime del job incluye la carga del cache (~50s)→
  subestima 2-5×. Cronometrar del log (step 0 → step 190 ×6144 tok/step)。
  1 epoch esqueleto(129.5M tok) ≈8 min pool realista / 15-22 min solo
  L40+Q6000.
- El pool real **fluctúa por minuto** (A100 se ocupan/liberan todo el tiempo}.
  Sinfo miente: usar scontrol (AllocTRES vs Gres)。
- pro6000 (cu128) NO se mezclan en DDP con cu121(muro HITO 3)—familias

  homogéneas por partición + rendezvous TCP entre nodos.

- **Muro NCCL IB/noib documentado**: L40 r32r25n01 es noib,A100 son ib → el L40
  fallaba `NCCLUtils.hpp:275`. Fix: NCCL_IB_DISABLE=1 + NCCL_SOCKET_IFNAME.



---

## 4. ESTADO VIVO (HPC, al 13-09)

- **TESIS UNAM — nueva fuente (2026-09-13)**: descarga en curso en
  `/beegfs/a474r867/tesis_unam/corrida_doct/` (pdfs + md + checkpoint.jsonl
  con metadata DGDU: uuid/handle/title/degree/disciplina/area). ~293 docs
  ya; sigue madurando. **Decisión**: solo tesis de ciencia; traducción
  ES→EN DESPUÉS (cluster, v4serve o gpt-oss dedicado); mates/física/ing
  curadas aparte COMO LATEX para entrenar después. Curador listo:
  `scripts/curate_tesis_unam.py` → `tesis_unam/curada_v1/` con
  `science/` (138 docs, front-matter LFDA/Neevia/comité/índice/bibliografía
  eliminados — cortes por encabezado-seguido-de-prosa, TOC residual 0) y
  `latex/` (155 docs, conservados tal cual) + `meta.jsonl`. Nota: muchas
  son "tesis por artículos" con capítulos ya en inglés — la traducción
  solo cubrirá las partes ES. **Pendiente**: cuando madure la descarga,
  rerun del curador + worker de traducción (por capítulos, preservando
  estructura) → prosa larga inglesa para el corpus.

- **B3 — RESULTADO (2026-09-13 ~01:30 UTC, 10K steps completos)**: battery
  base: L0 0.527/L1 0.498/L2 0.606***/L3 0.518 → "estructura sin
  inferencia". Battery **dense**: L0 0.540*/L1 0.520/L2 0.665***/
  **L3 0.600*** (p=1e-5, n=475) → SEÑAL INFERENCIAL. Subtipos dense L3:
  direction_word 0.762 (n=193), causal_phrase 0.682, temporal_phrase
  0.667, mechanism 0.667, environment 0.636; **number 0.346 (n=52),
  negation 0.209 (n=43) → NO-GO en su criterio** (GO: >0.6). La capa
  sintética FLD no reparó la lógica fina; la tesis madre NO falsada
  (L3 sigue significativo). **Hipótesis nueva que B3 revela**: los
  subtipos que fallan son exactamente los que el 30% near-miss
  corrompía — entrenar contradicciones como positivos de denoising
  puede SUPRIMIR activamente la discriminación → test B4.
- **B4 — ABLACIÓN LANZADA (2026-09-13, pretok 29230699 → train
  29230700, afterok)**: receta B1/B3 EXACTA, mismo seed 42, único
  cambio: corpus sintético **100% válido** (0% near-miss) —
  `train_corpus_synth_b4.jsonl` 30K docs → `train_ids_b4.npz` →
  `runs/f0-span-v3-synthb4`. GO: number/negation >0.6 sin degradar
  (si suben → el supresor era el contraste enmascarado como positivo,
  hallazgo publicable; si no → el cuello es capacidad/objetivo).
  Bug gramatical cazado al inspeccionar el corpus: `dv2`/`dv3` usaban
  `dv_for()` que devuelve forma base con triggers plurales → "which
  lengthen" (~10% docs, también en corpus B3 ya entrenado). Fix:
  sujeto siempre singular → siempre 3a persona. Regenerado pre-pretok.
- **B4 — RESULTADO (2026-09-13 ~04:45 UTC, 10K steps)**: dense L3
  **0.585*** (n=475), L2 0.617***, direction 0.762, mechanism 0.667,
  causal_phrase 0.636 — pero **number 0.308 / negation 0.186**,
  idénticos a B3 → **NO-GO: el near-miss-como-positivo NO era el
  supresor**. La frontera number/negation es de mecanismo/capacidad,
  no de composición del corpus. Se cierra el eje "quitar contraste".

- **EVOG0 — búsqueda evolutiva por torneo LANZADA (2026-09-13,
  jobs 29230976-983 + prep 29230975)**: la meta sube a piso por
  subtipo (min(number,negation)>0.6 antes de pensar en 0.7). G0 =
  pantalla de 8 ejes single-gen vs control B4 (~90M, 10K steps,
  `train_ids_b4.npz`): `g0-aug` (etapas inferenciales REALES aug_b2,
  gen nunca testado), `g0-marked` (corpus 70/30 con etapa final
  `[VALIDEZ]` post-CONCLUSION — validez como dimensión derivable),
  `g0-cap` (hidden768/l8 ~150M), `g0-ep2` (20K steps, pro6000),
  `g0-cf05` (candidate_focus 0.50), `g0-nocur` (sin curriculum),
  `g0-lr4` (lr 4e-4), `g0-norole` (role_mask off). Fitness =
  0.5·L3_dense + 0.5·min(number,negation) — maximin anti-direction.
  Selección sobre dev `pairs_hard_v3_eval` (congelado); holdout
  `pairs_hard_v4_holdout` (n=2000, seed 4242) solo para el campeón.
  Infra: `scripts/g0_run.slurm` (runner genérico env-driven,
  auto-detect Blackwell→overlay/venv vs python3 base), `g0_prep.slurm`
  (corpus b4m + b4aug + holdout), `leaderboard.py` (fitness+ranking).
- **EVOG0 — RESULTADO (2026-09-13)**: 8/8 evaluados. Ranking fitness
  (0.5·L3+0.5·min(num,neg)): **g0-cf05 0.431** (L3 0.606, neg 0.256),
  **g0-aug 0.406** (L3 0.602, num 0.365 — único >0.35 con solo 902 docs
  aug ≈0.3% share), norole 0.394 ≈ B4-control 0.393, marked 0.387,
  nocur 0.371, cap-150M 0.367, ep2-20K 0.358, lr4 0.333. Lectura:
  candidate_focus sube L3+negación; las etapas inferenciales REALES
  mueven number aun con share mínimo. Capacidad, épocas, lr-alto,
  marcador [VALIDEZ], quitar curriculum o role_mask: nada ayuda → la
  frontera responde a énfasis-de-candidato + dato inferencial real.
  Fix en ruta: `suite_smoke_logicdiff` ganó `--model-override` (JSON
  pisa cfg.model; g0-cap evaluaba hidden512 contra ckpt 768) + el
  fallback `module.` solo aplica si el prefijo existe (g0-cap había
  fallado con keys truncadas-7 enmascarando el size mismatch real).
- **EVOG1 — LANZADA (2026-09-13, jobs 29231473-482 + watch 29231481)**:
  6 mutantes de top-2 + 2 probes, todos heredan CF=0.5 salvo
  intensificación: `g1-cf07`, `g1-cf05aug`, `g1-cf07aug`,
  `g1-cf05norole`, `g1-numbias` (ROLE_CONFIG number_weight=6 +
  negación "not/no/never/cannot" como conectivas — sesgo de masking
  directo a la frontera), `g1-aug5x` (aug ×5 = 4510 docs), probes
  `g1-cf05rand` (mask_type=random: ¿span importa bajo candidate-focus?)
  y `g1-cf05lr1` (lr 1e-4). Watcher G1 activo.
- **EVOG1 — RESULTADO (2026-09-13)**: BREAKOUT — `g1-cf05rand`
  (mask_type=random + CF=0.5): FIT **0.495**, L3 0.619, **num 0.385,
  neg 0.372** (los dos subtipos débiles ~duplicados vs ~0.31/0.19).
  Mecanismo: el span forzaba coherencia local; el masking aleatorio
  obliga a inferir tokens desde evidencia dispersa — mejor alineado
  con la discriminación. 2° `g1-cf07aug` 0.445 (dir récord 0.793);
  `g1-numbias` señal débil (neg 0.233). -> **EVOG2 lanzada**
  (g2_launch.sh, 8 jobs): mutantes del ganador — randaug, randcf07,
  randnumb, randaug5x, randws0, randnr + probes randhi (mask_p
  0.30-0.99) y cf07aug (intensificación total).
- **EVOG2 — RESULTADO + HOLDOUT (2026-09-13)**: ningún mutante superó
  a cf05rand en dev (top g2-cf07aug 0.456). PERO el **holdout v4**
  (n≈2000/nivel, number 201 / negation 157, seed 4242 — primera
  medición de alta resolución) **cambió el campeón**: `g2-cf07aug`
  (random + CF0.7 + aug): **L3 0.617, L2 0.702, number 0.408,
  negation 0.408** vs cf05rand 0.593/0.383/0.274 (su negación dev
  0.372 era mirage de n chico). Frontera verificada: ~0.41 en ambos
  subtipos débiles — real pero <0.6. Lección metodológica: seleccionar
  en dev n=475 sobrevende; el holdout redefine el podio.
- **EVOG3 — LANZADA**: mutantes del nuevo campeón sobre corpus aug
  refrescado `train_ids_b4aug2.npz` (B2 maduro: 11,964 docs únicos,
  13x vs los 902 de b4aug) — 13 configs saturando el pool
  (g3_launch.sh): replica-varianza `g3-champ`, eje aug-viejo
  `g3-champv1`, barrido CF {0.5,0.85,1.0}, mecanismo {ws0, norole,
  numbias}, densidad random {hi 0.30-0.99, lo 0.02-0.50}, receta
  {lr4, ep2-20K, cap-150M}. Watcher g3. Holdout v4 queda reservado
  para el campeón de campeones. Réplicas añadidas mid-flight:
  `g3-champ-s2/s3`, `g3-numb-s2`, `g3-cf085-s2`, `g3-ws0-s2` (18 runs).
- **PROTOCOLO seeds/réplicas (vigente desde G4)**: toda corrida lleva
  `SEED=k` explícito (`--seed` en trainer; init + orden de datos +
  masking quedan deterministas por seed). Convención de tag:
  `gN-<genoma>-s<k>`; la corrida base sin sufijo cuenta como s1.
  Réplicas = mismo genoma, solo cambia seed. Selección por MEDIA del
  fitness entre seeds (`leaderboard.py` agrupa `-sK` y reporta
  mean±sd) — nunca best-of-N (infla al ganador). Candidato a campeón
  requiere ≥2 seeds antes de tocar holdout; una mejora solo cuenta si
  supera al campeón en media por más que la sd entre seeds. Registro
  de seeds usados por generación en el launcher correspondiente.
- **Gen de datos `logic` (build_logic_synth.py + g4_prep_logic.slurm)**:
  corpus sintético de consistencia fina, 30K docs **100% válidos**
  (lección B4), donde número/negación son load-bearing — CONCLUSION
  determinada por EVIDENCIA: magnitud exacta (20%), magnitud derivada
  40→60 ⇒ "+50%" (30%), nulo con polaridad preservada (25%),
  comparación/umbral (15%), temporal (10%). Corpus:
  `train_ids_b4aug2logic.npz` = skeleton_v2 + augv2(11,964) + logic(30K)
  = 341,498 docs. Motivo: el corpus esqueleto YA es denso (73% docs con
  dígitos, 29% con negación) — el cuello no es frecuencia sino que el
  objetivo no premia consistencia fina; este corpus hace que el token
  correcto esté forzado por contexto bajo denoising. `g3-logic` lanzado
  como probe temprano (receta campeón rand+CF0.7 + corpus logic).
- **EVOG3 — RESULTADO (19 runs, 14 configs)**: nuevo líder dev
  **`g3-hi`** (mask_p denso 0.30-0.99): FIT **0.492**, num 0.365, neg
  0.372 — supera al campeón (0.446±0.012, n=3) por ~4σ de corrida.
  Sorpresas del cierre: `g3-nr` (role_mask off) mejor L3 0.625 y mejor
  negación 0.419 — el sesgo por rol ayudaba bajo span pero distorsiona
  bajo random; `g3-cf10` (CF=1.0) 2º (0.471) — el eje CF quiere el
  extremo. Negativos: `g3-cap` divergió (NaN ~step 2360: 150M+random
  inestable a lr2e-4), `g3-logic` no movió la frontera (num 0.346 pero
  neg 0.233), aug 13x plano (champv1≈champ), numbias inestable
  (σ_neg alta). Lectura convergente: la frontera responde a *volumen
  de práctica inferencial* (densidad de masking + steps), no a
  arquitectura ni a marcadores.
- **EVOG4 — LANZADA (15 jobs, g4_launch.sh + añadidos nr/cf10)**:
  mutantes del líder hi: réplicas s2/s3 (σ propia), hi×ep2 (2 seeds —
  cruce de los ejes que movieron negación), hi×ws0 (2), hi×nr (2),
  hi×cf10, hi×nr×cf10 (combo de ganadores), barrido mask_p
  {0.45-0.99, 0.20-0.90}, hicf05, hilogic, hilr1. Watcher g4.
- **EVOG4 — RESULTADO PARCIAL FUERTE (13/15 con batería)**:
  `g4-hicf10` (hi+CF=1.0, n=1): **L3 0.640 / neg 0.651** — primera
  medición por encima del piso 0.6 de negación en dev. `g4-hinrcf10`
  (hi+nr+cf10): FIT 0.480, neg 0.558, dir 0.788. CF=1.0 bajo densidad
  = la condición de eval aplicada al 100% de ejemplos; parece además
  *estabilizar* el régimen denso: `hi` solo dio neg {0.302, 0.186}
  entre seeds (sd 0.036) — el líder G3 fue en parte seed afortunado.
- **EVOG4 — CERRADA (18 runs)**: medias por config — hinrcf10 n=2
  FIT 0.477/neg 0.546; hicf10 n=3 FIT 0.474/neg 0.527 (0.651→0.395:
  rango amplio pero piso lejos de ~0.28 del campeón); hiep2 n=2
  FIT 0.478/neg 0.419. PPL proxy: champ 2043 < hicf10 2468 (+21%)
  < hinrcf10 2988 (+46%) — el avance en razonamiento cuesta calidad
  LM; restricción PPL activa. **Backbone G5 = hinrcf10** (media,
  simplicidad). Holdout lanzado para las 5 réplicas de ambos.
- **EVOG5 — LANZADA (8 jobs, base=hinrcf10)**: corr-s1/s2 (p=0.10),
  corrhi-s1 (0.20), numw-s1/s2 (3.0/2.5), corrnw, cos (schedule coseno
  puro), mrd (decaimiento b_h 0.99→0.30). Conducción autónoma:
  gen_wait + gen_overseer.
- **HOLDOUT v4 (primera medición verificada de la frontera)**:
  hinrcf10 n=2 → L3 0.628 / num 0.396 / neg 0.592 (s1: 0.643);
  hicf10 n=3 → L3 0.618 / num 0.420 / neg 0.514. L3 ~0.62 supera
  el piso 0.55 pre-registrado → la tesis no queda falsada a esta
  escala; num mejoró vs dev (~0.32→0.40). La frontera a empujar:
  num+neg → ≥0.6 simultáneos en holdout.
- **EVOG5 — CERRADA (8 runs, base=hinrcf10)**: schedules cos/mrd
  empatan al backbone (L3 0.638-0.642, neg 0.56-0.58) — bajo CF=1.0
  la mascara es siempre etapa completa: genes de tasa/schedule INERTES
  (lectura mecanistica confirmada). numw NEGATIVO confirmado (n=2:
  neg 0.256/0.279, num plano) — ponderar loss en digitos degrada.
  corrective bimodal con gradiente adverso: p=0.10 -> {0.279,0.581},
  p=0.20 -> 0.419, corr+numw -> 0.163. num inamovible en 0.327 en
  TODA la generacion -> posible piso estructural del formato.
- **EVOG6 — LANZADA (8 jobs)**: mras-s1/s2 (gen nuevo: masking
  adaptativo por EMA de CE; bajo CF10 pondera eleccion de etapa),
  corrlo-s1/s2 (p=0.05, media dosis), cormras (corr+mras), mrashi
  (gamma=1.0), base-s1/s2 (control interno).
- **EVOG6 — PARCIAL (base=hinrcf10)**: mras-s1 0.634/neg 0.488,
  mras-s2 0.615/neg 0.465 -> masking adaptativo ~= backbone, no lo
  supera. base-s2 (control) 0.625/0.512 valida la referencia interna.
  corrlo/cormras/mrashi/base-s1 quedaron en cola (cluster saturado).
  **Convergencia del GA: hinrcf10 es optimo local duro — todos los
  genes lit-review empatan o degradan.**
- **EVOG7 — CERRADA (dev v3_eval)**: champ-s4 L3 0.602/neg 0.349,
  champ-s5 L3 0.642/neg 0.512 → hinrcf10 n=6 total (g4 s1/s2 +
  g6-base s1/s2 + g7 s4/s5): L3 0.602-0.642, neg 0.349-0.558,
  num 0.308-0.327. **chnrep-s1 (20K steps): L3 0.632/neg 0.465/
  num 0.346 — el eje "más práctica" (2x steps) NO mueve la frontera.**
  PPL proxy: s4 2275, s5 2920, chnrep 3102 (2x steps degrada PPL
  sin ganar frontera). Backbone confirmado como óptimo local.
- **LEAKAGE CHECK (embeddings e5-small, emb_v1)**: audit sobre
  skeleton_v2 + holdout. Dedup: 5,882 docs duplicados (2%)>0.95.
  **Fuga real detectada**: el holdout NO se separo por documento
  fuente — 82 docs del corpus y 215 items de holdout comparten
  abstract fuente (>0.95; la masa >0.90 = familia-de-template, no
  copia). Holdout ~97% limpio; inflacion acotada pero real.
  Construido  (7,509 items).
  **Re-eval limpio confirma la frontera**: hinrcf10 L3 0.632/
  num 0.389/neg 0.645 (vs contaminado 0.633/0.643 — la fuga no
  inflaba la metrica). hicf10 0.624/0.425/0.561. neg>0.6 en
  holdout limpio = frontera verificada estricta.
- **CORPUS v3 (data/corpus_v3.jsonl, 344,837 docs)**: skeleton_v2
  filtrado (dedup+leak) + augv2 + synth_logic + UNAM 48,591 chunks
  (lang=es, domain=unam_<area>). Balanceo: cap 40K/dominio -> max
  11.6% (antes medgen 33%). Metadatos src/lang/domain por doc.
  knn_edges.tsv (16-NN) disponible para hard-negative mining.
  **Pretokenize COMPLETO → train_ids_c1.npz (147.6M tok, seq768).**
  Audit c1 final (report.json): skeleton dedup 5,882 drop / leak
  56 estrictos (48,913 a >0.9 = familia-de-template); UNAM dedup
  10,386 drop (21% — mucha tesis por-artículos repetida), leak 683.
- **EVOG8 — CERRADA (2026-09-14, 4 runs)**: ablación de corpus sobre
  hinrcf10, misma receta, solo cambia DATA_CACHE (eval dev v3_eval en
  ambos). `g8-c1` = corpus v3 completo (147.6M tok, incl. ~40K docs
  UNAM-ES crudo): s91 L3 0.568/neg 0.302, s92 L3 0.598/neg 0.349 →
  **media L3 0.583, neg 0.326 — DEGRADA ~5σ vs backbone**. `g8-c1e` =
  v3 solo-EN (116.9M tok, 304,585 docs sin UNAM → train_ids_c1e.npz):
  s91 L3 0.646/neg 0.651/num 0.365/dir 0.777, s92 L3 0.608/neg 0.395/
  num 0.308/dir 0.756 → **media L3 0.627, neg 0.523 — RECUPERA la
  frontera** (hinrcf10 en mismo v3_eval: 0.626). PPL proxy: c1e 2723
  vs c1 2234 (irrelevante: c1 tiene mejor PPL y peor L3 — el ES crudo
  "ayuda" al LM y daña la discriminación EN).
  **Veredicto: el coste era el español crudo, NO el cap medgen/dedup.
  Corpus base nuevo = c1e (corpus_v3_en.jsonl). UNAM solo entra
  traducido ES→EN.** Lección permanente: nunca ES crudo al corpus.
- **REVISIÓN DE LITERATURA (Perplexity Agent API, preset medium;
  docs/lit_review/pplx_*.md)** — mapeo de nuestros genes ganadores a
  evidencia publicada y ejes nuevos que el GA no descubre por mutación:
  - `random`+`hi` (masking denso): Wettig 2023 confirma que masking
    uniforme requiere tasas altas vs span; MAE (75%) análogo en visión.
    MATIZ honesto: en difusión para *likelihood*, schedules coseno
    (media ~0.36, sesgo a ruido BAJO) ganan a uniforme — nuestro eval es
    discriminativo, no likelihood, lo que puede explicar por qué `hi`
    nos funciona aunque iría contra el consejo de la literatura de PPL.
  - `nr` (role_mask off): "Mask Is What DLLM Needs" (2026) — el masking
    estático por listas desperdicia señal; la versión correcta es
    densidad *adaptativa* (máscara ∝ dificultad actual del token).
  - `numbias` nuestra vs **DSFT** (2025): ellos ponderan la *LOSS* en
    tokens numéricos (w>1), no el masking — gen nuevo barato `numw`.
  - ELECTRA/RTD existe pero detecta *provenance*, no corrección; el
    precedente exacto a lo que necesitamos es **Corrective Diffusion
    Language Models (2025)**: corrupción mixta = masks + tokens
    visiblemente mutados supervisados a predecir el original →
    entrena "este número visible está mal" = alineación directa con
    el eval pairwise. Gen nuevo `rtd`/corrective (implementación media).
  - Curriculum de bloque fine→coarse (DreamReasoner, T⋆) y MRD
    (masking-ratio decay 30%→15%): soportan curricula; nuestra
    dirección span16→64 ya es la versión barata.
  - Fase RL (d1/diffu-GRPO, d2): factible a ~100M para tareas
    verificables estrechas; NO demostrado para razonamiento amplio a
    esa escala; rollout cost real pero nuestro modelo es barato de
    muestrear. Backlog post-plateau, no gen G5.
  → Ejes G5 propuestos: `noiseskew` (schedule coseno/Beta vs uniforme
  denso), `numw` (loss-weight numérico), `corrective` (mutaciones
  visibles + corrección supervisada), `mrd` (decay de ratio).
- **GENES `numw` + `corrective` IMPLEMENTADOS** (train_mdlm_moe_v2.py
  V3.4, env: LOSS_NUM_W / LOSS_NEG_W / CORRECTIVE_P / CORRECTIVE_BOOST):
  - `numw`: CE ponderada por token (w en dígitos y negaciones; el resto
    1.0). DSFT-style — pondera loss, no masking.
  - `corrective`: con prob por posición, corrompe tokens VISIBLES
    (complemento de la máscara) y supervisa a predecir el original.
    Selección ponderada hacia tokens informativos (boost=8); mutaciones
    con sentido: dígito→dígito, negación→neutralizador ("not"→"also",
    "without"→"with"), verbo de dirección→antónimo ("increases"→
    "decreases"), resto→vocab uniforme. Verificado vs tokenizer real:
    num_ids=10 (nivel dígito), flip_ids=104 con variantes de case.
  Ambos no-ops con defaults; compatibles con resume y seeds.

- **PLAN NOCTURNO G5-G8 (autonomía 23:17→08:00, Devin al mando)** —
  infraestructura añadida: `eval_ppl_proxy.py` (CE de denoising a fracs
  {0.15,0.5,0.85} sobre ctx+ok del holdout — pseudo-PPL comparable entre
  runs, nunca toca etiquetas), `gen_overseer.py` (marca corridas muertas
  con verdict terminal → cierre sin colgar), `gen_wait.sh` (espera de
  generación), leaderboard ahora reporta PPL. **Nota de interpretación**:
  la familia `hi` llevaba CURRICULUM=1 implícito — el gen ganador real es
  "rampa b_h 0.30→0.95 sobre random", no denso plano desde step 0.
  - **G5** (~01:30): genes de la lit-review sobre backbone G4
    (hicf10|hinrcf10 según media de réplicas): `corr` 0.10 (2 seeds),
    `corrhi` 0.20, `numw` 3.0/2.5 (2 seeds), `corrnw`, `cos` (cosine puro,
    hipótesis contraria likelihood-lit), `mrd` (decaimiento b_h — dirección
    opuesta a la rampa heredada). Pregunta: ¿gen de objetivo (corrective/
    numw) > gen de schedule?
  - **G6** (~03:30): cruces de ganadores G5 + barrido fino del gen que
    gane (p.ej. corrective_p∈{0.05,0.10,0.15}, boost∈{4,8,16}, o dosis
    numw). Réplicas s2/s3 de lo mejor.
  - **G7** (~05:30): consolidación — 3+ seeds del mejor recipe;
    si media supera al incumbente por >σ entre seeds → candidato a
    holdout. Verificación PPL: un campeón que degrade ppl_proxy >50% vs
    baseline se reporta con asterisco, no se esconde.
  - **G8** (~07:00): holdout del campeón validado (≥2 seeds) +
    reporte matutino. Si nada supera al incumbente en media+σ, el
    resultado negativo limpio se documenta igual.

- **AUDITORÍA 2.0 cross-pipeline (2026-09-12, commit `7bec25f`)** — repaso
  completo de la cadena activa dLLM (línea confirmada como ruta de
  investigación). Fixes aplicados, todos compatibles con resume/olas:
  - `train_mdlm_moe_v2.py`: lookup de etiquetas de etapa vía `tolist()`
    (~30K allocs GPU menos por step; equivalencia verificada 300/300);
    parse defensivo de `checkpoint-g*` en `resume()` y retención-2 (un dir
    no conforme ya no mata save/resume).
  - `suite_smoke.py` + `suite_smoke_logicdiff.py`: el import del trainer
    heredaba su handler SIGUSR1 (guarda ckpt con glob_model=None → crash
    feo en eval ante preemption) → ahora se restaura el handler previo.
    **ctx tail-trunc**: `_load_pairs` conservaba `ctx[:max_ctx]` (cabeza);
    ahora `ctx[-max_ctx:]` (cola = etapa inmediatamente previa al
    candidato, la más informativa). Aplica a la battery de B3 esta noche.
  - `build_b3_synthetic.py`: retry-loop reescrito — `labels.append+shuffle`
    dentro del `for` re-asignaba labels ya consumidos → el corpus de prod
    salió 69.89% válido (esperado 70%; count correcto por convergencia,
    ratio levemente desviado — corpus usable, no regenerar). Y
    `NOM["slowed"]` "slow"→"slowdown" ("a slow of 41%" era inglés roto).
  - `augment_inferential_stage.py`: HTTP 4xx permanentes (400/401/403/404/
    422) → `"fatal"` → `CONFIG_EXHAUSTED` + exit 3 = **sin resubmit** (un
    400 en bucle quemaba cuota OR en puros rechazos — ya ocurrió con el
    array de 4 modelos). Renombrado `ebody` (sombreaba el dict request).
  - `verdict.py`: `--step` se ignoraba (siempre el último punto) → ahora
    decide en el último step ≤ N.
  - Verificado sin cambios: `b3_pretok_merge.slurm` (OOB check OK),
    `f0_v3_synthb3.slurm` (finalize/resubmit correcto, flag por slurm),
    `build_skeleton.py`, `eval_curve.py` (ya tenía tmp-por-PID + merge
    anti-lost-update), `build_pairs_hard_v3.py` (mutaciones L3 bien),
    `ecobench/` (línea secundaria, sin bugs críticos). Nota conocida:
    `train_mdlm_moe_hetero.py` no escribe `training_complete.flag` — el
    slurm lo escribe en exit 0; watchdogs externos deben usar
    `state.json step==TARGET` o `sacct`.

- **B3 — CAPA SINTÉTICA FLD LANZADA (2026-09-12, job 29226824, r23r09n01
  RUNNING)**: receta B1 EXACTA (span64/curriculum/role_mask/candidate_focus
  0.30, 10K steps) sobre el corpus mezclado `data/train_ids_b3.npz` =
  skeleton_v2 (299,534 docs) + 30K docs sintéticos deductivos
  (`build_b3_synthetic.py`: cadenas premisa→derivada→conclusión con
  vocabulario ecológico, 70% válidos / 30% near-miss con dirección o número
  roto; subtipos débiles forzados: números %, negación, conectivas,
  temporales; plantillas gramaticales cortas por RSD). Mezcla ~5% por tokens,
  9% por docs. Pretok+merge (job 29226808) verificado OOB_OK (max 126077).
  Battery dense+base automática al COMPLETE. **GO: subtipos number/negation
  >0.6 sin degradar el resto** (vs B1: number 0.31/negation 0.19).
- **B2 — AUMENTO INFERENCIAL REDIMENSIONADO (2026-09-12, jobs 29226763-766,
  4 shards × N=2500 = 10K docs totales, test-and-drop)**: el dimensionado
  previo (N=40000/shard = 160K) era 3-5× la meta del doc y con el ritmo
  inicial medido (0.04 docs/s) daba ETA 547h/shard — CANCELADO y relanzado.
  Ritmo real nuevo con el batch del serve lleno: ~0.29 docs/s/shard →
  estima ~4h para 10K. `train_skeleton_aug_b2.jsonl_s{0..3}` + .done
  (resume por pid). Al COMPLETE: merge → pre-tokenize → training con la
  receta B1 → battery dense (GO: dense L3 sube Y mejora en causal_* /
  mechanism).
  **Rev 12-09 ~17:15 CDT**: yield de parseo ~45% (el teacher emite
  [HIPOTESIS] pero omite [PREDICCION]; solo 2/15 fails contienen
  "prediction") → ritmo efectivo ~0.21 docs/s → ~7h+/shard → 2+ olas.
  **FIX APLICADO + RELANZADO (12-09 ~18:00 CDT, jobs 29226869-872)**:
  prompt endurecido ("EXACTLY two lines and nothing else" + recordatorio
  en el turno user) + parser de bloques (`OUT_MARK`: contenido misma
  línea o bloque siguiente hasta línea en blanco/otro marcador;
  `**Hypothesis:**`, bullets, `[HIPÓTESIS]\n<contenido>`) + net_err no
  marca .done (aborta FATAL tras 5). Resultado: **fail≈0** en las
  primeras ~25 llamadas (vs ~55%). CUELLO REAL medido: teacher
  ~30s/llamada con 4 clientes paralelos (NUM_PARALLEL=8, solo 4 en uso)
  → ~0.03-0.04 calls/s/shard → 2500 ok ≈ **18-21h/shard → ~4 olas**.
  Cadena de serves v4 cubre la muerte de 29226410 (~20:15 local):
  29226860→29226861→29226862 (afterany, ~18h; quizá haga falta 1 más
  mañana). Acelerador disponible NO aplicado: 2 hilos por shard para
  usar los 8 slots del serve (~mitad de tiempo).
  **Backend OpenRouter IMPLEMENTADO + LANZADO (12-09 ~19:15, worker
  29227048 RUNNING)**: egress desde cómputo verificado (200 a OR desde
  srun). Key en `~/.openrouter-key` (600) del cluster — el script la
  lee por `OPENROUTER_KEY_FILE`/`~/.openrouter-key`, nunca viaja en el
  env del job. Cuenta NO free-tier (is_free_tier=False) → 1000 req/día
  cuenta-wide en `:free`, 20 rpm. `models` fallback server-side
  **max 3 items** (400 si más — bug cazado en prod), `reasoning.enabled
  =false` (los :free son razonadores y consumían max_tokens sin emitir
  content). Config viva: OR_MODELS='gemma-4-31b-it,nemotron-3-super-
  120b,nemotron-3.5-lightning' PACE=3.2 MAX_TOKENS=600 REVERSE=1
  OUT=train_skeleton_aug_b2_or.jsonl_s0 CLAIMS/DONE_GLOB compartidos
  con los shards modulo (claims atómicos O_EXCL + .done hermanos;
  merge final dedup por pid). Yield ~96%+, ~5.3s/doc → quema la cuota
  en ~1.5h → QUOTA_EXHAUSTED sin resubmit (re-lanzar mañana si hace
  falta; valor real = hedge si se pierden las L40, no throughput).
- **B1 — CANDIDATE_FOCUS NO-GO (2026-09-12, `f0-span-v3-candfocus` 10K steps,
  battery automática)**: battery base (random15): L0 0.52 / L1 0.51 / L2
  0.582*** / L3 0.518 ns → "estructura sin inferencia"; battery **dense**
  (canónico Fase A): L0 0.544* / L1 0.530 ns / L2 0.663*** / **L3 0.583******
  (p=1.7e-4) → señal presente PERO **débil del mejor base (contrastive 0.640,
  v3-role 0.592)** → **NO-GO según criterio (≥0.66 / Δ+0.02)**. Subtipos:
  direction_word 0.741 (n=193) sigue siendo el grueso; **number 0.31 y
  negation 0.19 DEGRADAN vs azar** — el objetivo candidate-focused no repara
  la lógica fina → **el objetivo no era el cuello: el INPUT sí**. Reporte:
  `docs/results/LOGICDIFF-FASE-B1-RESULTADO.md`. Siguiente: B2 (input real-
  aumentado) y B3 (sintético FLD) en paralelo.
- **LOGICDIFF FASE A — LA FAMILIA dLLM SE REABRE POR CORRECCIÓN DE MÉTRICA
  (2026-09-12)**: el NO-GO de V3.3 queda **supersedido**. El scorer de la
  battery (máscara random 15% sobre ctx+candidato) diluía la señal L3 ~6× en
  ruido de contexto. El nuevo modo `dense` de `harness/suite_smoke_logicdiff.py`
  (enmascarar el 100% del candidato, reconstruir condicionado al contexto)
  revela: **v3-contrastive-fixed L3 0.640*** (p<10⁻⁵, IC95 [0.596,0.682])**,
  y **8/8 checkpoints entrenados significativos** (0.545-0.640, p<0.03 todos)
  vs random-init 0.459 ns (piso limpio). El scheduler de orden lógico NO es
  el mecanismo (staged≈rev≈dense; rand peor) — la barrera era el evaluador.
  Desglose por subtipo: la señal se concentra en `direction_word` 0.76-0.78
  (n=193) y causal/mechanism; `number`/`negation` siguen en azar o invertidos
  → discriminación direccional real, lógica fina ausente. Documento:
  `docs/results/LOGICDIFF-FASE-A-RESULTADO.md`. Uso defendible: el dLLM como
  **scorer de continuaciones** dentro del controller/verificator (rerank de
  argumentos, verificación de drafts), no como generador.

- **V3.3 CONTRASTIVO — NO-GO SUPERSEDIDO (2026-09-11, rev. 09-12)**: con el fix
  anti-contaminación (split por contexto 373/93, mask_id del tokenizer, guard
  de memorización; auditoría en docs/results/AUDITORIA-V3.3-CONTAMINACION.md),
  el training (job 29184168) completó y eval_acc_holdout ranking = 0.7742.
  Pero la battery LIMPIA (eval set pairs_hard_v3_eval seed 9999, 96.6% ctx
  nuevos; jobs 29184174/75) da L0 0.485 ns / L1 0.506 ns / L2 0.594*** /
  **L3 0.516 ns (p=0.26)**. La señal ranking NO transfiere al mecanismo denoise
  de la battery → artefacto del mecanismo de scoring del fine-tune, no
  inferencia. Según criterio del doc (<0.52): **NO-GO — familia dLLM CERRADA**
  (9 runs, L3 nunca significativo). Indicación: archivar dLLM-puro como
  negativo publicable y pivotar cómputo a controller/verificator (Opción D,
  match_args 98.2-98.6%). Datos en docs/results/v3_contrastive/.
  V3.2 role (job 29184149) aún corriendo como cierre de familia; su veredicto
  no cambiará la decisión salvo sorpresa mayúscula.

- **V3.1 CURRICULUM LANZADO (2026-09-10, job 29068111, r23r09n01 RUNNING)**: run
  `f0-span-v3-curriculum`, receta piloto v2 (span/uniform/whole_stage, corpus
  esqueleto v2 fix, 10K steps, batch8+accum2, lr 2e-4 cosine warmup 200) +
  **curriculum de masking** (CURRICULUM=1): `cur_stages=[[0,2000,16,0.30],
  [2000,5000,32,0.60],[5000,10000,64,0.95]]` con interpolación lineal continua
  de span_len y b_h (implementación Devin PR #3, mergeada d9c8a4e; test
  scripts/test_curriculum.py pasa: boundaries exactos + monotonicidad).
  Log verificado: `curriculum on: steps=3 stages, start(span=16, b_h=0.30)`,
  loss 11.83→8.16 en primeros 90 steps (~10 steps/min → 10K ≈ 3.5-4h).
  **Battery al COMPLETE: automática contra pairs_hard_v3** (PAIRS_DIR v3) →
  L3 comparable directo con V3.0 (0.515). Watchdog: cron `ecoreasoner-v31-watchdog`.
  **PITFALL lanzamiento**: `sbatch --export='A=...;B=...'` vía ssh NO llega
  íntegro (las comillas simples se pierden → el `;` parte el comando → solo la
  primera var llega; `--export-file` tampoco se respeta en este clúster).
  **ÚNICO método validado**: archivo `runs/f0-v3curriculum-env.sh` con
  `export VAR=...` (CUR_STAGES entre comillas simples DENTRO del .sh) +
  `source ... && sbatch --parsable --export=ALL scripts/...`.
  El run usa el envío correcto; verificar job→log "curriculum on" antes de dar
  por bueno cualquier relanzamiento.

- **V3.2 ROLE-AWARE MASKING LISTO, SIN LANZAR (2026-09-10, Devin PR #4
  mergeada ba7752b)**: masking ponderado por rol semántico (conectivas
  causales/adversativas, verbos de relación, números, entidades por mayúsculas
  + keywords; SIN NER). Aditivo: `--role_mask` default OFF → comportamiento
  idéntico al V3.1. Interacción curriculum: el curriculum fija span_len/b_h,
  role_mask decide DÓNDE ubicar los spans (priorizando regiones informativas);
  con whole_stage pondera la selección de etapas. Verificado por Hermes: test
  scripts/test_role_mask.py (a) sin flags == V3.1 misma seed, (b) 2.65× tokens
  de rol enmascarados (random) y 0.62 vs 0.41 (span), (c) compat curriculum,
  (d) slurm moe_v4_micro_v2.slurm exporta ROLE_MASK/ROLE_CONFIG en resubmit de
  olas + CLI. Config: harness/configs/f0-span-v3-role.yaml (validada 13/13).
  **Lanzamiento CONDICIONADO al veredicto V3.1** (§4 criterio: L3 0.52-0.54 →
  iterar V3.2-V3.4): si V3.1 no despega, V3.2 es la primera iteración a probar.

- **ABLACIONES PILOTO v2 COMPLETAS (4/4, 2026-09-10) — todas STAGE_GRAMMAR**:
  no-whole-stage L2 0.596*** / weight-tying L2 **0.629*** / rope L2 0.596*** /
  50m L2 0.557**. L3 NUNCA significativo (0.47-0.51 ns) en NINGUNA variante.
  Lectura: el whole-stage NO es responsable de L2 (conserva 0.596 sin él);
  weight-tying da el mejor L2 (0.629); 50M no es limitante (0.557) → la ausencia
  de L3 no se explica por capacidad. Línea dLLM-puro = estructura SIN inferencia
  (material negativo publicable). Reporte: docs/results/ABLACIONES-V2-RESULTADO.md.
  Decisión pendiente: (A) archivar línea falsada-en-L3, (B) objetivo contrastivo
  L3 explícito, (C) focalizar en controller/verificator (Opción D, 98.6%).

- **PILOTO v2 COMPLETADO — VEREDICTO STAGE_GRAMMAR (2026-09-09 20:52)**: job
  28998750 COMPLETED 0:0 en 3:48:38 (10K steps, loss 11.79→6.91, checkpoint-g10000).
  Battery L0-L3: **L0 0.5039 ns / L1 0.4724 ns / L2 0.6118*** / L3 0.5065 ns**.
  interpretation.json: "El modelo aprendió orden/rol de etapas, pero no el
  contenido inferencial" → STAGE_GRAMMAR, recomendación ablate_or_scale.
  NO encaja en el tricótomo §13.4 (FALSIFY exige L2≤0.52; DIRECCIONAL/GO exigen
  L0/L1≥0.55): es el PRIMER resultado donde la receta v2 (whole-stage span sobre
  esqueleto) enseña gramática estructural SIN atajo temático (vs F2 que falsó todo;
  Sanity F2: L0 0.57/L1 0.56/L2 0.47/L3 0.51). L2 0.61 p=0.0 CI95 [0.57,0.65] sólido.
  Siguiente: ablaciones que Devin preparó (WHOLE_STAGE=0, WEIGHT_TYING=1,
  USE_ROPE=1, 50M) — lanza Hermes cuando Devin las pase.

- **FASE 3 — subcorpus eco GENUINO listo**: filter domain_fine v5 (17 dominios
  finos eco) → 142,054 docs, 63,188 candidatos doc×tool, 25 regiones. Tagger GBIF
  (retry 0 errores): **93 especies reales** (usageKey+conf); 14+ ecológicas de
  campo reales (Picea abies, Pinus sylvestris, Fagus sylvatica, Daphnia magna,
  Vulpes vulpes, Cervus elaphus, Quercus robur, Ursus arctos, Rangifer tarandus,
  Oncorhynchus, Salmo salar, Capreolus...). El domain_fine SÍ separa eco de biomed.
  Pool listo para el siguiente bloque de tool-calls. Reporte:
  docs/results/MINERIA-ESPECIES-SUBCORPUS-ECO-V7.md + species_eco_fine_tagged.json.

- **F2-SPANES COMPLETO — FALSIFY** (50,000/50,000, job 28982555 COMPLETED 00:47:38,
  flag COMPLETE 21:34). Veredicto oficial (verdict_f2.json): pairwise_acc final
  **0.5273** (< falsify_threshold 0.55), slope últimos 3 **-6e-06**, delta primeros 3→
  últimos 3 +0.0546, n=69 pts. La curva tocó picos 0.5664 (28,351) / 0.5586 (30,951)
  pero **decayó al final** (últimos 10: 0.52-0.54) — el pico es el artefacto de
  comparaciones múltiples que la auditoría advirtió; el punto final decide. Receta
  micro ganadora ×5 pasos NO produjo HIT. **Detalle**: el run original 28962112 murió
  TIMEOUT 88% por el deadlock SIGUSR1 (confirmado) y se relanzó 28982555 con el fix;
  el resume a 44,151→50,000 fue limpio.
- **SANITY L0-L3 (batería Devin PR #1, ejecutada 2026-09-08)**: L0 0.5738** / L1 0.5607** /
  L2 0.465 ns / L3 0.5089 ns → solo coherencia temática (atajo overlap); **FALSIFICACIÓN
  FUERTE** de la línea inferencial. Vía libre al controller/verificator.
  Doc: docs/results/F2-SPANES-RESULTADO.md (anexo).
- **DECISIÓN (pre-registrada, ejecutada)**: línea f2 **archivada como falsada** —
  `docs/results/F2-SPANES-RESULTADO.md`. NO relanzar a 100K. **Siguiente: arquitectura
  controller/verificator** (deepseek/glm genera tool-calls + dLLM como conocimiento/
  repair — opción D del A/B). El dLLM propio queda como posible línea de fluidez 155M
  SOLO con gate de falsación reformulado (word/rep4/uniq son engañables).
- **OPCIÓN D — VALIDACIÓN 300/10 TOOLS ACEPTADA (2026-09-09)**: replicación sobre 300 pares
  / 10 tools verificada de forma independiente (no self-report): match_func 298/300 (99.3%),
  match_args 291/300 (97.0%). El controller zero-shot aguanta 10 tools sin degradación.
  Reporte: docs/results/REPLICACION-300-10TOOLS-RESULTADO.md + replication_300.jsonl.
  PENDIENTE: resolvers reales de las 7 tools nuevas (iucn/try/filogenia quedan mock),
  estratificar gbif (84 vs maxent 12) si va a banco público. Aceptado como hito.
- **REPLICACIÓN FASE 3 (500 tool-calls, 7 tools) — ACEPTADA**: match_func 500/500 (100%),
  match_args 458/500 (92%). fallos=42: 26 year-mismatch (2010->2015), 13 region
  (africa occidental->asia oriental), 3 species cercanas — errores de VOCABULARIO del
  controller, no de diseño. **RUN v2 (fix vocabulario): match_args 491/500 (98.2%)**,
  year 26->0 (2010 exacto, antes ni estaba en KNOWN_VALUES), region 13->0.
  Fix: KNOWN_VALUES años 2010-2026 + 23 regiones del gold; SYSTEM_PROMPT lista
  vocabulario válido. Residual 9: 4 ambigüedad Quercus ilex->suber, 3 región
  traducida es->en (southeast asia), 2 tool sinonima (inaturalist vs gbif, args exactos).
  Reporte: docs/results/REPLICACION-500-FASE3-RESULTADO.md (v2). Dataset:
  data/l1/toolcalls_fase3_500.jsonl (123 especies reales GBIF+Devin).
- **FASE 3 EN CURSO (2026-09-09)**: minería de literatura -> tool-calls.
  **Replicación 500 v2: match_args 92%->98.2%** (year 26->0, region 13->0; ver arriba).
  **Minería especies**: fulltext PMC completo → 1.32M candidatas pero BIOMEDICAS;
  subcorpus eco v7 (919K docs, 63%) → 84 GBIF-reales, solo 23 ecológicas reales.
  HALLAZGO: dominios GRUESOS v7 no separan eco de biomed (S.aureus/M.tb/levaduras dominan
  aun en "eco"). Para poso eco GENUINO: domain_fine del v5 (climate_ecology/population_ecology...)
  o fuentes eco-nativas (EcoEvoRxiv, ecoseek-litdump, arXiv eco).
  Reporte: docs/results/MINERIA-ESPECIES-SUBCORPUS-ECO-V7.md.
  Diseño: docs/designs/toolcalls-expansion-2026-09-09.md.
- **OPCIÓN D VALIDADA (2026-09-09)**: controller/verificator demo end-to-end.
  `verify_toolcall.py` (verificador M1-M5, schemas empiricos 533 toolcalls reales, 100% valido) +
  `ecobench/run_controller_verificator.py` (controller + verificator + retry) +
  `scripts/eval_controller_replication.py` (HITO 3: en 17 prompts tool-call genuinos
  match_func 76% / match_args 65%, 100% JSON valido; el resto del corpus = gold contaminado
  [prompts NO de las 3 tools etiquetados gbif], no fallo del controller).
  Teacher v4flash REACTIVADO como serve slurm + tunel :20006 (keepalive `2d884420a767` reanudado).
  PENDIENTE REAL: (b) HITO 4 opcional: resolve_tool SIN mock (GBIF/CHELSA reales).
  Commits: a9d85fe, 5b796f3, 7c68053, 548abbf.
- **GOLD LIMPIO (2026-09-09)**: curacion manual de los 60 prompts -> 14 tool-calls PURAS
  (12 gbif + 2 bioclim; 46 descartados: RF/filogenia/diversidad funcional/slurm/literatura/
  otras APIs — gold ruidoso del teacher original). REMEDICION sobre subset limpio:
  **match_func 14/14 (100%) | match_args 12/14 (86%)** — los 2 fallos de args son gold
  inconsistente con el prompt (Yucatan vs neotropico en item 2; year 2020 vs 2050 en item 4),
  no error del controller. Best-case de la opcion D medido y limpio. Artefactos:
  docs/results/replication_pure_14.* + curation_report.json.
- **Legado técnico (validado en prod, commits bf25fec/03a2cde)**: anti-reentrada
  SIGUSR1 (flag SAVING, test EAGAIN), sort -V ckpt, --index absoluto, strict=True en
  suite_smoke, tmp PID en eval_curve, warmup REAL (era arg muerto), verdict_f2.py
  arreglado (f-string <3.12).
- **GAP**: `train_mdlm_moe_hetero.py` NO escribe `training_complete.flag` (solo
  loguea COMPLETE) — watchdogs/herederos: detectar fin por `state.json step==TARGET`
  o `sacct COMPLETED`, NO por el flagfile. (El micro trainer SÍ escribe el flag.)




---

##5. PARA RETOMAR RÁPIDO(checklist)

**Línea viva = GA logicdiff sobre dLLM ~90-155M (ver §4 EVOG0-G8). Estado al
14-09 ~13:30 UTC:**

1. **G8 CERRADA — corpus base nuevo = c1e** (corpus_v3_en.jsonl,
   train_ids_c1e.npz, 116.9M tok). c1e media L3 0.627 = backbone 0.626
   en v3_eval (recupera frontera); c1 con UNAM-ES crudo degradó a 0.583.
   Nada en cola ecoreasoner (`squeue -u a474r867 | grep -E 'g[0-9]|c1'`).
2. Leaderboard: `ssh kuhpc 'cd /beegfs/a474r867/ecoreasoner && python3
   scripts/leaderboard.py --runs runs'`. Top FIT: g1-cf05rand 0.495,
   g3-hi 0.492, g7-chnrep 0.489, g5-mrd 0.485, g8-c1e 0.482.
3. Frontera verificada (holdout limpio): hinrcf10 L3 0.632 / num 0.389 /
   neg 0.645. Piso a empujar: num+neg ≥0.6 simultáneos — num ~0.31-0.37
   en TODAS las configs; neg oscila 0.33-0.65 por seed (alta varianza).
   GA convergió en hinrcf10; ejes agotados: densidad, CF, seeds, steps,
   schedules, genes lit-review.
4. Ejes sin probar si se reabre: corrective a dosis FINA controlada
   (p=0.10 fue gradiente adverso — probar p≈0.02-0.05), hard-negative
   mining con knn_edges.tsv (16-NN de emb_v1 ya calculado), arquitectura
   (objetivo, no receta).
5. UNAM (local /home/reumanlab/tesis_unam_scraper): flota w6-8 VIVA
   descargando (~1800 md nuevos + corrida_doct; slices 0-5 nunca
   lanzaron — decisión pendiente: 6 slices ×~977 docs más).
   Curador en loop 30min → curada_v2: ~1900 science + 155 latex y
   creciendo. **Traducción ES→EN LANZADA (14-09 ~21:44 local)**:
   `traducir_v4.py` (chunking por párrafos 8KB, detección ES/EN por
   stopwords — caps. ya-EN pasan intactos, escritura atómica por doc,
   resume por .en.md, backoff de endpoint muerto) contra DOS serves
   vivos: :20006=deepseek-v4-flash (v4serve 29435631, muere ~23:40 UTC)
   y :20000=glm-4.7-flash (q6000 29436238, muere ~01:50 UTC — túnel
   repuntado a r22r05n01:38595; el viejo 45947 estaba muerto).
   Medido: ~160 tok/s agg en los serves. **2º worker OpenRouter
   LANZADO (14-09 ~22:00 local)**: `or:inception/mercury-2.5` (dLLM
   de Inception — producción fabricando datos para el nuestro),
   $0.04/M in + $0.15/M out → corpus completo ~$10-15. Modo
   `--reverse` (docs grandes primero) anti-colisión con el worker
   local que va ascendente; mismo outdir/report, re-check de .en.md
   al abrir. **~6,600-7,600 tok/s agg medido** — grueso del corpus
   viable esta noche. Backup tier `or2:` = `minimax/minimax-m2.7:low`
   (solo si mercury se aparca; mezclar lento+rápido degrada cada doc
   porque espera a TODOS sus chunks). reasoning.enabled=false en
   mercury (lección B2); minimax-m2.5/m2.7 EXIGEN reasoning →
   sufijo `:low` + headroom max_tokens. Key en ~/.openrouter-key
   local (600, archivo — nunca env). Log: traduccion_or.log.
   PITFALL: un Q6000 NO puede servir deepseek-284B — modelo por
   endpoint obligatorio. Piloto qwen3:8b local = 4 tok/s (inviable).
   NO meter ES crudo al corpus (lección g8-c1).
   **PIVOT 15-09: traducción al CLUSTER, $0** (presupuesto $5 no
   alcanzaba el resto por API: ~51M tok out restantes ≥$8).
   OpenRouter PAUSADO (gasto ~$4.7). Modelo = `tencent/HY-MT1.5-7B`
   (MT especializado, HunYuanDenseV1ForCausalLM, ~p90 FLORES),
   HF cache `/beegfs/a474r867/hf_cache`, corpus en
   `/beegfs/a474r867/unam_mt_in` (463MB) → `unam_md_en` (seed
   341 docs ya hechos mergeados). Prompt card: "Translate the
   following segment into English, without additional
   explanation." + t0.7/topk20/topp0.6/rep1.05.
   GPU map real del cluster: "q6000"=Quadro RTX6000 Turing 24GB
   (SIN bf16 → fp16 obligatorio, ~30 tok/s HF eager); pro6000=
   RTX PRO 6000 Blackwell 97GB (sm_120 — necesita torch cu128);
   a100/a40/l40 también modernas. Stacks: `pylibs_mt` (transformers
   4.56 sobre torch 2.4.1 SIF, q6000), `pylibs_cu128` (torch 2.7.0
   +cu128+tv0.22 — para a100/pro6000 HF), `pylibs_vllm2` (vllm
   0.10.0+torch 2.7.1+cu128+transformers 4.56.2+tokenizers 0.22.1
   +numpy 2.2.6 — PROBADO: **491 tok/s agg en pro6000**, 7× eager).
   Triton necesita CC: `conda_cc` env en beegfs con gcc_linux-64.
   Workers: `traducir_batch.py` (HF, micro-batch por TOKBUDGET de
   prompt tokens — los logits fp32 de prefill dominan VRAM;
   apply_chat_template devuelve token_type_ids que HunYuan rechaza
   → filtrar a input_ids+attention_mask) y `traducir_vllm.py`
   (vllm offline llm.chat, olas de MAXDOCS docs). Slurm:
   mt_batch.slurm (q6000 fp16), mt_batch_cu128.slurm, mt_vllm.slurm
   — todos con SHARD_BASE+NSHARDS. Flota NSHARDS=44: pro6000 0-4,
   a40 5-8, l40 9-12, a100 13-23, q6000-fp16 24-43. ETA: con ~10
   GPUs vllm ≈ 5K tok/s → resto en ~3h de GPU real; se reencadena
   por ventanas sixhour (resume por .en.md).
   Merge final: rsync beegfs:unam_md_en → curada_v2/md_en (mismo
   nombre .en.md, dedup natural).
   **SPRINT→prototipo dLLM v1 (watcher automático)**:
   `scripts/watch_mt_done.sh` corre local (nohup, log
   `data/watch_mt_done.log`). Cada 5min: si pendientes>0 sin tasks
   → relanza arrays; si pendientes==0 → cae la BANDERA
   `data/UNAM_EN_DONE.flag` y encadena: (a) merge beegfs↔local
   md_en; (b) `corpus_v4en_build.slurm` = corpus_v3_en +
   unam_en.jsonl (`unam_en_to_jsonl.py`, filas
   {"pid","domain":"unam","lang":"en","source":"unam-en"}) →
   corpus_v4_en.jsonl + pretok → train_ids_v4en.npz; (c)
   `mdlm_retrain_v4.slurm` con --dependency=afterok:build —
   continúa retrain_v3/g8000 sobre v4en, TARGET_STEPS=11000,
   2nodos×4q6000, auto-relanza olas = prototipo dLLM v1.
   Eval posterior: `g_eval_holdout.slurm` vs frontera
   (0.632/0.389/0.645). PITFALL 15-09: el rsync de slurms puede
   quedar viejo — verificar el archivo EN beegfs, no el local
   (un array corrió shards 0-8 sin offset por eso).
   chat_proto.py: REPL local sobre checkpoint-gN (mask-predict,
   completion de prosa — modelo base, no instruct).

   **ÁRBOL DE CAPAS post-eval** (cada capa = corpus + pass sobre
   el ckpt anterior; arquitectura nueva solo en L5):
   L1 verificador = discriminación holdout (ya existe).
   L2 chat-SFT = diálogo/QA científico; teacher = LLaDA-8B-Instruct
   servido en cluster (llada_serve_1gpu.slurm); el propio v4 filtra
   respuestas malas (closed loop).
   L3 razonamiento = trazas premisa→paso→conclusión; reusar
   build_logic_synth.py / build_logicdiff_dataset.py como SFT.
   L4 tool calls = gen_toolcalls_* ya probado; scaffold externo
   parsea/ejecuta/reinyecta (agent-loop dLLM = denoise por bloques).
   L5 ESCALA (si v4 ≥ frontera): MoE ~1.5B activos / ~4B total,
   seq 768→1024. Necesita A100/pro6000 (q6000-24GB capa ~1B y
   sin bf16); ~2.5-3B tok/semana en 8×A100.
   L6 NUBE (una vez probado 1.5B/4B): costear renta — regla
   GPU-hr ≈ 6·N_activos·T_tokens / FLOPs_efectivos; ballpark
   1.5B-act × 10B tok ≈ ~170 A100-hr ≈ $250-500 spot.

---

##6. ARCHIVOS CLAVE

| Qué | Dónde |
|---|---|---|
| Diseño Fase 3 -new(autoridad) | `docs/designs/ecoreasoner-Fase3-DESIGN-new.md` |
| Diseño v0.2 (histórico) | `docs/designs/ecoreasoner-Fase3-DESIGN.md` |
| Resultado F1 (NO-GO,2026-09-08) | `docs/results/F1-HETERO-RESULTADO.md` |
| Estimación F1 heterogénea | `docs/designs/ecoreasoner-F1-HETERO-ESTIMACION.md` |
| Resultados micro-sweep | `docs/results/MICRO-SWEEP-F0-RESULTADOS.md` |
| Trainer heterogéneo | `scripts/train_mdlm_moe_hetero.py` |
| Lanzador pool multiplicidad | `scripts/f1_hetero.slurm` |
| Extractor esqueletos | `scripts/build_skeleton.py` |
| Harness (eval + report) | `harness/` |
| Fuente de verdad runs | `/beegfs/a474r867/ecoreasoner/runs/index.jsonl` |
| Corpus esqueletos | `/beegfs/a474r867/ecoreasoner/data/skeleton/train_skeleton.jsonl` |

---

##7. REGLAS QUE NO SE ROMPEN(aprendidas con sangre)

1. **Loss NO es proxy de lenguaje** — siempre smoke/eval de discriminación.
2. **test-and-drop**: hipótesis validable en <4h GPU ANTES de escalar.
3. **Pool teórico ≠ pool real** — medir con scontrol;lanzar todo el pool a la vez。
4. **No tocar trainer vivo** — copiar(train_mdlm_moe_hetero.py),nunca editar
   el que corre(bw5_spanhi usa train_mdlm_moe.py）。5. **sync a repo SIEMPRE** después de tocar scripts en HPC(scp + commit)—el
   repo es la fuente de verdad.
6. **YAML válido + sbatch --export sin comas**(usar `;`)—validar con
   validate_configs.py antes de lanzar。