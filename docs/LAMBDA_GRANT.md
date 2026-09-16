# EcoReasoner → Lambda Research Grant (plan de solicitud)

Fecha: 2026-09-16. Estado: documentación de diseño — solicitud pendiente
hasta tener paper #1 con resultado de holdout.

## Estrategia

Secuencia: **retrain_v4 eval (holdout limpio) → paper #1 → grant Lambda
~$5000 → L5/L6 en nube**. El resultado falsificable (positivo o negativo
controlado) es la propuesta; el pipeline documentado es la evidencia de
capacidad de ejecución.

## Hallazgo clave: el grant = cómputo + colaboración

Toda la cohorte 2026 comparte coautor **Jianwen Xie (afiliación "Lambda"**
en AgentFlow). El programa premia papers terminados/casi terminados con
venue top + código abierto + angle de colaboración con Lambda — no
propuestas exploratorias. Solicitar sin paper listo = quemar la oportunidad.

## Cohorte 2026 (propuestas ganadoras)

| Proyecto | Link | Tipo |
|---|---|---|
| PixARMesh | https://mlpc-ucsd.github.io/PixARMesh/ | generación 3D/multimodal |
| (poster CVPR'26) | https://cvpr.thecvf.com/virtual/2026/poster/40153 | multimodal |
| AgentFlow (ICLR'26 oral, top 1.1%) | https://agentflow.stanford.edu/ | agentes planner/executor/verifier/generator + Flow-GRPO |
| OpenReview gF31wuYdk7 | https://openreview.net/forum?id=gF31wuYdk7 | (por revisar) |
| EdiVal-Agent | https://arxiv.org/abs/2509.13399 | framework eval multi-turn image editing |
| OffTopicEval | https://arxiv.org/abs/2509.26495 | eval operational safety de agentes |
| Meerkat | https://arxiv.org/abs/2506.03337 | federated ZO fine-tuning + sparsity |
| OpenReview Xn6EnJZghu | https://openreview.net/forum?id=Xn6EnJZghu | (por revisar) |
| ESPO | https://arxiv.org/abs/2512.03759 | **RL para dLLMs** (sequence-level ELBO) |
| (2604.03911) | https://arxiv.org/abs/2604.03911 | (por revisar) |
| TangoFlux | https://tangoflux.github.io/ | generación de audio |
| VideoNSA (ICLR'26) | https://arxiv.org/abs/2510.02295 | sparse attention para video |
| ECF8 | https://arxiv.org/abs/2510.02676 | compresión lossless FP8 |

## Qué gana (patrones)

- Agentes + RL verificable (AgentFlow: +14.9% search, +14.5% math con 7B)
- **RL para diffusion LLMs** (ESPO: +20-40 pts en Countdown) — directamente
  aplicable a nuestra capa L3
- Evals/benchmarks falsificables (EdiVal, OffTopicEval)
- Eficiencia/métodos (VideoNSA, ECF8, Meerkat)
- Todos: código + datos abiertos, venue top (ICLR/CVPR)

## Nuestro nicho (diferenciador)

La cohorte hace fine-tuning de modelos existentes. EcoReasoner entrena un
**dLLM-MoE desde cero** sobre corpus científico curado (118.5M tok EN,
pipeline completo documentado) con verificador + holdout falsificable:

- Más compute-hambriento → justificación natural del grant.
- Toca 3 ejes ganadores a la vez: difusión (ESPO), agentes
  (AgentFlow = nuestra arquitectura controlador/verificador), eval
  falsificable (holdout logicdiff).
- Ablación ya documentada: ES-crudo degrada el corpus → curaduría +
  traducción HY-MT1.5-7B (2,031 tesis UNAM, $0 en cluster).

## Qué compra $5K (precios aprox. Lambda on-demand)

- 8×A100-80GB (~$1.10-1.50/hr c/u) ≈ 3-4 semanas ≈ **10-15B tokens**
  → MoE 1.5B-activos/4B-total convergido, o 2-3B-act entrenable.
- O multi-run: pretrain + 3-4 ablaciones controladas (mejor para paper).
- Regla de costeo: GPU-hr ≈ 6·N_activos·T_tokens / FLOPs_efectivos;
  1.5B-act × 10B tok ≈ ~170 A100-hr ≈ $250-500 (spot puede ser menor).

## Checklist de solicitud (cuando toque)

- [ ] Paper #1 con tabla holdout (frontera vs v4) + ablación ES-crudo.
- [ ] Tabla de cómputo medido en cluster (tok/s por GPU, GPU-hr/step)
      → presupuesto defendible.
- [ ] Repo público o compartible: scripts + configs reproducibles.
- [ ] Presupuesto: ablaciones controladas > un solo run grande.
- [ ] Angle de colaboración Lambda (la cohorte lleva coautor afiliado).
- [ ] Backups: NAIRR pilot, GCP research credits, Modal/Anyscale.
