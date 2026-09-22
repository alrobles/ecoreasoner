# Quiz progress — EcoReasoner

Tracker del skill `/quiz`. Se actualiza al final de cada sesión.
Regla de dominio: un tema "pasa" con ≥80% en últimas 10 preguntas y streak ≥3.

## Lecturas (modo learn)

Formato: `✓` = lectura completada y comprensión evaluada.

| unidad | tema | lecturas hechas / total |
|---|---|---|
| U1 PyTorch core | tensors | L1 en curso (2/3 preguntas) |
| | autograd | 0/2 |
| | modules | 0/2 |
| U2 Sistemas | ddp-zero | 0/2 |
| | memory | 0/2 |
| | ckpt | 0/2 |
| U3 Arquitectura | arch | 0/3 |
| U4 Difusión+datos | dllm | 0/3 |
| | data | 0/2 |
| U5 Infra+método | infra | 0/2 |
| | ga | 0/2 |

## Preguntas (repaso + comprensión + exámenes)

| tema | asked | correct | pts | streak | last_seen | notas |
|---|---|---|---|---|---|---|
| tensors | 2 | 0 | 2 | 0 | 2026-09-21 | P1: confundió shape de gate (N×D vs N×n_experts). P2 (.float() pre-softmax): intuición correcta pero mecanismo vago — no mencionó colapso de logits en bf16 ni grads del balance_loss |
| autograd | 0 | 0 | 0 | 0 | — | balance_loss detach vs diff, no_sync |
| modules | 0 | 0 | 0 | 0 | — | register_buffer, state_dict, EMA |
| ddp-zero | 0 | 0 | 0 | 0 | — | DDP wrap, ZeRO-1, consolidate |
| memory | 0 | 0 | 0 | 0 | — | grad_ckpt, fused AdamW, memmap |
| ckpt | 0 | 0 | 0 | 0 | — | atomic save, _try_load, SIGUSR1 |
| arch | 0 | 0 | 0 | 0 | — | MdLMMoE, Block, RoPE, TiedHead |
| dllm | 0 | 0 | 0 | 0 | — | mask schedules, curriculum, CE-masked |
| data | 0 | 0 | 0 | 0 | — | pretok, ids.npy, sft_mdlm |
| infra | 0 | 0 | 0 | 0 | — | slurm auto-VRAM, watchdog, señales |
| ga | 0 | 0 | 0 | 0 | — | G8-G11 falsación, logicdiff |

## Exámenes de unidad

| unidad | fecha | nota /20 | temas a repetir |
|---|---|---|---|
| u1 | — | — | — |
| u2 | — | — | — |
| u3 | — | — | — |
| u4 | — | — | — |
| u5 | — | — | — |

## Historial de sesiones

### 2026-09-19 — learn · tensors L1
- Lectura L1 presentada (dispatch del router, l.240-263).
- Comprensión 1/3: shape de `g` → **1 pt** (dijo N×D; correcto: N×n_experts,
  fila = distribución de routing del token).
- Pendiente al reanudar: P2 (¿por qué `.float()` antes del softmax?) y P3.

### 2026-09-21 — learn · tensors L1 (cont.)
- Comprensión 2/3: `.float()` pre-softmax → **1 pt** (intuición "transformar en
  alta precisión" correcta; faltó mecanismo: bf16 ~8 bits mantissa → logits
  cercanos colapsan a ties + grads degradados en balance_loss).
- P3 planteada (dispatch con k=2: ¿por cuántos expertos pasa un token, por qué
  `+=`?) — sin responder; usuario salió del modo estudio.
- Pendiente al reanudar: responder P3 → cerrar L1 → mini-quiz tensors.

