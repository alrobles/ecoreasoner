# Quiz progress — EcoReasoner

Tracker del skill `/quiz`. Se actualiza al final de cada sesión.
Regla de dominio: un tema "pasa" con ≥80% en últimas 10 preguntas y streak ≥3.

## Lecturas (modo learn)

Formato: `✓` = lectura completada y comprensión evaluada.

| unidad | tema | lecturas hechas / total |
|---|---|---|
| U1 PyTorch core | tensors | L1 en curso (1/3 preguntas) |
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
| tensors | 1 | 0 | 1 | 0 | 2026-09-19 | confundió shape de gate: dijo N×D, es N×n_experts |
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

