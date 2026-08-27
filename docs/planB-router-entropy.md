# Plan B — Regularización de balance del router (entropía) para E2+E4 v3

> Fecha: 2026-08-27 · Estado: **APROBADO — implementado en ola v3 (job a lanzar)**
> Contexto: la ola v2 (27452507) terminó con **effN 2.49** (loss 0.593, aux 0.51).
> El Switch aux (f·P, alpha=0.01) **no logró desplegar el router**: subió effN de 1.76
> (mínimo en warmup) a 2.49, pero sigue lejos de 8. El plan A es insuficiente.
> Implementación: `entropy_reg()` en `moe_exp_E2_router_timestep.py` (L_ent = -beta*H,
> con grafo), flag `--ent_beta` en el trainer, slurm `moe_e2_v3.slurm` (aux 0.01 + ent 0.1,
> 6000 steps, cosine).

## Diagnóstico

El loss aux Switch `alpha * N * sum(f_e * P_e)` (con P detach) regula la **media** del gate,
pero:
1. Con alpha=0.01, el gradiente del CE domina (100×) — el router sigue optimizando CE puro
   y solo "siente" el aux cuando las asignaciones se polarizan mucho.
2. Switch aux penaliza la covariación f·P pero **no empuja directamente a que cada experto
   tenga masa de probabilidad** — permite que un experto quede con P baja si f baja con él.
3. Que effN suba de 1.76→2.49 sugiere que el aux SÍ tiene efecto, pero demasiado débil.

## Propuesta: Regularización por entropía del gate (Plan B)

En lugar de f·P, añadir un término que premie entropía MÁXIMA del softmax del gate por token:

```
L_ent = -beta * mean_tokens( H(gate(x_t)) )        # H = -sum p log p
```

- Se computa sobre las probs reales del gate (con grafo, NO detach — queremos que el grad
  fluya hacia el gate para hacerlo más "plano").
- beta controla la fuerza. Beta ~0.05-0.2 (probar barrido).
- Efecto: directamente desincentiva que el gate sea un one-hot; empuja a que el router
  distribuya probabilidad entre todos los expertos (entropía alta → softmax más plana).

**Mezcla con aux Switch**: mantener ambos — Switch para balance de carga (asignación
uniforme de tokens) + entropía para evitar colapso de capacidad. `L_total = CE + alpha*L_switch + beta*L_ent`.

## Implementación (cambios sobre train_mdlm_moe_expE2.py y router)

En `MoEMLP.forward`, ya guardamos `self._gate_probs` (detach). Para la entropía necesitamos
**con grafo**. Añadir:

```python
# en forward, training mode:
self._gate_probs_live = g          # con grafo (para entropía)
self._gate_probs     = g.detach()  # para switch (como ahora)

# nuevo método:
def entropy_reg(self, beta=0.1):
    """-beta * mean H(gate). Con grafo: grad empuja a softmax más plana."""
    if self._gate_probs_live is None:
        return torch.zeros((), device=self.gate.weight.device)
    g = self._gate_probs_live
    H = -(g * (g + 1e-12).log()).sum(-1)
    return -beta * H.mean()
```

En el trainer:
```python
ent = sum(b.mlp.entropy_reg(beta=ARGS.ent_beta) for b in model.blocks)
loss_total = loss + ARGS.aux_coeff * aux + ent   # si ent_beta>0
```

Flags nuevos: `--ent_beta 0.0` (default off; probar 0.05/0.1/0.2).

## Métricas para decidir

- **effN (N efectivo) por capa** al final: objetivo >4 en TODAS las capas (incl capas altas).
- Entropía media del gate por capa: objetivo >1.5 (con 8 expertos, uniforme = 2.079).
- Loss CE: no debe subir >5% relativo vs v2 (tolerancia de coste del balance).
- Correlación entre capa y effN: la pendiente (capa alta → menos expertos) debe aplanarse.

## Hipótesis

Con `ent_beta>=0.05`, el effN final sube de ~2.5 a >4, y la capa 11 (que era ~1 experto)
recupera al menos 3-4 expertos efectivos, SIN degradar el loss final por encima de 0.62-0.65.

## Próximos pasos si se aprueba

1. Añadir `entropy_reg()` al router + flags `--ent_beta` al trainer + log de |grad| del gate.
2. Smoke test (CPU, mini npy) → verificar que loss_total = CE + aux + ent converge.
3. Lanzar ola v3: 6000 steps, aux 0.01 + ent_beta 0.1, mismo ckpt_every.
4. Comparar: loss, effN por capa, entropía por capa, capa-11 effN → vs v1 y v2.
5. El ganador entre v2 (solo aux) y v3 (aux+ent) se revisa con analyze_activation + board.