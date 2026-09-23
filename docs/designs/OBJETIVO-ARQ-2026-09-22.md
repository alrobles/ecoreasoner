# Línea Objetivo–Arquitectura (post-falsación B2)

Fecha: 2026-09-22. Estado: **diseño activo — brazo E-obj1 en preparación**.
Predecesor: `REPLANTEO-2026-09-22.md` (diagnóstico) + resultado B2 (#10).

---

## 0. Qué sabemos ahora (hechos, no hipótesis)

1. **El instrumento viejo mentía.** `dense` premiaba reconstrucción holística;
   `consistency`-sum (contexto visible, solo slot enmascarado) invirtió el
   ranking: v5-1b es la frontera real (0.652 inferable, 30% gen_exact).
2. **B2 falsado.** Datos de derivación *sintéticos* al 20.5% dañaron el binding
   inferable (−2pts vs control pareado) y provenance (−2.9pts). Aprendió su
   corpus (loss normal) pero no transfirió — el shift de superficie domina
   sobre la operación.
3. **Meseta.** Ambos brazos ≈ init → v5-1b está ~0.65 consistency a esta
   escala/receta; 5K steps de corpus real apenas lo mueven.
4. **La maquinaria del objetivo correcto YA EXISTE en `train_mdlm_moe_v2.py`
   y nunca se ha activado a escala:**
   - `--corrective_p` (V3.4): corrompe tokens **VISIBLES** con sentido
     científico (número→otro número, negación→polaridad inversa,
     dirección→antónimo, conectiva contrastiva↔causal) y supervisa predecir
     el token original. **Es literalmente el eval `consistency` convertido
     en objetivo de entrenamiento, sobre corpus real** — sin shift de
     distribución (los documentos son los de siempre, la corrupción es local).
   - `--contr_w` + `--contr_margin`/`--contr_p`: hinge in-loss
     `logp[real] − logp[mutado] ≥ margen` en posiciones mutables — alineación
     **directa** con la métrica pairwise, sin forward extra.
   - `--role_mask`/`--role_config`: masking ponderado por rol semántico.
   - `--loss_num_w`/`--loss_neg_w`: ponderación de loss por tipo de token.
   - `--mras_*`: masking adaptativo por dificultad (EMA de CE por token).
   - Estado comprobado en `runs/v5-1b/train.log`: `role_mask=False`,
     `corrective_p=0`, `contr_w=0` — **todos a cero**. El GA los dejó
     inactivos: se construyeron (V3.4/G10) cuando el fitness era `dense`,
     la métrica que no medía esta habilidad.

## 1. Lectura de la falsación

B2 distingue "más datos con la operación" de "el objetivo entrena la
operación". El corpus deriv SÍ contenía slots forzados, pero:

- La supervisión llegaba por *denoising aleatorio*: de cada doc deriv solo una
  fracción de steps enmascaraba el slot forzado con contexto visible — la
  señal se diluye en reconstrucción genérica.
- El texto templado (aun con anti-molde) desplazó la distribución → costo de
  interferencia sin beneficio.

→ **El eje "datos" no está falsado en abstracto; está falsado "más tokens
sintéticos con denoising random".** La forma barata de enseñar la operación
no es más corpus: es **cambiar qué se corrompe y qué se supervisa sobre el
corpus real que ya funciona**.

## 2. Árbol de hipótesis

### H-obj-A (favorita): corrective + hinge sobre corpus real

Activar en el run pareado:

```
--corrective_p 0.08  --corrective_boost 8.0 \
--contr_w 0.5 --contr_margin 2.0 --contr_p 1.0
```

- `corrective_p=0.08`: ~8% de los tokens VISIBLES se corrompen por ejemplo;
  con boost 8 la gran mayoría cae en slots informativos (números, negaciones,
  dirección, conectivas) — miles de instancias de "detecta el token que
  contradice el contexto" por step, sobre prosa real.
- `contr_w=0.5`: margen de 2 nats entre el real y su mutación en slots
  enmascarados — gradiente pairwise exacto, gratis en el mismo forward.
- Por qué puede ganar donde deriv falló: misma operación, **cero** shift de
  corpus; cada ejemplo de entrenamiento porta una señal de verificación.
- Riesgo conocido (V3.4-notes): la corrupción visible puede dañar fluidez si
  `p` es alto → 8% es dosis conservadora; si el gate es positivo pero débil,
  el siguiente punto es 0.15.

### H-obj-B: masking dirigido a slots (role_mask agresivo)

`role_mask` con `role_config` cargado sobre conectivas/verbos/números/
entidades. Difiere de A: cambia QUÉ se enmascara (denoising sigue siendo el
objetivo), no añade señal de verificación. Complementario con A más que
alternativo — pero si A muestra que la señal *corrective* es lo activo, B
aisla el componente "más práctica en slots" vs "supervisión de corrección".

### H-obj-C: verifier decoding (inferencia, sin entrenamiento)

Generar candidato → rescore slots en modo `consistency` → remuestrear los
peores. Costo: cero entrenamiento. Es el puente inmediato a "chatea con
sustancia": el modelo se auto-verifica al decodificar. Independiente de A/B —
puede correr sobre v5-1b ya. Útil aunque A falle.

### H-arq: arquitectura (diferida, no descartada)

- **Block-diffusion semi-autoregresivo**: generar por bloques izq→der con
  difusión interna — orden más compatible con chat; LLaDA-2 lo usa.
- **Cabeza verificadora auxiliar**: segunda salida "¿este token es correcto
  dado el contexto?" — ya cubierta funcionalmente por `contr_w` (mismo
  gradiente, sin params extra).
- **Deliberation/planning tokens**: estructura explícita de trazas — es C2
  (trace-SFT), no cambio de backbone.
- Veredicto: la arquitectura no es el primer sospechoso — el mismatch
  objetivo↔eval está demostrado; los levers existentes son más baratos que
  cualquier cambio estructural. Solo si A+B+C fallan en conjunto.

## 3. Plan de experimentos (ordenado por costo)

| exp | qué | costo | gate |
|---|---|---|---|
| **E-obj1** | brazo `b2-corr`: init v5-1b@g168001 + corpus `train_ids_b2base` (mismo stream que ctrl2) + `corrective_p 0.08 + contr_w 0.5`, 5K steps | ~1 día GPU | consistency-inf > ctrl2 +2σ; provenance ≥ ctrl2 −1σ |
| **E-obj2** | verifier decoding sobre v5-1b (re-score + resample de slots) en pares inferable | horas CPU/GPU | gen_exact +Δ>0 sin bajar consistencia |
| **E-obj3** | si E-obj1 positivo: dose-response (0.15) o +role_mask; si nulo: role_mask solo | ~1 día | idem |
| E-arq | solo si E-obj1–3 falsan | semanas | — |

### Configuración pareada (ya en marcha)

Desde **v5-1b@g168001** (mismo init para todos):

| brazo | corpus | extras | job |
|---|---|---|---|
| `b2-ctrl2` | b2base (420M) | — | 30070596 |
| `b2-qa` | b2qa = base + deriv_qa real 20.8% | — | 30070595 |
| `b2-corr` | b2base (420M) | corrective 0.08 + contr_w 0.5 | por lanzar |

El control `b2-ctrl2` sirve a los dos experimentos — comparación pareada
limpio: datos-reales (qa) vs objetivo (corr) vs nada (ctrl2).

**Criterio de falsación del eje objetivo** (pre-registrado): si `b2-corr` no
supera ctrl2 en +2σ inferable-L3 → los levers correctivos tampoco mueven la
habilidad → siguiente sospechoso = arquitectura (E-arq) o escala.

## 4. Notas de implementación

- `v5_1b_ddp.slurm` ahora acepta `EXTRA_ARGS` (propagado en resubmits).
- `--corrective_p` corrompe visibles *además* de enmascarar — la fracción
  enmascarada efectiva sube ~8pts: vigilar loss de arranque vs ctrl2.
- `contr_w` hinge usa solo posiciones enmascaradas mutables — inocuo donde
  no hay slots mutables.
- Los tres brazos evalúan en `pairs_L3_inferable` + `pairs_L3_prov` con
  consistency-sum y argmax generativo (misma eval que B2).
