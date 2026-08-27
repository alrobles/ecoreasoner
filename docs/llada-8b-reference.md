# LLaDA-8B-Instruct como referencia del dLLM (Fase A "que sirva")

Fecha: 2026-08-27 · Estado: **REFERENCIA MEDIDA (1 vez, servidor lanzado en MI210)**

## Qué se hizo
Servir `GSAI-ML/LLaDA-8B-Instruct` (8B, masked-diffusion dLLM preentrenado) como
**referencia de calidad/throughput** para comparar contra nuestro dLLM-MoE propio (e2_v3, 733M).

## Resultado
- **Modelo cargado y sirviendo** vía `llada_serve_1gpu.slurm` (MI210 r06r30n01, port 8080).
- Gen de texto **coherente en español**. Ejemplo (40 tok): "La ecofisiología es la investigación de las relaciones entre la física y la ecología, y la interacción natural con el medio ambiente."
- **~9 tok/s** en 1 MI210 (bf16), model load ~54s.
- **Nota**: con `num_steps`=a `max_tokens`=128 generó solo 13 tokens — el sampling simple del slurm (greedy CSS por confianza) no es el algoritmo oficial de LLaDA (que usa remasking+CFG). Para un throughput fiable usar num_steps << token count y configurar bien. La referencia (que SÍ genera) está establecida.

## Problemas de versión resueltos (transformers 5.x vs remote code 4.x)
El entorno del sif usa transformers **5.15.1**, cuyo remote code de LLaDA (escrito para 4.x)
rompe en varios puntos. Se parchearon:
1. `PYTHONPATH` unbound en el slurm → `${PYTHONPATH:-}`.
2. `device_map="auto"` no soportado por LLaDAModelLM → quitar, usar `model.to(DEVICE)`.
3. `all_tied_weights_keys` no existe en modeling_llada → añadir `_tied_weights_keys = []` + property `all_tied_weights_keys` que devuelve `{}`.
4. `tie_weights()` no acepta kwargs en transformers 5 → firmar `tie_weights(self, *args, **kwargs)`.
5. `use_cache` no es atributo en configuration_llada → `self.use_cache = use_cache` explícito.
6. `mask_token` del tokenizer era None → usar `model.config.mask_token_id` (126336 = `<|mdm_mask|>`).
7. Doble forward `model(input_ids)` en sampling → un solo `out = model(input_ids)`.

Estos parches viven en el snapshot HF (los `.py` de remote code editados):
`/beegfs/a474r867/hf-cache/models--GSAI-ML--LLaDA-8B-Instruct/snapshots/08b.../configuration_llada.py` y `modeling_llada.py`.

## Cómo relanzar
```bash
# desde el cluster:
cd /beegfs/a474r867/ecoreasoner && sbatch scripts/llada_serve_1gpu.slurm
# verificar (en el nodo donde esté):
curl -s http://<nodo>:8080/v1/models
```

## Limpieza feliz
La **referencia queda medida**. Para no comprometer GPUs ni atender un servidor permanente,
NO está pensado como servicio 'always-on' — es un recurso de referencia para el benchmark del dLLM
(M2-A del protocolo): comparar (i) calidad vs e2-v3, (ii) throughput frente a AR core. Cuando haga
falta el fine-tune (3-30K ejemplos ecológicos → LoRA), se relanza en un MI210 libre.

## Estado job actual al cierre
- Job `llada-serve` (27469599) RUNNING en r06r30n01 (MI210). Se deja hasta que se complete la
  comparación M2 mini, salvo que se decida apagar antes (el usuario prefiere no dejar servidores permanentes).