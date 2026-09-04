# Experiments

Каталог содержит локальные protocol streams, checkpoints, WAV, таблицы, графики и
статистические отчёты. Большие воспроизводимые артефакты не коммитятся.

## Текущий эксперимент

Запустить paired stimulus/control corpus из общих equilibrated checkpoints:

```text
just causal-run
```

Команда создаёт 6 checkpoints, 30 ветвей и 24 causal pairs, затем рассчитывает
absolute/causal diagnostics и checkpoint-aware evidence.

Пересчитать статистику по готовым causal embeddings без новых симуляций:

```text
just causal-analyze
```

Основные результаты:

- `experiments/logs/controlled_equilibration_baseline.json` — общая сводка;
- `experiments/controlled_equilibration/baseline/causal_evidence.json` — accuracy,
  результаты по roots/checkpoints, bootstrap и permutation;
- `experiments/controlled_equilibration/baseline/causal_diagnostics.png` — variance
  decomposition и distance distributions.

Старые Stage 1 matrix, calibration и probe scripts сохранены только для
воспроизводимости уже полученных результатов и не являются текущим workflow.
